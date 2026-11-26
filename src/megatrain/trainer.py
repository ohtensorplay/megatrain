# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Generic TensorPlay trainer: the *how* of a training step.

Owns the optimizer, FSDP/compile integration, gradient accumulation, frozen
teacher forwards and atomic checkpoint/resume. It deliberately knows nothing
about rollouts, rewards, advantages, importance ratios or policy versions; the
objective is injected per step as ``loss_fn``. RL frameworks subclass this
class and add their own ``update``/``set_version`` methods on top.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import tensorplay as tp

from .batch import TrainBatch, pack_samples, slice_batch
from .config import TrainerConfig
from .losses import TensorBatch, sft_loss
from .stateless_adam import StatelessAdam

MANIFEST = "manifest.json"


def _as_float(value: Any) -> float:
    """Normalize a diagnostic scalar (tensor or Python number)."""
    if hasattr(value, "detach"):
        return float(value.detach().flatten()[0])
    return float(value)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optimizer_state_to_mega(state: dict) -> dict:
    """Stringify integer parameter indices MEGA forbids as dict keys.

    ``AdamW.state_dict()`` keys its per-parameter moments by integer index and
    repeats those indices in ``param_groups[*]['params']``. MEGA requires string
    dict keys, so both are converted together; tensor data and hyperparameters
    pass through untouched.
    """
    return {
        "state": {str(index): value for index, value in state.get("state", {}).items()},
        "param_groups": [
            dict(group, params=[str(p) for p in group.get("params", [])])
            for group in state.get("param_groups", [])
        ],
    }


def _optimizer_state_from_mega(state: dict) -> dict:
    """Inverse of :func:`_optimizer_state_to_mega`."""
    return {
        "state": {int(index): value for index, value in state.get("state", {}).items()},
        "param_groups": [
            dict(group, params=[int(p) for p in group.get("params", [])])
            for group in state.get("param_groups", [])
        ],
    }


class BaseTrainer:
    """Real optimizer-step trainer on TensorPlay.

    ``policy`` must be a TensorPlay module whose ``__call__(input_ids)`` returns
    logits shaped ``[B, L, vocab]``; ``forward_fn`` can override the call when
    a model needs extra arguments (e.g. vision features). ``teacher_model``,
    when given, is always frozen, never optimized and never checkpointed.

    Subclasses (e.g. an RL trainer) can override ``tensor_batch_cls``,
    ``_slice_batch`` and ``_keep_tokens`` to carry extra fields and to define
    which tokens contribute to the loss, without copying the training loop.
    """

    tensor_batch_cls: type = TensorBatch

    def __init__(
        self,
        policy: Any,
        config: TrainerConfig | None = None,
        *,
        optimizer: Any | None = None,
        forward_fn: Callable[[Any, Any], Any] | None = None,
        teacher_model: Any | None = None,
        teacher_fn: Callable[[Any, Any], Any] | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.config.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if self.config.max_grad_norm is not None and self.config.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 when set")
        if self.config.optimizer_kind not in ("adamw", "stateless_adam"):
            raise ValueError(f"unknown optimizer_kind: {self.config.optimizer_kind}")
        if self.config.amp_dtype not in ("float16", "bfloat16"):
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
        if self.config.seed is not None:
            random.seed(self.config.seed)
            tp.manual_seed(self.config.seed)
            try:
                import numpy as np

                np.random.seed(self.config.seed)
            except ImportError:  # pragma: no cover
                pass
        self.policy = policy
        # Forward hooks receive the full TensorBatch so packed/varlen models can
        # use attention_mask, position_ids and cu_seqlens, not just input_ids.
        self.forward_fn = forward_fn or (lambda model, batch: model(batch.input_ids))
        self.teacher_model = teacher_model
        self.teacher_fn = teacher_fn or (lambda model, batch: model(batch.input_ids))
        self.scheduler = scheduler
        self._device_type = (
            "cuda" if str(self.config.device).startswith("cuda") else str(self.config.device)
        )
        params = list(policy.parameters())
        self._params = params
        if optimizer is None and self.config.optimizer_kind == "stateless_adam":
            optimizer = StatelessAdam(
                params,
                lr=self.config.learning_rate,
                betas=self.config.adam_betas,
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay,
                bias_correction=self.config.stateless_bias_correction,
            )
        self.optimizer = optimizer or tp.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            betas=self.config.adam_betas,
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
            **self._adam_kwargs(),
        )
        self.step_count = 0
        self._amp_enabled = bool(self.config.amp) and self._device_type != "cpu"
        self._amp_dtype = getattr(tp, self.config.amp_dtype) if self._amp_enabled else None
        # GradScaler is only needed for fp16. bf16 has fp32 dynamic range,
        # so skipping scale/unscale removes several kernel launches per step.
        self._scaler_enabled = self._amp_enabled and self.config.amp_dtype == "float16"
        self.scaler = tp.amp.GradScaler(self._device_type, enabled=self._scaler_enabled)
        self._compiled: Callable | None = None
        if self.config.fsdp:
            from tensorplay.distributed.fsdp import fully_shard

            self.policy = fully_shard(policy)
        if self.config.compile_policy:
            self._compiled = tp.compile(
                self._policy_logits,
                dynamic=self.config.compile_dynamic,
                mode="reduce-overhead",
            )
        if self.config.checkpoint_dir:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # -- subclass hooks --------------------------------------------------
    def _adam_kwargs(self) -> dict[str, Any]:
        """Optimizer kwargs for the built-in AdamW.

        ``fused=True`` selects TP's single-kernel multi-tensor AdamW. On CUDA it
        is measurably cheaper per step once micro-batches are large enough to
        amortize launches (2.83 -> 2.74 ms at 160M); ``capturable=True`` keeps
        the step legal under CUDA-graph capture. Never request it off CUDA.
        """
        if self.config.fused_optimizer and self._device_type == "cuda":
            return {"fused": True, "capturable": True}
        return {}

    def _slice_batch(self, batch: TrainBatch, start: int, stop: int):
        """Row slicing hook; RL subclasses preserve their extra batch fields."""
        return slice_batch(batch, start, stop)

    def _needs_auto_teacher(self) -> bool:
        """Hook: whether :meth:`train_step` should compute teacher log-probs.

        An RL subclass whose objective only needs a teacher for some algorithms
        overrides this to avoid a wasted frozen forward on every other update.
        """
        return True

    def checkpoint_artifacts(self) -> dict[str, Any]:
        """Named objects saved next to ``model``/``optimizer`` (e.g. a critic).

        Values are serialized with ``tp.save`` and registered in the manifest,
        so checksum verification and pruning apply to them too.
        """
        return {}

    def load_checkpoint_artifacts(self, root: Path, files: dict[str, str]) -> None:
        """Restore artifacts produced by :meth:`checkpoint_artifacts`.

        ``files`` maps every artifact name in the manifest to its digest.
        """
        return None

    def _keep_tokens(self, batch: TrainBatch) -> int:
        """Hook defining the normalization unit (target tokens by default)."""
        return batch.keep_tokens()

    def _keep_tokens_tensors(self, tensors) -> int:
        """Tensor-path counterpart of :meth:`_keep_tokens`."""
        return int(tensors.loss_mask[:, 1:].sum().flatten()[0])

    def _slice_tensor_batch(self, tensors, start: int, stop: int):
        """Zero-copy row slicing for already-materialized device tensors."""
        sliced = copy.copy(tensors)
        rows = int(tensors.input_ids.shape[0])
        for name, value in vars(tensors).items():
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) > 0 and shape[0] == rows:
                setattr(sliced, name, value[start:stop])
        return sliced

    # -- forward helpers -------------------------------------------------
    def _policy_logits(self, input_ids: Any) -> Any:
        """Backward-compatible call: logits from raw input ids."""
        return self.policy(input_ids)

    def _forward(self, tensors: TensorBatch) -> Any:
        return self.forward_fn(self.policy, tensors)

    @staticmethod
    def _logprobs(logits: Any, input_ids: Any) -> Any:
        """Per-token log-probabilities from ``[B, L, vocab]`` logits."""
        from .losses import token_logprobs

        return token_logprobs(logits, input_ids)

    def _teacher_logprobs(self, tensors: TensorBatch) -> Any:
        """Frozen teacher log-probs for the whole batch, shape ``[B, L]``."""
        return self._logprobs(self.teacher_fn(self.teacher_model, tensors), tensors.input_ids)

    def _no_sync(self):
        cm = getattr(self.policy, "no_sync", None)
        return cm() if cm is not None else nullcontext()

    def state_dict(self) -> dict:
        """Policy state for weight export/sync hooks in a downstream framework."""
        return self.policy.state_dict()

    # -- training --------------------------------------------------------
    def train_step(
        self,
        batch: TrainBatch,
        *,
        loss_fn: Callable = sft_loss,
        loss_kwargs: dict[str, Any] | None = None,
        aux: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Run one optimizer step over ``batch`` with an injected objective.

        ``loss_fn(model_output, tensor_batch, **kwargs)`` must return
        ``(loss, diagnostics)`` and normalize the loss over its own target
        tokens. Microbatches are then weighted by their share of target tokens
        so accumulation reproduces a single full batch exactly. ``model_output``
        is whatever the model returns: logits for :func:`~megatrain.sft_loss`,
        or hidden states for :func:`~megatrain.sft_loss_with_head`, which fuses
        the vocabulary projection into the loss.

        ``aux`` carries per-row tensors (e.g. ``teacher_logprobs``) that are
        sliced along dimension 0 per microbatch; non-tensor values pass through.
        ``dry_run`` keeps gradients for auditing and skips the optimizer step.
        """
        cfg = self.config
        device = cfg.device
        # Zero-copy path: a data plane can hand us device tensors directly,
        # avoiding a host-side rebuild of every batch on every step.
        tensor_input = hasattr(batch, "input_ids") and hasattr(
            getattr(batch, "input_ids", None), "shape"
        )
        batch_size = int(batch.input_ids.shape[0]) if tensor_input else batch.size
        n = min(cfg.gradient_accumulation_steps, batch_size)
        size = (batch_size + n - 1) // n
        spans = [
            (start, min(start + size, batch_size)) for start in range(0, batch_size, size)
        ]
        micros: list = []
        micro_tensors: list = []
        if tensor_input:
            micro_tensors = [self._slice_tensor_batch(batch, s, e) for s, e in spans]
            total_keep = self._keep_tokens_tensors(batch)
            micro_keeps = [self._keep_tokens_tensors(m) for m in micro_tensors]
        else:
            micros = [self._slice_batch(batch, s, e) for s, e in spans]
            total_keep = self._keep_tokens(batch)
            micro_keeps = [self._keep_tokens(m) for m in micros]

        kwargs = dict(loss_kwargs or {})
        if loss_fn is sft_loss and "ce_chunk_tokens" not in kwargs and cfg.ce_chunk_tokens:
            kwargs["ce_chunk_tokens"] = cfg.ce_chunk_tokens
        extra = dict(aux or {})
        if (
            self.teacher_model is not None
            and self._needs_auto_teacher()
            and "teacher_logprobs" not in extra
            and "teacher_logprobs" not in kwargs
        ):
            with tp.no_grad():
                teacher_input = batch if tensor_input else self.tensor_batch_cls(batch, device=device)
                extra["teacher_logprobs"] = self._teacher_logprobs(teacher_input)

        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        diag_accum: dict[str, float] = {}
        for index, ((start, stop), keep) in enumerate(zip(spans, micro_keeps)):
            tensors = (
                micro_tensors[index]
                if tensor_input
                else self.tensor_batch_cls(micros[index], device=device)
            )
            with tp.amp.autocast(
                self._device_type, dtype=self._amp_dtype, enabled=self._amp_enabled
            ):
                logits = (
                    self._compiled(tensors.input_ids)
                    if self._compiled
                    else self._forward(tensors)
                )
            fn_kwargs = dict(kwargs)
            for key, value in extra.items():
                shape = getattr(value, "shape", None)
                fn_kwargs[key] = (
                    value[start:stop]
                    if shape is not None and len(shape) > 0 and shape[0] == batch_size
                    else value
                )
            accumulation_cm = self._no_sync() if index < len(spans) - 1 else nullcontext()
            weight = keep / max(total_keep, 1)
            with accumulation_cm:
                loss, diag = loss_fn(logits, tensors, **fn_kwargs)
                weighted = loss * weight
                (self.scaler.scale(weighted) if self._scaler_enabled else weighted).backward()
            loss_sum += _as_float(loss) * weight
            for key, value in diag.items():
                diag_accum[key] = diag_accum.get(key, 0.0) + _as_float(value) * weight
        if self._scaler_enabled:
            self.scaler.unscale_(self.optimizer)
        grad_norm = None
        if cfg.max_grad_norm is not None:
            grad_norm = tp.nn.utils.clip_grad_norm_(
                [
                    param
                    for group in self.optimizer.param_groups
                    for param in group["params"]
                ],
                cfg.max_grad_norm,
            )
        if not dry_run:
            if self._scaler_enabled:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step_count += 1
        metrics = {f"train/{key}": value for key, value in diag_accum.items()}
        metrics["train/loss"] = loss_sum
        metrics["train/step"] = float(self.step_count)
        if grad_norm is not None:
            metrics["train/grad_norm"] = float(grad_norm.flatten()[0])
        if self._scaler_enabled:
            metrics["train/scale"] = float(self.scaler.get_scale())
        return metrics

    # -- checkpoint / resume --------------------------------------------
    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra: dict | None = None,
        **fields: Any,
    ) -> Path:
        """Atomically publish a resumable checkpoint.

        ``extra`` is a generic sidecar; keyword ``fields`` are written at the
        manifest top level so downstream RL trainers can keep version/reward
        state without changing this format. Re-saving an existing version is a
        no-op rather than a destructive overwrite.
        """
        root = self.config.checkpoint_dir or Path(path).parent
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        target = root / Path(path).name
        if target.exists():
            return target
        staging = root / f".{target.name}.tmp"
        if staging.exists():
            _rmtree(staging)
        staging.mkdir(parents=True)
        tp.save(self.policy.state_dict(), staging / "model.mega")
        tp.save(_optimizer_state_to_mega(self.optimizer.state_dict()), staging / "optimizer.mega")
        files = ["model.mega", "optimizer.mega"]
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            tp.save(self.scheduler.state_dict(), staging / "scheduler.mega")
            files.append("scheduler.mega")
        artifacts = self.checkpoint_artifacts()
        for name, artifact in artifacts.items():
            if name in files:
                raise ValueError(f"checkpoint artifact shadows a core file: {name}")
            tp.save(artifact, staging / name)
            files.append(name)
        payload = {
            "format": 1,
            "step": int(self.step_count),
            "extra": extra or {},
            "rng": _rng_state(),
            "files": {name: _digest(staging / name) for name in files},
        }
        payload.update(fields)
        (staging / MANIFEST).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(staging, target)
        pointer = root / "LATEST"
        tmp_pointer = root / ".LATEST.tmp"
        tmp_pointer.write_text(target.name, encoding="utf-8")
        os.replace(tmp_pointer, pointer)
        self._prune(root)
        return target

    def load_checkpoint(self, path: str | Path) -> dict:
        """Verify digests, restore weights/optimizer/RNG, return manifest fields."""
        given = Path(path)
        if given.is_file():
            given = given.parent / given.read_text(encoding="utf-8").strip()
        root = given
        payload = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        if payload.get("format") != 1:
            raise ValueError(f"unsupported checkpoint format: {payload.get('format')}")
        for name, expected in payload["files"].items():
            actual = _digest(root / name)
            if actual != expected:
                raise ValueError(f"checksum mismatch for {name}: {actual} != {expected}")
        self.policy.load_state_dict(tp.load(root / "model.mega"))
        if "optimizer.mega" in payload["files"]:
            saved = tp.load(root / "optimizer.mega")
            self.optimizer.load_state_dict(_optimizer_state_from_mega(saved))
        else:
            raise ValueError("checkpoint has no optimizer artifact")
        if self.scheduler is not None and "scheduler.mega" in payload["files"]:
            self.scheduler.load_state_dict(tp.load(root / "scheduler.mega"))
        self.load_checkpoint_artifacts(root, payload["files"])
        _restore_rng(payload["rng"])
        self.step_count = int(payload["step"])
        result = dict(payload)
        result.pop("format", None)
        result.pop("files", None)
        return result

    def _prune(self, root: Path) -> None:
        keep = self.config.max_checkpoints
        if keep < 1:
            return
        dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        dirs.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for stale in dirs[keep:]:
            _rmtree(stale)

    # -- production loop -------------------------------------------------
    def fit(
        self,
        samples: Any,
        *,
        max_tokens: int,
        epochs: int = 1,
        alignment: int = 1,
        shuffle: bool = True,
        seed: int | None = None,
        checkpoint_every: int | None = None,
        checkpoint_dir: str | Path | None = None,
        on_step: Callable[[dict], None] | None = None,
    ):
        """Streaming pretraining/SFT loop over tokenized samples.

        Samples are split into aligned windows and packed by token budget each
        epoch; checkpoints are published at a fixed optimizer-step cadence and
        are resumable via :meth:`load_checkpoint` (which restores the RNG and
        optimizer/scheduler state). Yields one metrics dict per step.
        """
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        root = Path(checkpoint_dir or self.config.checkpoint_dir or ".")
        for epoch in range(epochs):
            batches = pack_samples(
                samples,
                max_tokens,
                alignment=alignment,
                shuffle=shuffle,
                seed=None if seed is None else seed + epoch,
            )
            for batch in batches:
                metrics = self.train_step(batch)
                metrics["train/epoch"] = float(epoch)
                if on_step is not None:
                    on_step(metrics)
                if checkpoint_every and self.step_count % checkpoint_every == 0:
                    self.save_checkpoint(root / f"step-{self.step_count}", epoch=epoch)
                yield metrics


def _rng_state() -> dict:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    py_state = random.getstate()
    state: dict[str, Any] = {
        "python": [list(py_state[1]), py_state[2]],
        "tensorplay": [int(v) for v in tp.get_rng_state().flatten().tolist()],
    }
    if np is not None:
        state["numpy"] = [int(v) for v in np.random.get_state()[1].tolist()]
    return state


def _restore_rng(state: dict) -> None:
    py = state["python"]
    random.setstate((3, tuple(py[0]), py[1]))
    tp.set_rng_state(tp.tensor(state["tensorplay"], dtype=tp.uint8))
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return
    if "numpy" in state:
        arr = np.random.get_state()
        np.random.set_state((arr[0], np.array(state["numpy"], dtype=np.uint32), arr[2], arr[3], arr[4]))


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


__all__ = ["BaseTrainer", "MANIFEST"]
