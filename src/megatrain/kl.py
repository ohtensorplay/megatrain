"""Sampled-token KL estimators.

Adapted from megaRL's ``kl.py`` (Apache-2.0). Every estimator is an element-wise
function of ``d = log p_new - log p_ref``; callers reduce it with their own loss
mask. The direction matches the pinned references (slime ``ppo_utils.py``,
AReaL ``data.py``, verl ``core_algos.py``):

- ``k1``         ``d``
- ``k2``         ``0.5 * d**2``
- ``k3``         ``exp(-d) + d - 1``   (Schulman low-variance)
- ``low_var_kl`` ``k3`` clamped to ±10
"""

from __future__ import annotations

import tensorplay as tp

KL_ESTIMATORS = ("k1", "k2", "k3", "low_var_kl")

__all__ = ["KL_ESTIMATORS", "approx_kl"]


def approx_kl(logp, ref_logp, *, kl_type: str = "k3"):
    """Element-wise KL estimate for ``d = logp - ref_logp``."""
    if kl_type not in KL_ESTIMATORS:
        raise ValueError(
            f"unknown kl_type: {kl_type}; choose one of {', '.join(KL_ESTIMATORS)}"
        )
    delta = logp - ref_logp
    if kl_type == "k1":
        return delta
    if kl_type == "k2":
        return 0.5 * delta * delta
    safe_delta = delta.clamp(min=-20.0, max=20.0)
    value = tp.exp(-safe_delta) + safe_delta - 1.0
    return value.clamp(min=-10.0, max=10.0) if kl_type == "low_var_kl" else value
