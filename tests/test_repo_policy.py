# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""The repository's own policy tools: commit schema, PR description, tree hygiene.

All three run from CI and from pre-commit, so they are exercised here through
their command line rather than by importing their internals.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMIT_SCHEMA = ROOT / "tools" / "commit_schema.py"
CHECK_PR_BODY = ROOT / "tools" / "check_pr_body.py"
PUBLIC_CHECK = ROOT / "tools" / "check_public.py"

GOOD_BODY = """## What and why

Fuse the vocabulary loss so the logits buffer is written once.

## Evidence

`bench/fused_ce.sh` on a quiet 3090: 78,264 -> 96,935 tok/s, step peak 4,280
-> 2,232 MiB, same loss.

## Checklist

- [x] I, the PR author, have personally reviewed every line of this PR.
- [x] `PYTHONPATH=src python -m pytest tests -q` passes.
"""


def run(*args: str, stdin: str | None = None, env: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
    )


def schema(*args: str):
    return run(str(COMMIT_SCHEMA), *args)


def body_check(body: str, env: dict[str, str] | None = None):
    if env is not None:
        return run(str(CHECK_PR_BODY), env=env)
    return run(str(CHECK_PR_BODY), stdin=body)


# --- commit schema -------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "feat(trainer): add the fused vocabulary loss",
        "fix: keep the zero-gradient rows at zero",
        "bench: record the 474M ablation",
        "docs: explain the quiet-window guard",
    ],
)
def test_commit_schema_accepts_the_house_style(message):
    result = schema("--message", message)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize(
    "message",
    [
        "trainer: add a hook",  # no type
        "feature(trainer): add a hook",  # unknown type
        "feat(Trainer): add a hook",  # upper-case scope
        "feat(trainer): add a hook.",  # trailing period
        "feat(trainer):" + "x" * 100,  # header too long
        "feat(): add a hook",  # empty scope
    ],
)
def test_commit_schema_rejects_malformed_headers(message):
    result = schema("--message", message)
    assert result.returncode == 1
    assert "expected: type(scope): subject" in result.stderr


def test_breaking_change_requires_its_footer():
    assert schema("--message", "chore!: drop the legacy CE path").returncode == 1
    assert schema("--message", "feat!: drop the legacy path").returncode == 1
    ok = schema(
        "--message",
        "feat!: drop the legacy path\n\nBREAKING CHANGE: sft_loss no longer accepts logits.",
    )
    assert ok.returncode == 0, ok.stderr


def test_signoff_is_enforced_when_asked(tmp_path):
    unsigned = tmp_path / "msg"
    unsigned.write_text("fix: keep the rows at zero\n")
    assert schema("--commit-msg-file", str(unsigned)).returncode == 0
    assert schema("--commit-msg-file", str(unsigned), "--require-signoff").returncode == 1

    signed = tmp_path / "signed"
    signed.write_text("fix: keep the rows at zero\n\nSigned-off-by: A Person <a@example.org>\n")
    assert schema("--commit-msg-file", str(signed), "--require-signoff").returncode == 0


def test_comment_lines_in_a_commit_message_are_ignored(tmp_path):
    message = tmp_path / "msg"
    message.write_text("# a comment\n\nfix: keep the rows at zero\n\n# another\n")
    result = schema("--commit-msg-file", str(message))
    assert result.returncode == 0, result.stderr


def test_merge_and_fixup_subjects_are_exempt():
    for header in ("Merge pull request #7 from x/y", "fixup! feat: add a hook"):
        assert schema("--message", header).returncode == 0


def test_range_validates_every_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str):
        return subprocess.run(
            ["git", "-c", "user.name=A", "-c", "user.email=a@example.org", *args],
            cwd=repo,
            capture_output=True,
            text=True,
        )

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-s", "-m", "feat: add the core")
    git("commit", "-q", "--allow-empty", "-s", "-m", "docs: explain it")

    result = subprocess.run(
        [sys.executable, str(COMMIT_SCHEMA), "--range", "HEAD~1..HEAD", "--require-signoff"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    git("commit", "-q", "--allow-empty", "-s", "-m", "benchmark: bad header")
    broken = subprocess.run(
        [sys.executable, str(COMMIT_SCHEMA), "--range", "HEAD~1..HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert broken.returncode == 1
    assert "unknown type" in broken.stderr


# --- pull-request description -------------------------------------------


def test_pr_body_accepts_a_complete_description():
    result = body_check(GOOD_BODY)
    assert result.returncode == 0, result.stderr


def test_pr_body_rejects_a_missing_or_unticked_review_box():
    missing = body_check(
        GOOD_BODY.replace(
            "- [x] I, the PR author, have personally reviewed every line of this PR.\n", ""
        )
    )
    assert missing.returncode == 1
    assert "missing the self-review checkbox" in missing.stderr

    unticked = body_check(GOOD_BODY.replace("- [x]", "- [ ]", 1))
    assert unticked.returncode == 1
    assert "not ticked" in unticked.stderr


@pytest.mark.parametrize("heading", ["What and why", "Evidence", "Checklist"])
def test_pr_body_requires_each_heading(heading):
    broken = GOOD_BODY.replace(f"## {heading}\n", "")
    result = body_check(broken)
    assert result.returncode == 1
    assert f"missing the '## {heading}'" in result.stderr


def test_pr_body_requires_actual_evidence():
    evidence = (
        "`bench/fused_ce.sh` on a quiet 3090: 78,264 -> 96,935 tok/s, "
        "step peak 4,280\n-> 2,232 MiB, same loss."
    )
    empty = GOOD_BODY.replace(evidence, "")
    assert body_check(empty).returncode == 1

    placeholder = GOOD_BODY.replace(evidence, "TBD")
    assert body_check(placeholder).returncode == 1
    assert "placeholder" in body_check(placeholder).stderr


def test_pr_body_ignores_markup_that_only_looks_like_evidence():
    # A ticked box and a heading inside a fence or a comment must not count.
    fenced = GOOD_BODY.replace("## Evidence", "<!-- ## Evidence -->")
    assert body_check(fenced).returncode == 1

    hidden_box = GOOD_BODY.replace(
        "- [x] I, the PR author, have personally reviewed every line of this PR.",
        "```\n- [x] I, the PR author, have personally reviewed every line of this PR.\n```",
    )
    assert body_check(hidden_box).returncode == 1


def test_pr_body_can_come_from_the_environment():
    env = dict(os.environ, PR_BODY=GOOD_BODY)
    result = body_check("", env=env)
    assert result.returncode == 0, result.stderr


def test_template_matches_the_checker():
    template = (ROOT / ".github" / "pull_request_template.md").read_text()
    for heading in ("What and why", "Evidence", "Checklist"):
        assert f"## {heading}" in template
    assert "I, the PR author, have personally reviewed every line of this PR." in template


# --- public tree hygiene --------------------------------------------------


def public_check(*paths: str):
    return subprocess.run(
        [sys.executable, str(PUBLIC_CHECK), *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_the_tracked_tree_is_publishable():
    result = public_check()
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_check_accepts_runtime_derived_paths(tmp_path):
    path = tmp_path / "portable.sh"
    path.write_text(
        "SCRATCH=${SCRATCH:-$(mktemp -d)}\n"
        "SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')/nvidia\n"
    )
    result = public_check(str(path))
    assert result.returncode == 0, result.stderr


def test_public_check_rejects_machine_paths_and_private_names(tmp_path):
    # Both needles are assembled from fragments so this test file stays clean.
    machine_path = "/" + "home" + "/someone/project"
    private_name = "re" + "fs"
    path = tmp_path / "bad.md"
    path.write_text(f"checkout at {machine_path} under {private_name}/\n")
    result = public_check(str(path))
    assert result.returncode == 1
    assert "machine-specific absolute path" in result.stderr
    assert "private checkout name" in result.stderr


def test_private_notes_are_ignored_by_git():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "devdocs/local/environment.md"],
        cwd=ROOT,
    )
    assert result.returncode == 0
