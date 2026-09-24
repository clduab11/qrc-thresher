#!/usr/bin/env bash
# CP4b.1 commit plan (v3) — referee script. Run from the repo root on the host with Git for
# Windows' bash:   & 'C:\Program Files\Git\bin\bash.exe' commit_plan_v3.sh
# (not WSL: its git sees CRLF/mode noise on /mnt/c).
#
# Base: ee227dd, the CP4b tip on origin/refactor/2026-09. Commits the builder's CP4b.1 working
# tree in the three groups of the CP4b.1 report §8, staging only paths git reports as changed.
# Never pushes, never amends, never creates a branch; stops before committing anything if a
# registered config or a CP3b-era file changed; stops with earlier commits standing on any path
# it cannot assign. Safe to rerun.
set -euo pipefail

BASE_TIP=ee227dd
BRANCH=refactor/2026-09
TRAILER_1='The builder wrote this in its sandbox. The referee reviewed and committed it.'
TRAILER_2='Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>'

die() { printf 'STOP: %s\n' "$*" >&2; exit 1; }

# ---- guards -----------------------------------------------------------------------------------
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "not on $BRANCH"
BASE_FULL=$(git rev-parse --verify "$BASE_TIP^{commit}" 2>/dev/null) || die "commit $BASE_TIP not found; git fetch origin $BRANCH first"
if ! git merge-base --is-ancestor "$BASE_FULL" HEAD; then
  die "HEAD $(git rev-parse --short HEAD) does not contain $BASE_TIP (the CP4b tip). Fetch and
      fast-forward first: git fetch origin && git reset --soft origin/$BRANCH"
fi
[ -z "$(git diff --cached --name-only)" ] || die "the index is not empty"
if git ls-files --error-unmatch docs/REFACTOR_BRIEF_2026-09.md >/dev/null 2>&1; then
  die "docs/REFACTOR_BRIEF_2026-09.md is tracked; it must stay excluded"
fi

# ---- the changed set (modified, added, untracked), minus the commit-plan files themselves -------
declare -A PENDING=()
while IFS= read -r line; do
  [ -n "$line" ] || continue
  path=${line:3}
  path=${path#\"}; path=${path%\"}
  case "$path" in
    commit_plan*.sh|commit_plan*.ps1|commit_msgs/*|docs/REFACTOR_BRIEF_2026-09.md) continue ;;
  esac
  PENDING["$path"]=1
done < <(git status --porcelain=v1 --untracked-files=all)

if [ ${#PENDING[@]} -eq 0 ]; then
  if [ "$(git rev-parse HEAD)" = "$BASE_FULL" ]; then
    die "nothing to commit: the working tree equals $BASE_TIP. Is the CP4b.1 tree in this checkout?"
  fi
  printf 'Nothing left to commit. Commits on top of %s:\n' "$BASE_TIP"
  git log --oneline "$BASE_FULL"..HEAD
  printf '\nNext (not run by this script):\n'
  printf '  uv run ruff check src tests; uv run pytest -q      # expect 438 passed, 2 deselected\n'
  printf '  git push -u origin %s\n' "$BRANCH"
  exit 0
fi
printf 'Changed relative to HEAD (%d paths):\n' "${#PENDING[@]}"
printf '  %s\n' "${!PENDING[@]}" | sort

# Registered files: a change here is a protocol change, never auto-committed.
for p in configs/alpha_lite.yaml configs/comparative.yaml configs/gates/COMPARATIVE.v1.yaml \
         configs/windowed_w2.yaml configs/windowed_w4.yaml configs/gates/G0.7.v1.yaml \
         pyproject.toml uv.lock; do
  [ -z "${PENDING[$p]:-}" ] || die "$p changed (registered or locked file). Review before anything is
      committed. Nothing was committed."
done
# CP3b-era files the builder does not touch; a change here needs a human look first.
for p in src/qrc_thresher/plugins/builtin.py src/qrc_thresher/reservoirs/ablations.py \
         tests/test_windowed_parity_known_answer.py tests/test_preregistration.py \
         tests/test_gate_g07.py; do
  [ -z "${PENDING[$p]:-}" ] || die "$p shows as changed; the builder flagged it as untouched.
      Decide before committing. Nothing was committed."
done

# ---- one group = one commit; only pending paths are staged; an empty group is skipped ---------
commit_group() {
  local subject=$1 body=$2; shift 2
  local staged=()
  for p in "$@"; do
    if [ -n "${PENDING[$p]:-}" ]; then
      git add -- "$p"; staged+=("$p"); unset 'PENDING[$p]'
    fi
  done
  if [ ${#staged[@]} -eq 0 ]; then
    printf 'skip (no changes): %s\n' "$subject"; return
  fi
  git commit --quiet -m "$subject" -m "$body" -m "$TRAILER_1" -m "$TRAILER_2"
  printf 'committed %s: %s\n' "$(git rev-parse --short HEAD)" "$subject"
  printf '    %s\n' "${staged[@]}"
}

S=src/qrc_thresher

commit_group \
'integrity: verified design hash, structural train_cv, loud runs.csv' \
'CP4b.1, commit 1 of 3 (D016 A1–A5). Two referee passes over the CP4b
modules (the PI on gates/comparative.py and metrics/paired.py; an
adversarial review of the rest) found no verdict-affecting defect and
these provenance gaps, each closed with a test:

- A1: the deployed QRC equals the validated one, by hash.
  deploy._verify_design_hash rebuilds design(pair) from the record and
  refuses it (DesignHashMismatch, naming the pair and both hashes)
  unless the rebuilt circuit hash is the record'"'"'s; the matched ablation
  is built only after that check. gates/g07.tuned_feature_map does the
  same and records the built hash with hash_verified: true; a refused
  record makes G0.7 INSUFFICIENT with nothing written. ESN and RKS
  already refused a mismatch.
- A2: selection_scope train_cv is structural. select_configuration
  receives only the training rows (candidates built on u[:train_end],
  targets[:train_end]); no train_end argument, no assert. Poisoning
  every row at or beyond train_end with NaN leaves winners, scores and
  record_sha256 unchanged. The record carries selection_rows.
- A3: a failed runs.csv write fails the run (RunsCsvWriteError naming
  the path; runs.csv is written before sqlite so a failure leaves no
  row anywhere); run, ablation and baseline exit 1.
- A4: a missing record, an absent pair or a hash mismatch is resolved
  for every pair before the first run (deploy.resolve_deployments); the
  writers exit 1 and write nothing. The former design=default /
  circuit_hash=n/a failure rows no longer exist.
- A5: the family JSON'"'"'s git_commit is run_manifest._git_commit_hash(),
  with -dirty when the tree has changes.

tuning.py in this commit also carries B6'"'"'s tune_all/tune_handler and
C8'"'"'s task_data guard (one file; see commit 2). tests/test_tuning.py:
the run paths now refuse a tuning config without its record (exit 1,
TuningRecordMissing, no runs.csv) instead of writing failure rows;
tests/test_g07_wiring.py: designs carry hash_verified. The suite is
green with commits 1 and 2 together (the new test file in commit 2
exercises both).' \
$S/deploy.py $S/engine.py $S/commands/run.py $S/commands/ablation.py $S/commands/baseline.py \
$S/commands/gate.py $S/gates/g07.py $S/gates/comparative.py $S/proof/run_manifest.py $S/db.py \
$S/baselines/esn.py $S/tuning.py tests/test_tuning.py tests/test_g07_wiring.py

commit_group \
'tune: one sweep id for all tasks; strict configs; hygiene (B6, C7-C9)' \
'CP4b.1, commit 2 of 3 (D016 B6, C7–C9).

- B6: `qrc-thresher tune --config FILE` (TASK omitted) tunes stm,
  parity and narma in one invocation and stamps one sweep_id across the
  three records (each keeps its own record_sha256); `tune TASK` remains
  for reruns. The family evaluator echoes one sweep_id when the records
  agree. Coinciding designs (design_STM(pair) == design_parity(pair))
  write two parity rows with one hash, which collapse as exact reruns
  when their values agree and make the member INSUFFICIENT, naming the
  pair, when they differ; tested both ways. Supersedes D015'"'"'s per-record
  stamps for the registered run; the {task: stamp} map stays as the
  fallback.
- C7: one evaluated family per synthetic sweep shared across
  TestFamilyEvaluation (module-scoped fixture; no assertion changed).
- C8: every config model forbids unknown keys; STM requires
  task.delay_max and parity task.parity_window (no silent K = 20 or
  d = 3); AblationConfig drops random_features (RKS is a baseline).
- C9: task_names.parse_task_name raises on an unknown name;
  --design default / --design-task without a tuning block is an error;
  gate family and gate G1..G4/G2.5 require an explicit --config;
  summary writes n/a for an empty group.

tests/test_cp4b1_integrity.py (34 tests) covers A1–A5, B6, C8 and C9;
tests/test_comparative_family.py gains the shared fixture and the
coinciding-designs test. Sandbox: 438 passed, 2 deselected; ruff clean.' \
$S/cli.py $S/config.py $S/task_names.py $S/commands/summary.py \
tests/test_cp4b1_integrity.py tests/test_comparative_family.py

commit_group \
'docs: D016, the parity-accuracy disclosure, BUILD_SPEC v1.1 and README' \
'CP4b.1, commit 3 of 3. docs/DECISIONS.md: D016 appended (D015 untouched)
— the PI accepts D015 as written with its first item superseded by B6;
records A1–A5, B6, C7–C9; the cross-environment reproduction (the CP3
G0.7 v1 ESN findings reproduce to every printed digit on Linux /
Python 3.11 from the same uv.lock against the sandbox'"'"'s Windows / 3.13;
the CP4b suite is 403 passed / 2 deselected in three environments:
sandbox 197 s, host 130 s, Linux 110 s); and the parity-accuracy
disclosure with its derivation (an additive readout'"'"'s thresholded
accuracy on XOR ranges 0.25–0.75 by cell geometry; D010 recorded the
upper tail, CP3'"'"'s esn_linear at task_seed 43 (0.32) is the lower tail;
the permutation null, the paired comparisons and the G2 floor make it
harmless). docs/METHODOLOGY.md §1.2: the disclosure paragraph.
docs/BUILD_SPEC.md: §17 and Appendix F rewritten to the CP4 commands
(no --seed; ablation NAME TASK; tune; baseline; gate family with an
explicit --config), changelog v1.1. README: quick start likewise; no
--seed remains in docs or README.' \
docs/DECISIONS.md docs/METHODOLOGY.md docs/BUILD_SPEC.md README.md

# ---- anything left is unassigned: report and stop, so a human decides -------------------------
if [ ${#PENDING[@]} -gt 0 ]; then
  printf '\nUNASSIGNED paths (not committed):\n'
  printf '  %s\n' "${!PENDING[@]}" | sort
  die "assign these to a group (or commit them separately) and rerun; the commits above stand."
fi

printf '\nCommits on top of %s:\n' "$BASE_TIP"
git log --oneline "$BASE_FULL"..HEAD
leftover=$(git status --porcelain | grep -v -E '^.. (commit_plan|commit_msgs/)' || true)
if [ -n "$leftover" ]; then
  printf '%s\n' "$leftover"; die "working tree not clean after the commits"
fi
printf '\nClean. Next (not run by this script):\n'
printf '  uv run ruff check src tests; uv run pytest -q      # expect 438 passed, 2 deselected\n'
printf '  git log --oneline %s..HEAD                        # expect 3 lines\n' "$BASE_TIP"
printf '  git push -u origin %s                # expect a SHA range, not "Everything up-to-date"\n' "$BRANCH"
