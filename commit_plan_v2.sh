#!/usr/bin/env bash
# CP4b commit plan v2 — referee script. Run from the repo root on the host (Git Bash), where git
# works. Written by the referee against the real state of origin/refactor/2026-09:
#
#   CP4a is ALREADY committed and pushed (7942e6b docs, d3ded0e configs, 4080c1a tests), so the
#   builder's groups 01-03 are pre-committed. This script commits only what git reports as changed
#   relative to 4080c1a, in the builder's dependency order (CP4b report §9, groups 4-9), plus the
#   post-CP4a deltas to docs and tests. It never pushes, never amends, never creates a branch, and
#   refuses to proceed if any changed path is one it cannot assign.
#
#   Safe to rerun: groups already committed have no pending paths and are skipped.
set -euo pipefail

CP4A_TIP=4080c1ab44deae54db2957f3e95ab052efabb12c
BRANCH=refactor/2026-09
TRAILER_1='The builder wrote this in its sandbox. The referee reviewed and committed it.'
TRAILER_2='Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>'

die() { printf 'STOP: %s\n' "$*" >&2; exit 1; }

# ---- guards -----------------------------------------------------------------------------------
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "not on $BRANCH"
if ! git merge-base --is-ancestor "$CP4A_TIP" HEAD; then
  die "HEAD $(git rev-parse --short HEAD) does not contain 4080c1a (CP4a). If HEAD is 9310a6c the
      CP4a commits were made elsewhere: run  git fetch origin && git reset --soft origin/$BRANCH
      (moves HEAD, leaves the working tree alone), then rerun this script."
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
  if [ "$(git rev-parse HEAD)" = "$CP4A_TIP" ]; then
    die "nothing to commit: the working tree equals 4080c1a. Is the CP4b tree in this checkout?"
  fi
  printf 'Nothing left to commit. Commits on top of 4080c1a:\n'
  git log --oneline "$CP4A_TIP"..HEAD
  printf '\nNext (not run by this script):\n'
  printf '  uv run ruff check src tests && uv run pytest -q      # expect 403 passed, 2 deselected\n'
  printf '  git push -u origin %s\n' "$BRANCH"
  exit 0
fi
printf 'Changed relative to HEAD (%d paths):\n' "${#PENDING[@]}"
printf '  %s\n' "${!PENDING[@]}" | sort

# A config that moved after the CP4a commit is a protocol change: review it, do not auto-commit.
for p in configs/alpha_lite.yaml configs/comparative.yaml configs/gates/COMPARATIVE.v1.yaml \
         configs/windowed_w2.yaml configs/windowed_w4.yaml configs/gates/G0.7.v1.yaml; do
  [ -z "${PENDING[$p]:-}" ] || die "$p changed after the CP4a commit (registered file). Review it
      (SHA re-pin? D-entry?) before anything is committed. Nothing was committed."
done
# CP3b-era files the builder said it did NOT stage; a change here needs a human look first.
for p in src/qrc_thresher/plugins/builtin.py src/qrc_thresher/reservoirs/ablations.py \
         tests/test_windowed_parity_known_answer.py; do
  [ -z "${PENDING[$p]:-}" ] || die "$p shows as changed; the builder flagged it as CP3b-era and
      unstaged. Decide before committing. Nothing was committed."
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
'docs: record D015 (builder) notes; align METHODOLOGY.md with D011–D014' \
'CP4b. D015 lists the builder'"'"'s implementation notes for D011–D014 at CP4b
(per-record sweep ids, the NARMA-10 minimum on every config, the
compare_arms match callback, run_pair shared by the engine and the run
command, two default QRC rows per pair with distinct design labels,
ablation rows assigned to a task through primary_metric_name, summary
without tabulate, evaluate_config returning an error for a missing
record). The PI accepted D015 as written in the CP4b review; nothing in
it changes a threshold, metric, direction, floor or the protocol file.
METHODOLOGY.md is brought into agreement with D011–D014 (stm_memory
over k >= 1, the ESN input bias, the windowed RKS at sigma/sqrt(d),
training.washout, reservoir.encoding_scale).' \
docs/DECISIONS.md docs/METHODOLOGY.md

commit_group \
'tests: CP4b edits under PI rulings 1, 3, 4, 6–10 and the washout ruling' \
'CP4b. Test files edited after the CP4a commit, each under a ruling of the
CP4b review (report §3): synthetic_rows (tuning_record_sha, the default
labels; rulings 1, 3), test_comparative_family (the G0.7 file'"'"'s hashes and
the mismatch test, default never flips, two candidate rows, family_json
and family_sha256, holm_family, the protocol pin; rulings 1, 3, 4, 5, 7,
9, 11), test_tuning (record fields; ruling 3), test_encoding_scale,
test_matched_ablations and test_windowed_qrc (the unconditional scale
suffix in the hash preimage; ruling 8), test_g07_wiring (tuned_qrc,
--tuning-config; ruling 4), test_g05_crosscheck (18 scale cases and the
scale power test; ruling 6), test_manifest_schema14 (31 columns, four
design labels; rulings 3, 10), test_comparative_commands (ruling 3),
test_washout (the NARMA-10 minimum of 10 rows on every config; the
washout ruling, builder departure recorded in D015). No assertion was
weakened; every other test file is as CP4a left it.' \
tests/synthetic_rows.py tests/test_baseline_run.py tests/test_comparative_commands.py \
tests/test_comparative_family.py tests/test_encoding_scale.py tests/test_esn.py \
tests/test_g05_crosscheck.py tests/test_g07_wiring.py tests/test_gates.py \
tests/test_manifest_schema14.py tests/test_matched_ablations.py tests/test_measurement_model.py \
tests/test_metrics.py tests/test_paired_stats.py tests/test_runs_csv_schema.py \
tests/test_tuning.py tests/test_washout.py tests/test_windowed_qrc.py

commit_group \
'metrics: paired statistics, d_z, sided power analysis and stm_memory' \
'CP4b, code 1 of 6 (D013; PI ruling 10). metrics/paired.py: paired_statistics
(one- and two-sided paired t in the registered direction, Wilcoxon with
zero_method wilcox and exact-without-ties, the 95% BCa CI with B = 2000
and rng seed 20260923, d_z null at zero variance) and compare_arms, the
one comparison routine every gate calls. metrics/stats.py: paired_test
names d_z; power_analysis takes a required sidedness and uses the
noncentral t; achieved_power. metrics/scoring.py: stm_memory, the memory
sum over k >= 1, the single scoring function used by every writer, tuner
and ablation (k = 0 reported as mc_k0, never counted).' \
$S/metrics/paired.py $S/metrics/stats.py $S/metrics/scoring.py

commit_group \
'tuning: one tuner under matched budgets; ESN input bias; tuning block' \
'CP4b, code 2 of 6 (D011, D012; PI ruling 3). tuning.py: one model-agnostic
selection routine over candidate feature matrices on 5 contiguous
validation blocks of the training rows after the washout, with the
harness readout; the tuning record results/tuning/<config_hash>/<task>.json
with selection_scope train_cv (asserted: no test index is touched),
cv_folds, seeds, the reservoir block, every configuration'"'"'s score, the
winner, sweep_id and record_sha256. baselines/esn.py: the input bias
b_in ~ Uniform(-1, 1) drawn after W_in and covered by weight_hash; the
tuning score without k = 0; tune_esn as a wrapper over the tuner.
config.py: reservoir.encoding_scale, training.washout and its
validator (structural minima only), the tuning block, baseline
extra=forbid, no gates block.' \
$S/tuning.py $S/baselines/esn.py $S/config.py

commit_group \
'reservoirs: encoding_scale in circuit and hash; windowed RKS; washout' \
'CP4b, code 3 of 6 (D011, D012; PI rulings 6, 8). reservoirs/windowed_qrc.py:
the encoding_scale field, RY(scale * u), and the hash preimage with the
scale unconditionally after the window suffix and before any ablation
suffix; rks_from_config with the window. baselines/random_features.py:
the windowed RKS at bandwidth sigma/sqrt(d) on the zero-padded input
window. reservoirs/qiskit_crosscheck.py: the scale argument on the
independent Qiskit side, for the 18 registered scale cases.
proof/benchmark_health.py: the ESN smoke check reads training.washout
and scores stm_memory.' \
$S/reservoirs/windowed_qrc.py $S/baselines/random_features.py \
$S/reservoirs/qiskit_crosscheck.py $S/proof/benchmark_health.py

commit_group \
'proof: manifest schema 1.4 and writers; run, ablation, baseline, summary' \
'CP4b, code 4 of 6 (D013; PI rulings 2, 3). proof/run_manifest.py: schema 1.4
(secondary_metrics, device, precision, design, sweep_id,
tuning_record_sha; 31 columns) and one row builder, manifest_row, which
db.py also writes through. task_names.py: the (model, task) -> task_name
and (member, task) -> row-selector mapping, the only place that parses
task_name suffixes. deploy.py: deployment of a tuned design by hash from
the record. engine.py and commands/run.py share run_pair; run runs every
seed pair and has no --seed. commands/ablation.py and commands/baseline.py
take the task and --design/--design-task; RKS runs under baseline.
commands/summary.py: the per-arm table.' \
$S/proof/run_manifest.py $S/db.py $S/task_names.py $S/deploy.py $S/engine.py \
$S/commands/run.py $S/commands/ablation.py $S/commands/baseline.py $S/commands/summary.py

commit_group \
'gates: the comparative family evaluator and G0.7 on the tuned design' \
'CP4b, code 5 of 6 (D013, D014; PI rulings 1, 4, 5, 7, 9, 11).
gates/comparative.py: the frozen COMPARATIVE.v1 spec (hash pinned),
Designs, the row selectors, evaluate_family in one pass with Holm over
m = 5, the floors, baseline_better with its two-sided p, the default
table that can never flip a verdict, one candidate row per pair, G1
clause (a) read from the newest G0.7.tuned_qrc file with its tuning
config hash and sweep_id verified and its path and SHA-256 recorded,
and write_family_report (the family JSON is the record, each per-gate
JSON a view carrying the family file'"'"'s path and SHA-256). gates/g07.py:
the tuned_qrc model, whose feature map is design_STM(pair) from the
record; the model details carry each pair'"'"'s depth, window, scale and
hash, the tuning config'"'"'s hash and the sweep_id.' \
$S/gates/comparative.py $S/gates/g07.py

commit_group \
'cli: tune, --design/--design-task, gate family and --tuning-config' \
'CP4b, code 6 of 6 (D011, D013; PI ruling 7). cli.py: the tune command,
--design {tuned,default} and --design-task {stm,parity,narma} on run,
ablation and baseline, gate family, gate G0.7 --model tuned_qrc
--tuning-config, and run without --seed. commands/gate.py: the family
routing; gate G1, G2, G2.5, G3 and G4 evaluate the whole family, print
that member and exit with its code (the pyproject entry points stay as
thin wrappers, listed as debt); the G0.5 scale cases; G5 reads
stm_memory. The builder'"'"'s sandbox run at this state: 403 passed,
2 deselected; ruff clean. The referee confirms on the host before the
push.' \
$S/commands/gate.py $S/cli.py

# ---- anything left is unassigned: report and stop, so a human decides -------------------------
if [ ${#PENDING[@]} -gt 0 ]; then
  printf '\nUNASSIGNED paths (not committed):\n'
  printf '  %s\n' "${!PENDING[@]}" | sort
  die "assign these to a group (or commit them separately) and rerun; the commits above stand."
fi

printf '\nCommits on top of 4080c1a:\n'
git log --oneline "$CP4A_TIP"..HEAD
leftover=$(git status --porcelain | grep -v -E '^.. (commit_plan|commit_msgs/)' || true)
if [ -n "$leftover" ]; then
  printf '%s\n' "$leftover"; die "working tree not clean after the commits"
fi
printf '\nClean. Next (not run by this script):\n'
printf '  uv run ruff check src tests && uv run pytest -q      # expect 403 passed, 2 deselected\n'
printf '  git push -u origin %s\n' "$BRANCH"
