#!/usr/bin/env bash
# CP4c — the registered comparative sweep on configs/comparative.yaml (D011–D016). Host runbook.
# Run from the repo root with Git for Windows' bash:  & 'C:\Program Files\Git\bin\bash.exe' cp4c_run.sh
# Expect ~4 h serial (D011 cost note); `tune` is the long pole. Every row must carry a pushed,
# clean commit, so the guards refuse a dirty tree or a HEAD that is not origin's tip.
#
# Gates are findings: their exit codes (1 FAIL, 2 INSUFFICIENT) are logged, never fatal.
# Every other step must exit 0 or the sweep stops where it is (the family will say what is missing).
set -uo pipefail

BRANCH=refactor/2026-09
CFG=configs/comparative.yaml
ALPHA=configs/alpha_lite.yaml
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=results/cp4c_${STAMP}.log     # results/ is gitignored
mkdir -p results

die() { printf 'STOP: %s\n' "$*" | tee -a "$LOG" >&2; exit 1; }
log() { printf '%s  %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

# ---- guards -----------------------------------------------------------------------------------
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || die "not on $BRANCH"
git fetch -q origin "$BRANCH" || die "cannot fetch origin/$BRANCH"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ] || die "HEAD $(git rev-parse --short HEAD) is not origin/$BRANCH $(git rev-parse --short "origin/$BRANCH"); push or pull first"
# The manifest writer appends -dirty whenever `git status --porcelain` prints anything, untracked
# files included, so this guard must be exactly as strict: nothing at all may be listed. Keep this
# script and any other helper OUTSIDE the repository (e.g. ..\qrc-tools\) and run it from the
# repo root, or list the helpers in .git/info/exclude.
STATUS=$(git status --porcelain)
[ -z "$STATUS" ] || die "working tree not clean; every row would carry -dirty. git status --porcelain:
$STATUS"
uv sync --frozen >>"$LOG" 2>&1 || die "uv sync --frozen failed (see $LOG)"

CONFIG_HASH=$(uv run python -c "from pathlib import Path; from qrc_thresher.proof.run_manifest import _config_hash; print(_config_hash(Path('$CFG')))") || die "cannot compute the config hash"
log "CP4c start  commit=$(git rev-parse HEAD)  config_hash=$CONFIG_HASH  python=$(uv run python -V 2>&1)"

# A stale tuning record for this config would be deployed silently; a pre-1.4 runs.csv is refused
# by the header guard. Archive both rather than guess.
if [ -d "results/tuning/$CONFIG_HASH" ]; then
  mv "results/tuning/$CONFIG_HASH" "results/tuning/${CONFIG_HASH}.archived_${STAMP}" || die "cannot archive the old tuning record"
  log "archived results/tuning/$CONFIG_HASH -> ${CONFIG_HASH}.archived_${STAMP}"
fi
for f in results/runs.csv results/experiments.db; do
  if [ -e "$f" ]; then
    mv "$f" "${f}.archived_${STAMP}" || die "cannot archive $f"
    log "archived $f -> ${f}.archived_${STAMP}"
  fi
done

# ---- steps ------------------------------------------------------------------------------------
step() {  # step NAME cmd...   (fatal on non-zero)
  local name=$1; shift
  local t0=$SECONDS
  log "BEGIN $name"
  if "$@" >>"$LOG" 2>&1; then
    log "END   $name  exit=0  $((SECONDS - t0)) s"
  else
    local rc=$?
    log "END   $name  exit=$rc  $((SECONDS - t0)) s"
    die "$name failed (exit $rc); see $LOG. Fix, then rerun from this step by hand — do NOT rerun 'tune TASK' for a single task (it would re-stamp one record)."
  fi
}
gate() {  # gate NAME cmd...   (exit code is a finding, logged, never fatal)
  local name=$1; shift
  local t0=$SECONDS
  log "BEGIN $name"
  "$@" >>"$LOG" 2>&1
  local rc=$?
  log "END   $name  exit=$rc  $((SECONDS - t0)) s  (0 PASS, 1 FAIL, 2 INSUFFICIENT)"
}

Q="uv run qrc-thresher"

step "tune (stm, parity, narma; ONE sweep_id)"   $Q tune --config $CFG

step "run stm (design_STM)"                      $Q run stm    --config $CFG
step "run parity (design_parity)"                $Q run parity --config $CFG
step "run narma (design_NARMA)"                  $Q run narma  --config $CFG
step "run parity with design_STM (G1b)"          $Q run parity --config $CFG --design-task stm
step "run stm default (w=2, w=1)"                $Q run stm    --config $CFG --design default
step "run parity default"                        $Q run parity --config $CFG --design default
step "run narma default"                         $Q run narma  --config $CFG --design default

step "ablation no_entangle stm (reported margin)" $Q ablation no_entangle stm    --config $CFG
step "ablation no_entangle parity, design_STM (G1b)" $Q ablation no_entangle parity --config $CFG --design-task stm
step "ablation no_entangle stm default"          $Q ablation no_entangle stm    --config $CFG --design default
step "ablation no_entangle parity default"       $Q ablation no_entangle parity --config $CFG --design default
step "ablation haar stm (G2.5)"                  $Q ablation haar stm --config $CFG
step "ablation haar stm default"                 $Q ablation haar stm --config $CFG --design default

step "baseline stm (ESN, RKS; G3)"               $Q baseline stm    --config $CFG
step "baseline stm default (esn_nonlinear)"      $Q baseline stm    --config $CFG --design default
step "baseline parity (RKS; G2)"                 $Q baseline parity --config $CFG
step "baseline parity default"                   $Q baseline parity --config $CFG --design default
step "baseline narma (ESN; G4)"                  $Q baseline narma  --config $CFG
step "baseline narma default"                    $Q baseline narma  --config $CFG --design default

gate "G0.7 v1 on design_STM (G1 clause a)"       $Q gate G0.7 --config $ALPHA --model tuned_qrc --tuning-config $CFG
gate "COMPARATIVE.v1 family"                     $Q gate family --config $CFG
step "summary cp4c"                              $Q summary --phase cp4c

# ---- wrap-up ----------------------------------------------------------------------------------
STATUS=$(git status --porcelain)
[ -z "$STATUS" ] || die "the working tree changed during the sweep; investigate before reading any verdict. git status --porcelain:
$STATUS"
log "working tree clean at the end: every row carries $(git rev-parse HEAD) without -dirty"
FAMILY=$(ls -1 results/gates/COMPARATIVE.v1.*.json 2>/dev/null | sort | tail -1)
log "family JSON: ${FAMILY:-none written}"
if [ -n "${FAMILY:-}" ]; then
  uv run python - "$FAMILY" <<'EOF' | tee -a "$LOG"
import json, sys
r = json.load(open(sys.argv[1], encoding='utf-8'))
print(f"protocol {r['protocol_sha256'][:16]}...  commit {r['git_commit']}  sweep {r['sweep_id']}  rows {r['n_rows']}")
for m, e in r['members'].items():
    c = e['comparison']
    print(f"{m:5s} {e['result']:22s} n={e['n_pairs']:2d} raw_p={e['raw_p']:.3g} holm_p={e['adjusted_p']:.3g} "
          f"mean_a={c.get('mean_a')} mean_b={c.get('mean_b')} d_z={c.get('d_z')} "
          f"floor={(e['floor'] or {}).get('passed')} baseline_better={e['baseline_better']}  {e['message'][:80]}")
EOF
fi
log "CP4c done. Send the referee: $LOG, $FAMILY, the G0.7.tuned_qrc JSON, results/runs.csv and results/tuning/$CONFIG_HASH/*.json."
