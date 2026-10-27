# Sourced by the job scripts. Every run gets runs/<NAME>_<job id>/ (weights, video,
# export, log, git.txt = the exact code: commit + uncommitted diff) and a local git tag
# run/<NAME>_<job id> on that commit.  List them: git tag -l 'run/*' -n1; ls runs/
RUN_ID="${NAME}_${SLURM_JOB_ID:-local$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR"
{ echo "run $RUN_ID  started $(date)"; echo "commit $(git rev-parse HEAD)  $(git log -1 --format=%s)"
  echo "== uncommitted changes:"; git status --short; git diff; } > "$RUN_DIR/git.txt"
git tag -f "run/$RUN_ID" HEAD >/dev/null 2>&1 || true

# record_finish <files...>: copy results into the run folder.
record_finish() {
    cp -p "$@" "$RUN_DIR/" 2>/dev/null || true
    [ -n "${SLURM_JOB_ID:-}" ] && cp -p oscar/logs/*_"$SLURM_JOB_ID".out "$RUN_DIR/" 2>/dev/null || true
    echo "recorded -> $RUN_DIR"
}
