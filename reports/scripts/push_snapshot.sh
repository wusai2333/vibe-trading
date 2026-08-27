#!/usr/bin/env bash
# Push a code snapshot (product code + research scripts + HANDOFF) to the
# personal GitHub mirror. Report docs (*.md except HANDOFF) and result/data
# files stay local-only.
#
# WHY orphan snapshots: the full history contains .github/workflows, which the
# deploy-key push cannot create (no `workflow` scope). An orphan (history-less)
# snapshot sidesteps that.
#
# WHY most of reports/ is excluded: report docs and result/data files are
# local-only. A previous version staged ALL of reports/ into the orphan
# branch, and the follow-up `git checkout -f main` then DELETED files from
# disk (six reports lost on 08-11/12/13, later recovered from dangling
# objects). Since 2026-08-27 (user directive) the research scripts
# (reports/data/*.py, reports/scripts/) and HANDOFF.md ARE pushed: they are
# re-added after the exclusion. This is safe against the old loss mode only
# because every reports/ file is tracked on main, so the checkout back
# restores rather than deletes - keep it that way, commit new reports files
# before pushing.
#
# HOW they are excluded: `git checkout --orphan` keeps main's index, and
# `git add -A -- . ':(exclude)reports'` does NOT drop already-staged entries,
# so it leaked every reports file that main still tracked (verified 08-14:
# 34 files leaked into two snapshot attempts). The reliable recipe is to
# stage everything and then `git rm --cached` the excluded paths.
#
# Usage: bash reports/scripts/push_snapshot.sh "snapshot message"
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MSG="${1:-Vibe-Trading snapshot $(date +%Y-%m-%d)}"
REMOTE=mine
BRANCH=push-snapshot

# Never run with a dirty tracked tree — that means uncommitted code that would
# be silently bundled or lost.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ERROR: tracked changes present. Commit or stash them first." >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

# If anything fails after we leave main (e.g. the push times out), return to
# main and delete the orphan branch instead of leaving the repo stranded.
cleanup() {
  if [ "$(git rev-parse --abbrev-ref HEAD)" != "main" ]; then
    git checkout -f -q main || true
    git branch -D "$BRANCH" >/dev/null 2>&1 || true
    echo "ABORTED: push failed; returned to main and deleted '$BRANCH'." >&2
  fi
}
trap cleanup ERR

git checkout -q --orphan "$BRANCH"
# Stage everything, then unstage local-only reports and the unpushable
# workflows, so they stay out of the snapshot and survive the checkout back.
git add -A
git rm -rq --cached reports .github/workflows
git add -A -- 'reports/data/*.py' reports/scripts reports/HANDOFF.md
git commit -q -m "$MSG"
git push -f "$REMOTE" "$BRANCH:main"
git checkout -f -q main
git branch -D "$BRANCH" >/dev/null
trap - ERR
echo "PUSH_DONE -> $REMOTE/main"
