#!/usr/bin/env bash
# Move the vendored trees onto new upstream tags.
#
# THERE IS NO PATCH-APPLY STEP IN THIS FORK. `scripts/build_image_source.sh` builds `vllm/` and
# `flashinfer/` exactly as they sit in git, and every fork change is a hand edit made directly in
# those trees. `patches/` is a historical record of what those edits are and why (see
# patches/README.md) -- it is NOT a runnable recipe, and most of it no longer applies to a current
# upstream tag.
#
# So an upstream bump is a 3-way MERGE of our vendored tree onto the new tag. Per tree:
#   1. clone upstream (blobless) into .revendor/<tree>
#   2. check out the CURRENT vendored tag, rsync our tree over it, commit  -> the fork delta
#   3. check out the NEW tag, `git cherry-pick` that commit                -> the 3-way merge
#   4. stop loudly on conflicts; once resolved, --sync-back copies the merged tree into the repo
#
# `git rebase` does NOT work here: vLLM's release tags sit on release branches carrying commits that
# are absent from a newer tag's history, so rebase tries to replay UPSTREAM commits. `cherry-pick`
# uses the old tag's tree as the merge base, which is exactly what we want.
#
# Usage:
#   scripts/revendor.sh <new_vllm_tag> <new_flashinfer_tag>   # e.g. v0.28.0 v0.6.16.post3
#   scripts/revendor.sh --sync-back                           # after resolving, copy trees back
#
# After --sync-back: review `git diff`, commit, then build + push yourself (no CI auto-build):
#   OWNER=<you> scripts/build_image_source.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WORK="$ROOT/.revendor"

vllm_repo=https://github.com/vllm-project/vllm.git
fi_repo=https://github.com/flashinfer-ai/flashinfer.git

# tree -> (scratch dir, rsync excludes); flashinfer's 3rdparty/ are submodules, empty in our tree
sync_back() {
  for t in vllm flashinfer; do
    [ -d "$WORK/$t" ] || continue
    if git -C "$WORK/$t" ls-files -u | grep -q .; then
      echo "REFUSE: $WORK/$t still has unresolved conflicts" >&2; exit 3
    fi
    local ex=(--exclude '.git/')
    [ "$t" = flashinfer ] && ex+=(--exclude '3rdparty/')
    rsync -a --delete "${ex[@]}" "$WORK/$t/" "$ROOT/$t/"
    echo "synced $WORK/$t -> $ROOT/$t"
  done
  if [ -f "$WORK/vllm.newtag" ]; then
    cp "$WORK/vllm.newtag" "$ROOT/UPSTREAM_VLLM_VERSION"
  fi
  echo
  echo "Review 'git diff', commit, then build + push locally:  OWNER=<you> scripts/build_image_source.sh"
}

rebase_tree() {
  local name=$1 repo=$2 old=$3 new=$4; shift 4
  local ex=(--exclude '.git/') dir="$WORK/$name"
  [ "$name" = flashinfer ] && ex+=(--exclude '3rdparty/')

  echo "== $name: $old -> $new =="
  rm -rf "$dir"
  git clone --quiet --filter=blob:none "$repo" "$dir"
  git -C "$dir" rev-parse -q --verify "$old^{commit}" >/dev/null || { echo "no such tag: $old" >&2; exit 2; }
  git -C "$dir" rev-parse -q --verify "$new^{commit}" >/dev/null || { echo "no such tag: $new" >&2; exit 2; }

  git -C "$dir" checkout --quiet -b fork-old "$old"
  rsync -a --delete "${ex[@]}" "$ROOT/$name/" "$dir/"
  git -C "$dir" add -A
  git -C "$dir" commit --quiet -m "fork: vendored $name edits on top of $old"
  echo "   fork delta: $(git -C "$dir" show --shortstat --format='' HEAD | tr -d '\n')"

  git -C "$dir" checkout --quiet -b fork-new "$new"
  if git -C "$dir" cherry-pick --no-commit fork-old >/dev/null 2>&1; then
    echo "   clean merge"
  else
    echo "   CONFLICTS -- resolve them in $dir, then re-run with --sync-back:"
    git -C "$dir" diff --name-only --diff-filter=U | sed 's/^/     /'
    CONFLICTED=1
  fi
}

if [ "${1:-}" = "--sync-back" ]; then sync_back; exit 0; fi

VLLM_TAG="${1:?usage: revendor.sh <vllm_tag> <flashinfer_tag>   |   revendor.sh --sync-back}"
FI_TAG="${2:?usage: revendor.sh <vllm_tag> <flashinfer_tag>   |   revendor.sh --sync-back}"
mkdir -p "$WORK"
CONFLICTED=0

rebase_tree vllm       "$vllm_repo" "$(cat "$ROOT/UPSTREAM_VLLM_VERSION")" "$VLLM_TAG"
echo "$VLLM_TAG" > "$WORK/vllm.newtag"
rebase_tree flashinfer "$fi_repo"   "v$(cat "$ROOT/flashinfer/version.txt")" "$FI_TAG"

echo
if [ "$CONFLICTED" = 1 ]; then
  echo "Resolve the conflicts listed above (git add the results; do NOT commit), then:"
  echo "  scripts/revendor.sh --sync-back"
  exit 1
fi
echo "Both trees merged clean. Copy them into the repo with:"
echo "  scripts/revendor.sh --sync-back"
