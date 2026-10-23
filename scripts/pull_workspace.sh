#!/usr/bin/env bash
# Pull the live Silver pipeline source from the Databricks workspace into silver_pipeline/.
# Files edited locally since the last pull are left alone (and reported) so unpushed work is never lost.
set -uo pipefail

REMOTE="/Workspace/Users/aes10130@nyu.edu/Silver Entity Resolution Pipeline"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="$ROOT/silver_pipeline"
SNAP="$ROOT/.databricks-pull/snapshot"  # copy of the workspace as of the last pull
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if ! databricks workspace export-dir "$REMOTE" "$TMP" >/dev/null 2>&1; then
  echo "[pull_workspace] Could not pull from Databricks (not logged in or offline?). Run: databricks auth login"
  exit 0
fi

mkdir -p "$LOCAL" "$SNAP"
updated=() skipped=()
while IFS= read -r -d '' f; do
  rel="${f#"$TMP"/}"
  if [[ -f "$LOCAL/$rel" ]] && ! cmp -s "$LOCAL/$rel" "$SNAP/$rel" 2>/dev/null \
     && ! cmp -s "$LOCAL/$rel" "$f"; then
    skipped+=("$rel")  # local edits not yet pushed
    continue
  fi
  if ! cmp -s "$LOCAL/$rel" "$f" 2>/dev/null; then
    mkdir -p "$(dirname "$LOCAL/$rel")"
    cp "$f" "$LOCAL/$rel"
    updated+=("$rel")
  fi
done < <(find "$TMP" -type f -print0)

rm -rf "$SNAP" && mkdir -p "$SNAP" && cp -R "$TMP"/. "$SNAP"/

echo "[pull_workspace] Pulled silver_pipeline/ from Databricks workspace."
((${#updated[@]})) && printf '  updated: %s\n' "${updated[@]}"
((${#skipped[@]})) && printf '  NOT updated (local edits not yet pushed; workspace may differ): %s\n' "${skipped[@]}"
exit 0
