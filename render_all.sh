#!/usr/bin/env bash
# Render every scene (in parallel), concatenate them into NMF_explained.mp4 and
# build NMF_explained.srt + SCRIPT.md from the logged captions.
# Usage: ./render_all.sh [-ql|-qm|-qh]     (default -qh = 1080p60)
set -euo pipefail
cd "$(dirname "$0")"
Q="${1:--qh}"
case "$Q" in -ql) RES=480p15 ;; -qm) RES=720p30 ;; -qh) RES=1080p60 ;; *) echo "bad quality $Q"; exit 1 ;; esac
# user-level TeX Live (see README), if present
for d in "$HOME"/texlive/bin/*/; do [ -d "$d" ] && PATH="$d:$PATH"; done
MANIM=./.venv/bin/manim; [ -x "$MANIM" ] || MANIM=./.env/bin/manim; [ -x "$MANIM" ] || MANIM=manim
SCENES=$(python3 -c "import ast
for n in ast.parse(open('nmf.py').read()).body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'SCENES':
        print(' '.join(ast.literal_eval(n.value)))")
mkdir -p media/logs
pids=()
for s in $SCENES; do
  "$MANIM" "$Q" --disable_caching nmf.py "$s" > "media/logs/$s.log" 2>&1 & pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] || { echo "a scene failed; see media/logs/"; exit 1; }
: > media/concat.txt
for s in $SCENES; do echo "file '$PWD/media/videos/nmf/$RES/$s.mp4'" >> media/concat.txt; done
OUT=NMF_explained.mp4; [ "$Q" = -qh ] || OUT="NMF_explained_$RES.mp4"
ffmpeg -y -loglevel error -f concat -safe 0 -i media/concat.txt -c copy "$OUT"
# captions -> .srt (soft subtitles) and SCRIPT.md (the narration, with timestamps)
python3 make_script.py "$RES" "${OUT%.mp4}.srt"
for s in $SCENES; do
  printf "%-18s %6.1f s\n" "$s" "$(ffprobe -v error -show_entries format=duration -of csv=p=0 media/videos/nmf/$RES/$s.mp4)"
done
d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")
printf "%s: %.1f s (%d:%02d)\n" "$OUT" "$d" "$(( ${d%.*} / 60 ))" "$(( ${d%.*} % 60 ))"
