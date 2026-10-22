#!/usr/bin/env bash
# Usage: ./render.sh <SceneName> [-ql|-qm|-qh]   (default -qh = 1080p60)
cd "$(dirname "$0")"
# user-level TeX Live (see README), if present
for d in "$HOME"/texlive/bin/*/; do [ -d "$d" ] && PATH="$d:$PATH"; done
MANIM=./.venv/bin/manim; [ -x "$MANIM" ] || MANIM=./.env/bin/manim; [ -x "$MANIM" ] || MANIM=manim
exec "$MANIM" "${2:--qh}" nmf.py "$1"
