#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/Library/TeX/texbin:$PATH"

report_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_dir="$(cd "$report_dir/.." && pwd)"
build_dir="$report_dir/build"
passes="${LATEX_PASSES:-3}"

mkdir -p "$build_dir/parts"
rm -f \
  "$build_dir/main.aux" \
  "$build_dir/main.log" \
  "$build_dir/main.out" \
  "$build_dir/main.pdf" \
  "$build_dir/main.toc" \
  "$build_dir"/parts/*.aux

for ((i = 1; i <= passes; i++)); do
  echo "pdflatex pass $i/$passes"
  (cd "$report_dir" && pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" main.tex)
done

cp "$build_dir/main.pdf" "$report_dir/report.pdf"
cp "$build_dir/main.pdf" "$project_dir/report.pdf"
