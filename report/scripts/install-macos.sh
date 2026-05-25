#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/Library/TeX/texbin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Этот скрипт предназначен для macOS." >&2
  exit 1
fi

if ! command -v pdflatex >/dev/null 2>&1 || ! command -v tlmgr >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Не найден Homebrew. Установите его: https://brew.sh" >&2
    exit 1
  fi

  echo "Устанавливаю BasicTeX через Homebrew..."
  brew install --cask basictex
  export PATH="/Library/TeX/texbin:$PATH"
fi

if ! command -v tlmgr >/dev/null 2>&1; then
  echo "tlmgr не найден. Проверьте установку TeX Live / BasicTeX." >&2
  exit 1
fi

packages=(
  extsizes
  iftex
  cmap
  cyrillic
  babel-russian
  hyph-utf8
  hyphen-russian
  ruhyphen
  tempora
  lh
  greek-fontenc
  cbfonts-fd
  cbfonts
  geometry
  setspace
  graphics
  tools
  float
  booktabs
  multirow
  amsmath
  amsfonts
  listings
  xcolor
  caption
  enumitem
  titlesec
  tocloft
  etoolbox
  hyperref
  url
  rerunfilecheck
)

echo "Устанавливаю LaTeX-пакеты для сборки отчета..."
sudo tlmgr install "${packages[@]}"

echo "Готово. Сборка отчета: make report"
