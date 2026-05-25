#!/usr/bin/env bash
set -Eeuo pipefail

if ! command -v pacman >/dev/null 2>&1; then
  echo "Этот скрипт предназначен для Arch Linux / pacman-based систем." >&2
  exit 1
fi

if [[ "${EUID}" -eq 0 ]]; then
  sudo_cmd=()
else
  sudo_cmd=(sudo)
fi

if pacman -Si texlive-basic >/dev/null 2>&1; then
  packages=(
    make
    texlive-bin
    texlive-basic
    texlive-latex
    texlive-latexrecommended
    texlive-latexextra
    texlive-fontsrecommended
    texlive-fontsextra
    texlive-langcyrillic
    texlive-langgreek
  )
else
  packages=(
    make
    texlive-core
    texlive-latexextra
    texlive-fontsextra
    texlive-langcyrillic
    texlive-langgreek
  )
fi

echo "Устанавливаю зависимости для сборки отчета..."
"${sudo_cmd[@]}" pacman -S --needed "${packages[@]}"

echo "Готово. Сборка отчета: make report"
