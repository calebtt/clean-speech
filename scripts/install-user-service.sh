#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/caleb/clean-speech-daemon"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_PATH="${UNIT_DIR}/clean-speech-daemon.service"

mkdir -p "${UNIT_DIR}"
cp "${PROJECT_DIR}/systemd/clean-speech-daemon.service" "${UNIT_PATH}"
systemctl --user daemon-reload

printf 'Installed %s\n' "${UNIT_PATH}"
printf 'Manual start: systemctl --user start clean-speech-daemon.service\n'
printf 'Stop when done: systemctl --user stop clean-speech-daemon.service\n'
