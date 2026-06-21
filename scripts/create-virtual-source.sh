#!/usr/bin/env bash
set -euo pipefail

FIFO_PATH="${1:-/tmp/clean-speech-daemon.pcm}"
SOURCE_NAME="${2:-clean_speech_microphone}"
DESCRIPTION="${3:-Clean Speech Microphone}"

if [[ ! -p "${FIFO_PATH}" ]]; then
  rm -f "${FIFO_PATH}"
  mkfifo "${FIFO_PATH}"
  chmod 666 "${FIFO_PATH}"
fi

pactl load-module module-pipe-source \
  source_name="${SOURCE_NAME}" \
  file="${FIFO_PATH}" \
  format=s16le \
  rate=48000 \
  channels=1 \
  source_properties="device.description=${DESCRIPTION}"
