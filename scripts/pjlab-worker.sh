#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 4 ]]; then
  echo 'Usage: pjlab-worker.sh mask.png scan.pcd intrinsic.json extrinsic.json' >&2
  exit 2
fi
# The adapter creates all four inputs in this unique directory. Native
# calibration needs no network, host source checkout, GPU, or model secrets.
exec docker run --rm --init --network none --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=$PWD" --workdir "$PWD" \
  "${OMNILABEL_PJLAB_DOCKER_IMAGE:-omnilabel-pjlab:crlf}" "$@"
