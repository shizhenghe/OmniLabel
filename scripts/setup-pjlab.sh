#!/usr/bin/env bash
# Optional native CPU engine. Does not install a segmentation model/checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
revision=7fe8d46caac040cbcd89cc65e79e5519df01bcd4
source_dir="$PWD/.cache/pjlab-source"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/PJLab-ADG/SensorsCalibration.git "$source_dir"
  git -C "$source_dir" sparse-checkout set lidar2camera/auto_calib
  git -C "$source_dir" checkout --detach "$revision"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]] || { echo 'PJLab cache has a different revision.' >&2; exit 1; }
git -C "$source_dir" diff HEAD --exit-code --quiet
[[ -z "$(git -C "$source_dir" ls-files --others --exclude-standard -- lidar2camera/auto_calib/src lidar2camera/auto_calib/include lidar2camera/auto_calib/CMakeLists.txt)" ]] || { echo 'PJLab source cache contains untracked build inputs.' >&2; exit 1; }
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
mkdir "$context/auto_calib"
cp -R "$source_dir/lidar2camera/auto_calib/src" "$source_dir/lidar2camera/auto_calib/include" "$source_dir/lidar2camera/auto_calib/CMakeLists.txt" "$context/auto_calib/"
cp scripts/pjlab.Dockerfile "$context/Dockerfile"
# Use the inherited proxy; retain TLS and package-signature verification.
ca_bundle="${CODEX_PROXY_CERT:-/etc/ssl/certs/ca-certificates.crt}"
docker build --secret "id=proxy_ca,src=$ca_bundle" -t "${OMNILABEL_PJLAB_DOCKER_IMAGE:-omnilabel-pjlab:crlf}" "$context"
echo "Set OMNILABEL_PJLAB_BINARY=$PWD/scripts/pjlab-worker.sh before starting OmniLabel."
