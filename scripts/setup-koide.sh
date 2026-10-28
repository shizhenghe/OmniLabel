#!/usr/bin/env bash
set -euo pipefail

# Run this deployment step explicitly on the machine hosting the backend.
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_koide_image="${OMNILABEL_KOIDE_DOCKER_IMAGE:-omnilabel/koide:humble}"
if ! command -v docker >/dev/null 2>&1; then
  echo 'Docker is required. Install/start Docker, then run this script again.' >&2
  exit 1
fi
docker build --tag "$task_koide_image" --file "$task_repo_root/docker/koide/Dockerfile" "$task_repo_root/docker/koide"
python3 "$task_repo_root/scripts/koide_worker.py" --docker-image "$task_koide_image" --check
printf 'Configure the backend process environment and restart it:\nexport OMNILABEL_KOIDE_DOCKER_IMAGE=%q\n' "$task_koide_image"
