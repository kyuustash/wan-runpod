#!/usr/bin/env bash
# Build this repo's Docker image and push to Docker Hub.
# Run on a Linux host with Docker (e.g. a RunPod Pod with Docker + enough disk for HF downloads).
#
# Usage:
#   export DOCKERHUB_USER="your-dockerhub-username"
#   export DOCKERHUB_TOKEN="your-dockerhub-access-token"  # not your account password
#   export HF_TOKEN="hf_..."   # optional; needed if model-downloads.sh pulls private/gated HF assets
#   bash scripts/build_and_push_dockerhub.sh
#
# If the repo is not next to this script, set REPO_DIR or GIT_REPO_URL:
#   export REPO_DIR="/workspace/wan-runpod"
#   # or
#   export GIT_REPO_URL="https://github.com/you/wan-runpod.git"
#   export GIT_REF="main"

set -euo pipefail

: "${DOCKERHUB_USER:?Set DOCKERHUB_USER}"
: "${DOCKERHUB_TOKEN:?Set DOCKERHUB_TOKEN (Docker Hub access token)}"

IMAGE_NAME="${IMAGE_NAME:-wan-runpod}"
TAG="${TAG:-latest}"
FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

# Repo root: directory that contains Dockerfile, docker/, workflows/, etc.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${REPO_DIR:-}" ]]; then
  :
elif [[ -f "/workspace/wan-runpod/Dockerfile" ]]; then
  REPO_DIR="/workspace/wan-runpod"
else
  REPO_DIR="$DEFAULT_REPO_DIR"
fi

GIT_REPO_URL="${GIT_REPO_URL:-}"
GIT_REF="${GIT_REF:-main}"

if [[ ! -f "${REPO_DIR}/Dockerfile" ]]; then
  if [[ -n "${GIT_REPO_URL}" ]]; then
    PARENT="$(dirname "${REPO_DIR}")"
    mkdir -p "${PARENT}"
    rm -rf "${REPO_DIR}"
    git clone --depth 1 --branch "${GIT_REF}" "${GIT_REPO_URL}" "${REPO_DIR}"
  else
    echo "No Dockerfile at ${REPO_DIR}/Dockerfile and GIT_REPO_URL is unset." >&2
    echo "Copy the full repo onto the volume or set REPO_DIR / GIT_REPO_URL." >&2
    exit 1
  fi
fi

export DOCKER_BUILDKIT=1

echo "Logging in to Docker Hub as ${DOCKERHUB_USER}..."
echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USER}" --password-stdin

BUILD_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  BUILD_ARGS+=(--build-arg "HF_TOKEN=${HF_TOKEN}")
fi

echo "Building ${FULL_IMAGE} from ${REPO_DIR} (platform linux/amd64)..."
docker build \
  --platform linux/amd64 \
  "${BUILD_ARGS[@]}" \
  -f "${REPO_DIR}/Dockerfile" \
  -t "${FULL_IMAGE}" \
  "${REPO_DIR}"

echo "Pushing ${FULL_IMAGE}..."
docker push "${FULL_IMAGE}"

echo "Done: ${FULL_IMAGE}"
