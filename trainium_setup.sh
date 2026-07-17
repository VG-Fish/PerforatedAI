#!/usr/bin/env bash
#
# setup_host.sh — TorchNeuron Beta 3 host setup (run on the Trainium instance)
# Idempotent: safe to re-run. Skips anything already done.
# Ends by dropping you into the container. Inside, run: /host-script/setup_docker.sh
#
# If anything fails, it stops and prints a diagnostics block to paste to Claude.
# If nothing fails, you never need to loop anyone in.

set -Eeuo pipefail

# ----------------------------- CONFIG ---------------------------------------
ECR_ACCOUNT="421672808698"
ECR_REGION="us-east-1"
ECR_REPO="concourse-release-0461d3b"
ECR_TAG="latest"   # doc truncated at ":lat..." — edit if your tag differs
IMAGE_REF="${ECR_ACCOUNT}.dkr.ecr.${ECR_REGION}.amazonaws.com/${ECR_REPO}:${ECR_TAG}"

COMMITTED_IMAGE_NAME="my-neuron-setup"   # if this local image exists, reuse it
PAI_REPO_URL="https://github.com/PerforatedAI/PerforatedAI.git"

HOST_CODE_DIR="$HOME/PerforatedAI"
HOST_DATA_DIR="$HOME/Datasets"

HARD_MIN_FREE_GB=25   # refuse below this
WARN_FREE_GB=60       # warn below this (ImageNet-scale work will need more)

LOG_DIR="$HOME/trainium_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/host_$(date +%Y%m%d_%H%M%S).log"

# ----------------------------- LOGGING / ERROR HANDLING ---------------------
CURRENT_STEP="startup"
STEP_HINT=""

log()  { echo -e "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
step() { CURRENT_STEP="$1"; STEP_HINT="${2:-}"; log "\n========== STEP: $1 =========="; }

diagnostics() {
  echo ""
  echo "################ DIAGNOSTICS (paste this whole block to Claude) ################"
  echo "FAILED_STEP: ${CURRENT_STEP}"
  echo "FAILED_COMMAND: ${BASH_COMMAND:-unknown}"
  echo "EXIT_CODE: ${1:-unknown}"
  echo "TIMESTAMP: $(date -Is)"
  [ -n "$STEP_HINT" ] && { echo ""; echo "LIKELY FIX: $STEP_HINT"; }
  echo ""
  echo "--- disk space ---"; df -h / /dev/shm 2>/dev/null || true
  echo ""
  echo "--- memory ---"; free -h 2>/dev/null || true
  echo ""
  echo "--- docker state ---"
  docker images 2>/dev/null | head -10 || echo "docker unavailable"
  docker ps -a 2>/dev/null | head -10 || true
  echo ""
  echo "--- neuron devices ---"; neuron-ls 2>&1 | head -30 || echo "neuron-ls failed"
  echo ""
  echo "--- neuron packages ---"; dpkg -l 2>/dev/null | grep -i neuron || echo "none"
  echo ""
  echo "--- last 40 log lines ---"; tail -n 40 "$LOG_FILE" 2>/dev/null || true
  echo "################################ END DIAGNOSTICS ###############################"
}

on_error() {
  local code=$?
  log "!!!!! FAILURE in step '${CURRENT_STEP}' (exit code ${code}) !!!!!"
  [ -n "$STEP_HINT" ] && log ">>> Likely fix: $STEP_HINT"
  diagnostics "$code" | tee -a "$LOG_FILE"
  exit "$code"
}
trap on_error ERR

free_gb_root() { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

# ----------------------------- STEPS ----------------------------------------
log "=== setup_host.sh starting ==="
log "Log file: $LOG_FILE"

step "preflight: required commands" "install missing tool (sudo apt-get install awscli docker.io git)"
for c in aws docker git; do
  command -v "$c" >/dev/null 2>&1 || { log "Missing required command: $c"; false; }
done

step "preflight: AWS credentials" "credentials rejected — check the access key/secret you entered, or attach an IAM instance role"
if ! aws sts get-caller-identity >>"$LOG_FILE" 2>&1; then
  log "No AWS credentials found on this instance. Let's set them up now."
  log "(You'll need your Access Key ID and Secret Access Key. Region should be: ${ECR_REGION})"
  aws configure
  # Re-check — fail properly (with diagnostics) if the entered creds don't work
  aws sts get-caller-identity >>"$LOG_FILE" 2>&1
fi
# Make sure a region is set even if creds came from an instance role
if [ -z "$(aws configure get region 2>/dev/null || true)" ]; then
  log "No default region set — setting to ${ECR_REGION}"
  aws configure set region "$ECR_REGION"
fi
log "AWS credentials OK: account $(aws sts get-caller-identity --query Account --output text)"

step "preflight: disk space"
AVAIL=$(free_gb_root)
log "Free space on /: ${AVAIL}GB"
# The hard minimum only matters if the big downloads (image pull + workspace
# extraction) haven't happened yet. On reruns, don't block.
HEAVY_WORK_DONE=false
if docker image inspect "$COMMITTED_IMAGE_NAME" >/dev/null 2>&1; then
  HEAVY_WORK_DONE=true
elif docker image inspect "$IMAGE_REF" >/dev/null 2>&1 && [ -d "$HOME/workspace/runtime_artifacts" ]; then
  HEAVY_WORK_DONE=true
fi
if [ "$HEAVY_WORK_DONE" = false ] && [ "$AVAIL" -lt "$HARD_MIN_FREE_GB" ]; then
  STEP_HINT="grow the EBS volume (Console -> Volumes -> Modify), then: sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1"
  log "BLOCKED: below ${HARD_MIN_FREE_GB}GB free and image/artifacts not yet downloaded. This killed a previous run."
  false
elif [ "$AVAIL" -lt "$WARN_FREE_GB" ]; then
  log "WARNING: only ${AVAIL}GB free. Setup can proceed, but dataset downloads and NEFF caches may fill this. Strongly consider growing the volume."
fi

step "preflight: docker group membership"
if ! id -nG "$USER" | grep -qw docker; then
  log "Adding $USER to docker group (takes effect on next login; using current session as-is)"
  sudo usermod -aG docker "$USER"
fi

step "check for committed image '${COMMITTED_IMAGE_NAME}'"
RUN_IMAGE="$IMAGE_REF"
if docker image inspect "$COMMITTED_IMAGE_NAME" >/dev/null 2>&1; then
  log "Found '${COMMITTED_IMAGE_NAME}' — reusing (skipping pull + installs)"
  RUN_IMAGE="$COMMITTED_IMAGE_NAME"
else
  step "ECR login + pull DLC image" "check credentials have ECR pull permission for account ${ECR_ACCOUNT}; check ECR_TAG in CONFIG is correct"
  if docker image inspect "$IMAGE_REF" >/dev/null 2>&1; then
    log "Base image already pulled, skipping"
  else
    aws ecr get-login-password --region "$ECR_REGION" \
      | docker login --username AWS --password-stdin \
        "${ECR_ACCOUNT}.dkr.ecr.${ECR_REGION}.amazonaws.com" >>"$LOG_FILE" 2>&1
    docker pull "$IMAGE_REF" 2>&1 | tee -a "$LOG_FILE"
  fi

  step "extract runtime artifacts from image"
  if [ -d "$HOME/workspace/runtime_artifacts" ]; then
    log "\$HOME/workspace already extracted, skipping"
  else
    cd "$HOME"
    docker rm -f tmp >/dev/null 2>&1 || true
    docker create --name tmp "$IMAGE_REF" >>"$LOG_FILE" 2>&1
    docker cp tmp:/workspace . 2>&1 | tee -a "$LOG_FILE"
    docker rm tmp >>"$LOG_FILE" 2>&1
  fi

  step "install build prerequisites (dkms, build-essential)"
  sudo apt-get update >>"$LOG_FILE" 2>&1
  sudo apt-get install -y dkms build-essential >>"$LOG_FILE" 2>&1

  step "install Neuron runtime debs from DLC artifacts" "a version conflict here may mean held packages: try 'sudo apt-mark unhold <pkg>' then re-run"
  sudo dpkg -i "$HOME"/workspace/runtime_artifacts/*.deb 2>&1 | tee -a "$LOG_FILE"
fi

step "verify NeuronCores visible" "if neuron-ls fails after driver install, reboot the instance and re-run this script"
if command -v neuron-ls >/dev/null 2>&1; then
  neuron-ls 2>&1 | tee -a "$LOG_FILE"
else
  log "neuron-ls not on host PATH — will verify inside container"
fi

step "prepare host mount directories"
mkdir -p "$HOST_CODE_DIR" "$HOST_DATA_DIR"
if [ ! -d "$HOST_CODE_DIR/.git" ]; then
  log "Cloning PerforatedAI to host (survives container restarts)"
  git clone "$PAI_REPO_URL" "$HOST_CODE_DIR" 2>&1 | tee -a "$LOG_FILE"
else
  log "PerforatedAI repo already on host"
fi

step "launch container"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
log "Launching with --ipc=host + volume mounts. Inside, run: /host-script/setup_docker.sh"
exec docker run -it --privileged --ipc=host \
  -v "$HOST_CODE_DIR":/workspace/PerforatedAI \
  -v "$HOST_DATA_DIR":/workspace/PerforatedAI/Examples/imagenet/Datasets \
  -v "$SCRIPT_DIR":/host-script:ro \
  -v "$LOG_DIR":/host-logs \
  "$RUN_IMAGE" /bin/bash
