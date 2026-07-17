#!/usr/bin/env bash
#
# setup_docker.sh — TorchNeuron Beta 3 in-container setup
# Run INSIDE the container launched by setup_host.sh:
#     /host-script/setup_docker.sh
#
# Idempotent: safe to re-run. If anything fails, it stops and prints a
# diagnostics block to paste to Claude. Otherwise no need to loop anyone in.

set -Eeuo pipefail

LOG_DIR="/host-logs"
[ -d "$LOG_DIR" ] || LOG_DIR="/tmp"
LOG_FILE="$LOG_DIR/container_$(date +%Y%m%d_%H%M%S).log"

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
  echo "--- neuron devices ---"; neuron-ls 2>&1 | head -30 || echo "neuron-ls failed"
  echo ""
  echo "--- python/torch stack ---"
  pip list 2>/dev/null | grep -e neuron -e torch -e nki -e transformers -e perforated || true
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

# ----------------------------- STEPS ----------------------------------------
log "=== setup_docker.sh starting (inside container) ==="
log "Log file: $LOG_FILE"

step "verify shm size" "relaunch container with --ipc=host (setup_host.sh does this automatically)"
SHM_KB=$(df --output=avail /dev/shm | tail -1 | tr -dc '0-9')
log "/dev/shm available: $((SHM_KB / 1024 / 1024))GB"
if [ "$SHM_KB" -lt 8000000 ]; then
  log "WARNING: /dev/shm under 8GB — DataLoader workers may crash. Container should be launched with --ipc=host."
fi

step "verify NeuronCores" "if this fails, the driver on the HOST is wrong — re-run setup_host.sh; if still failing, reboot the instance"
neuron-ls 2>&1 | tee -a "$LOG_FILE"

step "apt update"
apt update >>"$LOG_FILE" 2>&1 || log "apt update had warnings (legacy key warnings are known and harmless)"

step "pip install transformers"
if python3 -c "import transformers" 2>/dev/null; then
  log "transformers already installed: $(python3 -c 'import transformers; print(transformers.__version__)')"
else
  pip install transformers 2>&1 | tail -5 | tee -a "$LOG_FILE"
fi

step "verify torch sees neuron device" "if the tensor move fails, the runtime libs in the container don't match the host driver — re-run setup_host.sh from a clean image"
python3 - <<'EOF' 2>&1 | tee -a "$LOG_FILE"
import torch, torch_neuronx
print("torch:", torch.__version__)
print("torch_neuronx:", torch_neuronx.__version__)
x = torch.ones(2, 2).to(torch.device("neuron"))
print("tensor on neuron device OK:", x.device)
EOF

step "verify PerforatedAI mount" "mount missing — container must be launched by setup_host.sh (it adds the -v flags)"
ls /workspace/PerforatedAI/Examples/imagenet >>"$LOG_FILE" 2>&1
log "PerforatedAI mounted OK"

step "install PerforatedAI library" "if torch got upgraded/clobbered, fix with: pip install 'torch==2.11.0' 'torchvision==0.26.2' then re-run"
if python3 -c "import perforatedai" 2>/dev/null; then
  log "perforatedai already importable, skipping"
elif [ -f /workspace/PerforatedAI/setup.py ] || [ -f /workspace/PerforatedAI/pyproject.toml ]; then
  # CRITICAL: pin torch to the container's Neuron-compatible version so pip
  # can't replace it with generic PyPI torch (this broke a previous run)
  TORCH_VER=$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')
  echo "torch==${TORCH_VER}" > /tmp/pai_constraints.txt
  log "Installing perforatedai with torch pinned to ${TORCH_VER}"
  pip install -e /workspace/PerforatedAI -c /tmp/pai_constraints.txt 2>&1 | tail -5 | tee -a "$LOG_FILE"
else
  log "No setup.py/pyproject.toml at repo root — install your closed-source pip package manually if needed"
fi

step "re-verify neuron device after installs" "torch was clobbered by a dependency — fix: pip install 'torch==2.11.0' 'torchvision==0.26.2'"
python3 - <<'EOF' 2>&1 | tee -a "$LOG_FILE"
import torch, torch_neuronx
x = torch.ones(2, 2).to(torch.device("neuron"))
print("post-install check OK — torch", torch.__version__, "on", x.device)
EOF

# ----------------------------- DONE -----------------------------------------
log "\n========== SETUP COMPLETE =========="
log "Smoke test (optional, first run compiles NEFFs for several minutes — normal):"
log "  cd /workspace/torch_neuron_eager/examples/gpt2-train-loop && python3 train.py"
log ""
log "Your training dir:"
log "  cd /workspace/PerforatedAI/Examples/imagenet"
log ""
log "Before exiting the container, commit it FROM THE HOST so next launch skips all of this:"
log "  docker commit \$(docker ps -lq) my-neuron-setup"
