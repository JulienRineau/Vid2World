#!/bin/bash
# Run Vid2World overfit test on SF-Fold dataset
# Usage: ./scripts/run_overfit_test.sh [--gpus N] [--wandb]
#
# This script validates the training pipeline by overfitting on 4 samples.
# Expected runtime: ~5-10 minutes on 8x H100 GPUs
#
# Prerequisites:
#   1. DynamiCrafter checkpoint: ./scripts/download_checkpoint.sh
#   2. SF-Fold dataset converted: sf_fold_npz_cumulative/
#   3. (Optional) wandb: pip install wandb && wandb login

set -e

# Configuration
CONFIG="configs/manipulation/config_sf_fold_overfit.yaml"
LOG_DIR="/home/ubuntu/Texas/personal/Vid2World/logs"
EXPERIMENT_NAME="sf_fold_overfit_test"
CHECKPOINT_PATH="checkpoints/dynamicrafter_512_v1/model.ckpt"

# Default settings
NUM_GPUS=8
ENABLE_WANDB=false
MASTER_PORT=12869

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --wandb)
            ENABLE_WANDB=true
            shift
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --gpus N      Number of GPUs to use (default: 8)"
            echo "  --wandb       Enable wandb logging"
            echo "  --port PORT   Master port for distributed training (default: 12869)"
            echo "  --name NAME   Experiment name (default: sf_fold_overfit_test)"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  Vid2World Overfit Test"
echo "=============================================="
echo ""

# Pre-flight checks
echo -e "${BLUE}Running pre-flight checks...${NC}"

# Check 1: Config file exists
if [ ! -f "${CONFIG}" ]; then
    echo -e "${RED}✗ Config not found: ${CONFIG}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Config found: ${CONFIG}${NC}"

# Check 2: Checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo -e "${RED}✗ Checkpoint not found: ${CHECKPOINT_PATH}${NC}"
    echo ""
    echo "  Run: ./scripts/download_checkpoint.sh"
    exit 1
fi
CKPT_SIZE=$(du -h "${CHECKPOINT_PATH}" | cut -f1)
echo -e "${GREEN}✓ Checkpoint found: ${CHECKPOINT_PATH} (${CKPT_SIZE})${NC}"

# Check 3: Dataset exists
DATASET_DIR="/home/ubuntu/Texas/personal/Vid2World/sf_fold_npz_cumulative"
if [ ! -d "${DATASET_DIR}" ]; then
    echo -e "${RED}✗ Dataset not found: ${DATASET_DIR}${NC}"
    exit 1
fi
NPZ_COUNT=$(ls -1 "${DATASET_DIR}"/*.npz 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Dataset found: ${DATASET_DIR} (${NPZ_COUNT} episodes)${NC}"

# Check 4: GPUs available
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "${AVAILABLE_GPUS}" -lt "${NUM_GPUS}" ]; then
    echo -e "${YELLOW}⚠ Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} available${NC}"
    NUM_GPUS=${AVAILABLE_GPUS}
fi
echo -e "${GREEN}✓ GPUs available: ${AVAILABLE_GPUS} (using ${NUM_GPUS})${NC}"

# Check 5: wandb (optional)
WANDB_OVERRIDE=""
if [ "${ENABLE_WANDB}" = true ]; then
    if python3 -c "import wandb" 2>/dev/null; then
        echo -e "${GREEN}✓ wandb available${NC}"
        WANDB_OVERRIDE="lightning.wandb.enabled=true"
    else
        echo -e "${YELLOW}⚠ wandb requested but not installed. Disabling.${NC}"
        ENABLE_WANDB=false
    fi
else
    echo -e "${YELLOW}○ wandb disabled (use --wandb to enable)${NC}"
fi

# Create log directory
mkdir -p "${LOG_DIR}"
echo -e "${GREEN}✓ Log directory: ${LOG_DIR}${NC}"

echo ""
echo -e "${BLUE}=============================================="
echo "  Starting Overfit Test"
echo "==============================================${NC}"
echo ""
echo "  Config:     ${CONFIG}"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Log Dir:    ${LOG_DIR}"
echo "  wandb:      ${ENABLE_WANDB}"
echo ""
echo "  Expected runtime: ~5-10 minutes"
echo "  Max steps: 500"
echo "  Samples: 4 (overfit mode)"
echo ""
echo "  Press Ctrl+C to cancel..."
echo ""
sleep 3

# Build the command
CMD="python3 -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --master_addr=127.0.0.1 \
    --master_port=${MASTER_PORT} \
    --node_rank=0 \
    ./main/trainer.py \
    --base ${CONFIG} \
    --train \
    --name ${EXPERIMENT_NAME} \
    --logdir ${LOG_DIR} \
    --devices ${NUM_GPUS} \
    lightning.trainer.num_nodes=1"

# Add wandb override if enabled
if [ -n "${WANDB_OVERRIDE}" ]; then
    CMD="${CMD} ${WANDB_OVERRIDE}"
fi

# Run the training
echo -e "${BLUE}Executing:${NC}"
echo "${CMD}"
echo ""

# Change to project directory
cd /lambda/nfs/Texas/personal/Vid2World

# Execute
eval ${CMD}

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=============================================="
    echo "  ✓ Overfit test completed successfully!"
    echo "==============================================${NC}"
    echo ""
    echo "  Logs: ${LOG_DIR}/${EXPERIMENT_NAME}"
    echo ""
    echo "  Next steps:"
    echo "    1. Check TensorBoard: tensorboard --logdir ${LOG_DIR}"
    echo "    2. Verify loss decreased (should drop significantly)"
    echo "    3. Check generated images in logs"
    echo "    4. If satisfied, run full training:"
    echo "       ./scripts/run_sf_fold_train.sh"
else
    echo ""
    echo -e "${RED}✗ Overfit test failed. Check logs above.${NC}"
    exit 1
fi
