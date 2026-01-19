#!/bin/bash
# Download DynamiCrafter 512 checkpoint for Vid2World
# Usage: ./scripts/download_checkpoint.sh [--resume]
#
# Prerequisites:
#   pip install huggingface_hub
#   (Optional) huggingface-cli login  # For rate limit avoidance
#
# The checkpoint will be saved to: checkpoints/dynamicrafter_512_v1/model.ckpt

set -e

# Configuration
CHECKPOINT_DIR="checkpoints/dynamicrafter_512_v1"
CHECKPOINT_FILE="model.ckpt"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_FILE}"

# HuggingFace model info
HF_REPO="Doubiiu/DynamiCrafter_512"
HF_FILENAME="model.ckpt"

# Expected file size (approximate, for verification)
EXPECTED_SIZE_GB=9.6

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  DynamiCrafter Checkpoint Downloader"
echo "=============================================="

# Check if checkpoint already exists
if [ -f "${CHECKPOINT_PATH}" ]; then
    FILE_SIZE=$(du -h "${CHECKPOINT_PATH}" | cut -f1)
    echo -e "${GREEN}✓ Checkpoint already exists: ${CHECKPOINT_PATH}${NC}"
    echo "  File size: ${FILE_SIZE}"

    # Verify size is reasonable (> 9GB)
    FILE_SIZE_BYTES=$(stat -f%z "${CHECKPOINT_PATH}" 2>/dev/null || stat -c%s "${CHECKPOINT_PATH}" 2>/dev/null)
    FILE_SIZE_GB=$(echo "scale=2; ${FILE_SIZE_BYTES} / 1073741824" | bc)

    if (( $(echo "${FILE_SIZE_GB} > 9" | bc -l) )); then
        echo -e "${GREEN}✓ File size looks correct (${FILE_SIZE_GB} GB)${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠ File size seems small (${FILE_SIZE_GB} GB). May be incomplete.${NC}"
        echo "  Re-run with --resume to continue download."
        if [ "$1" != "--resume" ]; then
            exit 1
        fi
    fi
fi

# Create checkpoint directory
echo ""
echo "Creating checkpoint directory: ${CHECKPOINT_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# Check for huggingface_hub
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    pip install huggingface_hub
fi

# Download checkpoint
echo ""
echo "Downloading checkpoint from HuggingFace..."
echo "  Repository: ${HF_REPO}"
echo "  File: ${HF_FILENAME}"
echo "  Destination: ${CHECKPOINT_PATH}"
echo ""

# Check for HF_TOKEN for authenticated downloads (avoids rate limits)
if [ -n "${HF_TOKEN}" ]; then
    echo -e "${GREEN}✓ Using HF_TOKEN for authenticated download${NC}"
fi

# Use Python to download with huggingface_hub
python3 << EOF
import os
import sys
from huggingface_hub import hf_hub_download

try:
    print("Starting download (this may take 10-20 minutes)...")
    local_path = hf_hub_download(
        repo_id="${HF_REPO}",
        filename="${HF_FILENAME}",
        local_dir="${CHECKPOINT_DIR}",
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✓ Downloaded to: {local_path}")

    # Verify the download
    import os
    file_size = os.path.getsize(local_path)
    file_size_gb = file_size / (1024**3)
    print(f"✓ File size: {file_size_gb:.2f} GB")

    if file_size_gb < 9:
        print("⚠ Warning: File size seems small. Download may be incomplete.")
        sys.exit(1)

except Exception as e:
    print(f"✗ Download failed: {e}")
    print("")
    print("Troubleshooting:")
    print("  1. Check your internet connection")
    print("  2. Try setting HF_TOKEN: export HF_TOKEN=your_token")
    print("  3. Login via: huggingface-cli login")
    print("  4. Re-run with --resume flag")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=============================================="
    echo "  ✓ Checkpoint download complete!"
    echo "=============================================="
    echo ""
    echo "  Location: ${CHECKPOINT_PATH}"
    echo ""
    echo "  Next step: Run the overfit test"
    echo "    ./scripts/run_overfit_test.sh"
    echo -e "==============================================${NC}"
else
    echo ""
    echo -e "${RED}✗ Download failed. See error above.${NC}"
    exit 1
fi
