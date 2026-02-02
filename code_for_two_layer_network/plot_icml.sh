#!/bin/bash
# ============================================================
# 📊 Activation (rows) × Rank (cols) Grid Plot Launcher (Test Loss)
# ============================================================

# ===================== ⚙️ Config =====================
LEARNING_RATE=0.2
WIDTH=4                 # K (from "widthK" in folder name)
BATCH_SIZE=5000
INPUT_DIM=100           # N (from "input_dimensionN" in folder name)

# Parent directory that contains results_lr... folders
BASE_PARENT="."

# Curves to overlay in each subplot
D_LIST=(0.1 0.5 0.9)

# Script name (the updated one)
PLOT_SCRIPT="plot_rank_and_activation_grid.py"   # or whatever you saved it as

# Outputs
BASE_LOG_DIR="./logs_grid_rowsACT_colsRANK_lr${LEARNING_RATE}_width${WIDTH}_bs${BATCH_SIZE}_N${INPUT_DIM}"
mkdir -p "$BASE_LOG_DIR"

RUN_NAME="grid_rowsACT_colsRANK_lr${LEARNING_RATE}_width${WIDTH}_bs${BATCH_SIZE}_N${INPUT_DIM}"
LOG_FILE="$BASE_LOG_DIR/${RUN_NAME}.log"

echo "=============================================="
echo "🖌️ Launching Activation×Rank grid plot (test loss)"
echo "  base_parent: $BASE_PARENT"
echo "  lr:         $LEARNING_RATE"
echo "  width(K):   $WIDTH"
echo "  bs:         $BATCH_SIZE"
echo "  N:          $INPUT_DIM"
echo "  d_list:     ${D_LIST[*]}"
echo "  script:     $PLOT_SCRIPT"
echo "=============================================="

nohup python3 "$PLOT_SCRIPT" \
  --base_parent "$BASE_PARENT" \
  --lr "$LEARNING_RATE" \
  --width "$WIDTH" \
  --batch_size "$BATCH_SIZE" \
  --N "$INPUT_DIM" \
  --d_list "${D_LIST[@]}" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Plot job launched (PID: $PID)"
echo "📄 Logs: $LOG_FILE"