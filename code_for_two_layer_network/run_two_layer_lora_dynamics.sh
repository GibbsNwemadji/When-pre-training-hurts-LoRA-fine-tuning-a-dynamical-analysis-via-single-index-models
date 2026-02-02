#!/bin/bash
# ============================================================
# 🚀 Teacher–Student LoRA Experiment Launcher
# ============================================================
#
# This script runs experiments for a TWO-LAYER neural network
# in a teacher–student setting with Low-Rank Adaptation (LoRA).
#
# Architecture (conceptual):
#   Input (dimension N)
#      ↓
#   Hidden layer (width K, activation σ)
#      ↓   + LoRA rank-r adaptation
#   Output
#
# By editing:
#   - ACTIVATIONS        → activation function σ
#   - OUTPUT_DIM (K)     → network width
#   - LORA_RANK (r)      → LoRA rank
#
# the same code runs for different neural architectures
# and fine-tuning regimes.
#
# ============================================================

# ===================== ⚙️ Global Training Config =====================
EPOCHS=10000                # Number of SGD steps
EPOCHS_SAVE_STEP=2          # Save model every N epochs
REALIZATIONS=2              # Number of independent runs
LEARNING_RATE=0.7
BATCH_SIZE=5000
INPUT_DIM=100               # Input dimension N
OUTPUT_DIM=4                # Width K of the hidden layer
LORA_RANK=1                 # LoRA rank r
TEST_SIZE=1000
previous_epochs=True        # Resume from previous checkpoints if available

# ===================== 🔁 Experiment Sweep Parameters =====================
SEED_LIST=(21)              # Random seeds
D_LIST=(0.9)                # Alignment / difficulty parameter μ or d

# Activation of the hidden layer
# Can be changed to: relu, linear, hermite, etc.
ACTIVATIONS=("relu")

# Hermite order (used only if activation="hermite")
HERMITE_ORDERS=(2)

# ===================== 📁 Output Directories =====================
# Results and logs are automatically organized by
# learning rate, width K, LoRA rank r, batch size, and input dim N
BASE_SAVE_DIR="./results_lr${LEARNING_RATE}_width${OUTPUT_DIM}_rank${LORA_RANK}batch_size${BATCH_SIZE}_input_dimension${INPUT_DIM}"
BASE_LOG_DIR="./logs_teacher_student${LEARNING_RATE}_width${OUTPUT_DIM}_rank${LORA_RANK}batch_size${BATCH_SIZE}_input_dimension${INPUT_DIM}"

PLOT_SCRIPT="plot_dynamics.py"                 # Live dynamics plots
PLOT_TIME_SCRIPT="plot_time_to_overlap.py"     # Escape-time plots
COMPARE_SCRIPT="compare_time_to_overlap_all.py" # Meta-comparison across runs
PLOT_REFRESH=10

mkdir -p "$BASE_SAVE_DIR" "$BASE_LOG_DIR"

# ===================== ⚡ Plot-Only Mode =====================
# Run: ./script.sh plot_only
# → skips training and only regenerates plots
PLOT_ONLY=false
if [[ "$1" == "plot_only" ]]; then
    PLOT_ONLY=true
    echo "⚡ Plot-only mode activated. Skipping training..."
fi

# ===================== 🧱 Main Experiment Loop =====================
for ACT in "${ACTIVATIONS[@]}"; do

    # ===== Case 1: Hermite activation (parameterized by order) =====
    if [ "$ACT" == "hermite" ]; then
        for HE_ORDER in "${HERMITE_ORDERS[@]}"; do

            # ----- Plot-only mode -----
            if [ "$PLOT_ONLY" = true ]; then
                RUN_NAME="${ACT}_He${HE_ORDER}_ALL_D"

                nohup python3 "$PLOT_SCRIPT" \
                    --base_dir "$BASE_SAVE_DIR" \
                    --activation "$ACT" \
                    --hermite_order "$HE_ORDER" \
                    --d_list "${D_LIST[@]}" \
                    --seed_list "${SEED_LIST[@]}" \
                    --compare \
                    > "$BASE_LOG_DIR/plot_${RUN_NAME}.log" 2>&1 &

                nohup python3 "$PLOT_TIME_SCRIPT" \
                    --base_dir "$BASE_SAVE_DIR" \
                    --activation "${ACT}_He${HE_ORDER}" \
                    > "$BASE_LOG_DIR/time_to_overlap_${RUN_NAME}.log" 2>&1 &

            # ----- Full training mode -----
            else
                for D in "${D_LIST[@]}"; do
                    RUN_NAME="${ACT}_He${HE_ORDER}_D${D}"
                    SAVE_DIR="$BASE_SAVE_DIR/$RUN_NAME"
                    LOG_FILE="$BASE_LOG_DIR/train_${RUN_NAME}.log"
                    mkdir -p "$SAVE_DIR"

                    # Main training call:
                    # This runs SGD on a two-layer network with
                    # activation σ, width K, and LoRA rank r
                    nohup python3 experiment.py \
                        --activation "$ACT" \
                        --base_dir "$BASE_SAVE_DIR" \
                        --previous_epochs "$previous_epochs" \
                        --hermite_order "$HE_ORDER" \
                        --epochs "$EPOCHS" \
                        --learning_rate "$LEARNING_RATE" \
                        --batch_size "$BATCH_SIZE" \
                        --N "$INPUT_DIM" \
                        --K "$OUTPUT_DIM" \
                        --r "$LORA_RANK" \
                        --epochs_save_Step "$EPOCHS_SAVE_STEP" \
                        --test_size "$TEST_SIZE" \
                        --d_list "$D" \
                        --seed_list "${SEED_LIST[@]}" \
                        > "$LOG_FILE" 2>&1 &

                    echo "💡 Launched training for D=$D"
                done

                nohup python3 "$PLOT_TIME_SCRIPT" \
                    --base_dir "$BASE_SAVE_DIR" \
                    --activation "${ACT}_He${HE_ORDER}" \
                    > "$BASE_LOG_DIR/time_to_overlap_${ACT}_He${HE_ORDER}.log" 2>&1 &
            fi
        done

    # ===== Case 2: Standard activations (e.g. ReLU, linear) =====
    else
        RUN_NAME="${ACT}"
        SAVE_DIR="$BASE_SAVE_DIR/$RUN_NAME"
        LOG_FILE="$BASE_LOG_DIR/train_${RUN_NAME}.log"
        mkdir -p "$SAVE_DIR"

        if [ "$PLOT_ONLY" = false ]; then
            nohup python3 experiment.py \
                --base_dir "$BASE_SAVE_DIR" \
                --activation "$ACT" \
                --previous_epochs "$previous_epochs" \
                --epochs "$EPOCHS" \
                --learning_rate "$LEARNING_RATE" \
                --batch_size "$BATCH_SIZE" \
                --N "$INPUT_DIM" \
                --K "$OUTPUT_DIM" \
                --r "$LORA_RANK" \
                --epochs_save_Step "$EPOCHS_SAVE_STEP" \
                --test_size "$TEST_SIZE" \
                --d_list "${D_LIST[@]}" \
                --seed_list "${SEED_LIST[@]}" \
                > "$LOG_FILE" 2>&1 &
        fi

        nohup python3 "$PLOT_SCRIPT" \
            --base_dir "$BASE_SAVE_DIR" \
            --activation "$ACT" \
            --d_list "${D_LIST[@]}" \
            --seed_list "${SEED_LIST[@]}" \
            --compare \
            > "$BASE_LOG_DIR/plot_${RUN_NAME}.log" 2>&1 &

        nohup python3 "$PLOT_TIME_SCRIPT" \
            --base_dir "$BASE_SAVE_DIR" \
            --activation "$ACT" \
            > "$BASE_LOG_DIR/time_to_overlap_${RUN_NAME}.log" 2>&1 &
    fi
done

# ===================== 📊 Final Meta-Comparison =====================
# Compares escape times across all activations / architectures
echo "🖌️ Generating combined time-to-overlap plot..."
nohup python3 "$COMPARE_SCRIPT" \
    --base_dir "$BASE_SAVE_DIR" \
    > "$BASE_LOG_DIR/compare_time_to_overlap_all.log" 2>&1 &

echo "✅ All experiments and plots launched! (training skipped: $PLOT_ONLY)"
