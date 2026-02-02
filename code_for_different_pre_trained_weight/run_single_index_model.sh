#!/bin/bash
# ============================================================
# 🚀 Single-Index Teacher–Student Experiment Launcher
#     + Live Plot + Escape-Time Meta-Comparison
# ============================================================
#
# We now focus on the SINGLE-INDEX setting (K = 1).
#
# Teacher direction:        ω★  ∈ R^N  (unit / normalized)
# Random direction:         ξ   ∈ R^N  (independent, unit / normalized)
#
# Pre-activation / initialization direction:
#     w0 = μ ω★ + (1 - μ) ξ
#
# where μ ∈ [0,1] controls alignment with the teacher:
#   - μ = 1   → perfectly aligned init (w0 = ω★)
#   - μ = 0   → random init (w0 = ξ)
#   - intermediate μ → partially aligned init
#
# In the code below, this alignment parameter is swept via D_LIST.
# (Here D_LIST plays the role of μ.)
#
# You can change:
#   - ACTIVATIONS   : student/teacher nonlinearity σ
#   - D_LIST        : alignment values μ to sweep
#   - INPUT_DIM (N) : ambient dimension
#   - OUTPUT_DIM (K): set to 1 for single-index (one hidden neuron)
#   - LORA_RANK (r) : rank of the LoRA adaptation (kept = 1 here)
#
# ============================================================

# ===================== ⚙️ Config =====================
EPOCHS=5000
EPOCHS_SAVE_STEP=2
LEARNING_RATE=0.2
BATCH_SIZE=5000

INPUT_DIM=100     # N: ambient dimension
OUTPUT_DIM=1      # K=1: SINGLE-INDEX model (one neuron / one direction)
LORA_RANK=1       # r: LoRA rank (rank-1 adaptation in this single-index setup)

TEST_SIZE=1000
previous_epochs=True # set False to always restart from scratch

# Seeds and alignment sweep:
SEED_LIST=(21)
D_LIST=(0.1)      # here D_LIST = {μ values}; μ controls w0 = μ ω★ + (1-μ) ξ

# Activation function σ
ACTIVATIONS=("linear")

# Only used if ACTIVATIONS contains "hermite"
HERMITE_ORDERS=(2)

# ===================== 📁 Output dirs =====================
BASE_SAVE_DIR="./results_lr${LEARNING_RATE}_batch_size${BATCH_SIZE}_input_dimension${INPUT_DIM}"
BASE_LOG_DIR="./logs_teacher_student"

PLOT_SCRIPT="plot_dynamics.py"
PLOT_TIME_SCRIPT="plot_time_to_overlap.py"
COMPARE_SCRIPT="compare_time_to_overlap_all.py"  # meta-plot across activations
PLOT_REFRESH=10

mkdir -p "$BASE_SAVE_DIR" "$BASE_LOG_DIR"

# ===================== ⚡ Options =====================
# Run: ./run.sh plot_only
# → only re-generates plots (no training)
PLOT_ONLY=false
if [[ "$1" == "plot_only" ]]; then
    PLOT_ONLY=true
    echo "⚡ Plot-only mode activated. Skipping training..."
fi

# ===================== 🧱 Sweep Experiments =====================
for ACT in "${ACTIVATIONS[@]}"; do

    # ===== Hermite activations (optional branch) =====
    if [ "$ACT" == "hermite" ]; then
        for HE_ORDER in "${HERMITE_ORDERS[@]}"; do

            if [ "$PLOT_ONLY" = true ]; then
                RUN_NAME="${ACT}_He${HE_ORDER}_ALL_mu"
                echo "=============================================="
                echo "📊 Plot-only | activation=$ACT | He${HE_ORDER} | mu_list=${D_LIST[*]}"
                echo "=============================================="

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

            else
                for D in "${D_LIST[@]}"; do
                    # D is μ in the single-index pre-activation:
                    # w0 = μ ω★ + (1-μ) ξ
                    RUN_NAME="${ACT}_He${HE_ORDER}_mu${D}"
                    LOG_FILE="$BASE_LOG_DIR/train_${RUN_NAME}.log"

                    nohup python3 experiment.py \
                        --activation "$ACT" \
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

                    echo "💡 Launched training for μ=$D"

                    nohup python3 "$PLOT_SCRIPT" \
                        --base_dir "$BASE_SAVE_DIR" \
                        --activation "$ACT" \
                        --hermite_order "$HE_ORDER" \
                        --d_list "$D" \
                        --seed_list "${SEED_LIST[@]}" \
                        --compare \
                        > "$BASE_LOG_DIR/plot_${RUN_NAME}.log" 2>&1 &
                done

                nohup python3 "$PLOT_TIME_SCRIPT" \
                    --base_dir "$BASE_SAVE_DIR" \
                    --activation "${ACT}_He${HE_ORDER}" \
                    > "$BASE_LOG_DIR/time_to_overlap_${ACT}_He${HE_ORDER}.log" 2>&1 &
            fi
        done

    # ===== Standard activations =====
    else
        RUN_NAME="${ACT}"
        LOG_FILE="$BASE_LOG_DIR/train_${RUN_NAME}.log"

        if [ "$PLOT_ONLY" = false ]; then
            # d_list here is the μ-sweep for the single-index initialization
            nohup python3 experiment.py \
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

        # Live dynamics plots (overlaps, losses)
        nohup python3 "$PLOT_SCRIPT" \
            --base_dir "$BASE_SAVE_DIR" \
            --activation "$ACT" \
            --d_list "${D_LIST[@]}" \
            --seed_list "${SEED_LIST[@]}" \
            --compare \
            > "$BASE_LOG_DIR/plot_${RUN_NAME}.log" 2>&1 &

        # Escape-time plots (time-to-overlap)
        nohup python3 "$PLOT_TIME_SCRIPT" \
            --base_dir "$BASE_SAVE_DIR" \
            --activation "$ACT" \
            > "$BASE_LOG_DIR/time_to_overlap_${RUN_NAME}.log" 2>&1 &
    fi
done

# ===================== ✅ Final meta-comparison =====================
echo "🖌️ Generating combined time-to-overlap plot for all activations..."
nohup python3 "$COMPARE_SCRIPT" \
    --base_dir "$BASE_SAVE_DIR" \
    > "$BASE_LOG_DIR/compare_time_to_overlap_all.log" 2>&1 &

echo "✅ All experiments, plotting tasks, and comparison launched! (training skipped: $PLOT_ONLY)"
