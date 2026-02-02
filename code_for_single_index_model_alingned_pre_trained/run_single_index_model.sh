#!/bin/bash
# ============================================================
# 🚀 Single-Index (K=1) Teacher–Student Launcher
#     Pretrained initialization = μ ω★
#     + live plots + escape-time meta-comparison
# ============================================================
#
# SINGLE-INDEX setting:
#   - one teacher direction ω★ ∈ R^N
#   - one student direction  w ∈ R^N
#   - labels: y = σ(x · ω★)
#
# PRETRAINED INITIALIZATION (this script):
#   The student starts already aligned with the teacher:
#       w0(μ) = μ ω★
#   where μ ∈ (0,1] controls the strength of pretraining / alignment.
#
# Interpretation of μ:
#   - μ = 1.0  : perfectly pretrained init (w0 = ω★)
#   - μ small  : weak pretraining (small aligned signal)
#
# In the code below:
#   - OUTPUT_DIM=1 enforces the single-index case (K=1).
#   - D_LIST sweeps the values of μ.
#   - ACTIVATIONS selects σ (linear / relu / hermite / etc.).
#
# Note: the actual implementation of "w0(μ)=μ ω★" is done in experiment.py.
# This script only launches runs for many μ and seeds, and calls plotting tools.
# ============================================================

# ===================== ⚙️ Config =====================
EPOCHS=10000
EPOCHS_SAVE_STEP=2
LEARNING_RATE=0.2
BATCH_SIZE=500

INPUT_DIM=1000        # N: ambient dimension
OUTPUT_DIM=1          # K=1: single-index
LORA_RANK=1           # r: rank of LoRA adaptation (kept at 1 here)

TEST_SIZE=1000
previous_epochs=True  # set False to restart each run from scratch

# Seeds and μ-sweep
SEED_LIST=(20 21 24)
D_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)   # here D_LIST = μ values

# Activation σ
ACTIVATIONS=("linear")
HERMITE_ORDERS=(2)    # only used if ACTIVATIONS includes "hermite"

# ===================== 📁 Output dirs =====================
BASE_SAVE_DIR="./results_lr${LEARNING_RATE}_batch_size${BATCH_SIZE}_input_dimension${INPUT_DIM}"
BASE_LOG_DIR="./logs_teacher_student"

PLOT_SCRIPT="plot_dynamics.py"
PLOT_TIME_SCRIPT="plot_time_to_overlap.py"
COMPARE_SCRIPT="compare_time_to_overlap_all.py"
PLOT_REFRESH=10

mkdir -p "$BASE_SAVE_DIR" "$BASE_LOG_DIR"

# ===================== ⚡ Plot-only mode =====================
# Run: ./run_single_index_mu_pretrained.sh plot_only
# → skips training and only regenerates plots from existing results
PLOT_ONLY=false
if [[ "$1" == "plot_only" ]]; then
    PLOT_ONLY=true
    echo "⚡ Plot-only mode activated. Skipping training..."
fi

# ===================== 🧱 Sweep Experiments =====================
for ACT in "${ACTIVATIONS[@]}"; do

    # ===== Branch for Hermite activation =====
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
                # Training mode: launch one job per μ
                for D in "${D_LIST[@]}"; do
                    # Here D is μ in w0(μ) = μ ω★
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

                    echo "💡 Launched training for μ=$D (activation=$ACT, He${HE_ORDER})"

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

    # ===== Standard activation branch =====
    else
        RUN_NAME="${ACT}"
        LOG_FILE="$BASE_LOG_DIR/train_${RUN_NAME}.log"

        if [ "$PLOT_ONLY" = false ]; then
            # d_list here is the μ sweep for pretrained init w0(μ)=μ ω★
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

        # Live plots of dynamics (loss / overlaps)
        nohup python3 "$PLOT_SCRIPT" \
            --base_dir "$BASE_SAVE_DIR" \
            --activation "$ACT" \
            --d_list "${D_LIST[@]}" \
            --seed_list "${SEED_LIST[@]}" \
            --compare \
            > "$BASE_LOG_DIR/plot_${RUN_NAME}.log" 2>&1 &

        # Time-to-overlap / escape-time plots
        nohup python3 "$PLOT_TIME_SCRIPT" \
            --base_dir "$BASE_SAVE_DIR" \
            --activation "$ACT" \
            > "$BASE_LOG_DIR/time_to_overlap_${RUN_NAME}.log" 2>&1 &
    fi
done

# ============================================================
# ✅ Final meta-comparison across activations (if multiple)
# ============================================================
echo "🖌️ Generating combined time-to-overlap plot for all activations..."
nohup python3 "$COMPARE_SCRIPT" \
    --base_dir "$BASE_SAVE_DIR" \
    > "$BASE_LOG_DIR/compare_time_to_overlap_all.log" 2>&1 &

echo "✅ All experiments, plotting tasks, and comparison launched! (training skipped: $PLOT_ONLY)"
