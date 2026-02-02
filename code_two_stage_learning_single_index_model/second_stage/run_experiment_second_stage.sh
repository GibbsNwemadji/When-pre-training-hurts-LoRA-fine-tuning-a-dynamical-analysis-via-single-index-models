#!/bin/bash
# ============================================================
# 🚀 Stage-2 (Fine-tuning) launcher — Teacher/Student single-index + LoRA
# ============================================================
#
# This script runs the **SECOND STAGE** of your two-stage learning pipeline.
#
# Setting (single-index, K=1):
#   - Inputs:        x ~ N(0, I_N)
#   - Teacher label: y = φ(x · ω★)
#   - Stage-2 target used for training is the **ground-truth label**:
#                   y_stage2 = y   (NOT squared)
#   - Student predicts: ŷ = σ(x · w)
#   - Loss:         1/2 E[(ŷ - y)^2]
#
# IMPORTANT:
#   The only difference vs Stage-1 is inside experiment.py:
#     Stage-1: loss uses (ŷ - y^2)^2
#     Stage-2: loss uses (ŷ - y)^2
#
# Pre-training strength / alignment knob (D, often μ in your theory):
#   - experiment.py uses D to build the student initialization, e.g.
#         w = D * ω★ + (LoRA correction)
#   - D close to 1 means “strong pretraining / strong alignment”.
#
# What this launcher does:
#   1) Sweep over teacher activations (φ) + Hermite order
#   2) Sweep over student activations (σ) + Hermite order
#   3) Sweep over D values
#   4) Launch training + plotting scripts in parallel
#
# Usage:
#   ./run_stage2.sh
#   ./run_stage2.sh plot_only     # skip training, only regenerate plots
# ============================================================


# ===================== ⚙️ Core hyperparameters =====================
EPOCHS=1000
EPOCHS_SAVE_STEP=2

LEARNING_RATE=0.01
BATCH_SIZE=1000

INPUT_DIM=1000     # N (dimension of x)
OUTPUT_DIM=1       # K (single-index => K=1)
LORA_RANK=1        # r (LoRA rank)

TEST_SIZE=1000
previous_epochs=True   # resume from saved .npz if it exists

# ===================== 🔁 Sweep settings =====================
SEED_LIST=(20 21 24)

# D is your pretraining strength / alignment parameter (often μ)
D_LIST=(0.325)

# --- Teacher activation φ ---
TEACHER_ACTIVATIONS=("hermite")
TEACHER_HERMITE_ORDERS=(3)

# --- Student activation σ ---
STUDENT_ACTIVATIONS=("hermite")
STUDENT_HERMITE_ORDERS=(3)

# ===================== 📁 Output folders =====================
BASE_SAVE_DIR="./results_lr${LEARNING_RATE}_batch_size${BATCH_SIZE}_input_dimension${INPUT_DIM}"
BASE_LOG_DIR="./logs_teacher_student"

# ===================== 📈 Plotting utilities =====================
PLOT_SCRIPT="plot_dynamics.py"
PLOT_TIME_SCRIPT="plot_time_to_overlap.py"
COMPARE_SCRIPT="compare_time_to_overlap_all.py"

mkdir -p "$BASE_SAVE_DIR" "$BASE_LOG_DIR"

# ===================== ⚡ Plot-only mode =====================
PLOT_ONLY=false
if [[ "$1" == "plot_only" ]]; then
    PLOT_ONLY=true
    echo "⚡ Plot-only mode activated. Skipping training..."
fi


# ============================================================
# 🧱 Sweep experiments (teacher φ, student σ, and D)
# ============================================================
for T_ACT_IDX in "${!TEACHER_ACTIVATIONS[@]}"; do
    TEACHER_ACT=${TEACHER_ACTIVATIONS[$T_ACT_IDX]}
    TEACHER_HE_ORDER=${TEACHER_HERMITE_ORDERS[$T_ACT_IDX]}

    for S_ACT_IDX in "${!STUDENT_ACTIVATIONS[@]}"; do
        STUDENT_ACT=${STUDENT_ACTIVATIONS[$S_ACT_IDX]}
        STUDENT_HE_ORDER=${STUDENT_HERMITE_ORDERS[$S_ACT_IDX]}

        # ------------------------------
        # Plot-only: regenerate plots
        # ------------------------------
        if [ "$PLOT_ONLY" = true ]; then
            RUN_NAME="T_${TEACHER_ACT}_He${TEACHER_HE_ORDER}_S_${STUDENT_ACT}_He${STUDENT_HE_ORDER}_ALL_D"

            echo "=============================================="
            echo "📊 Plot-only mode (Stage-2)"
            echo "Teacher : $TEACHER_ACT (He${TEACHER_HE_ORDER})"
            echo "Student : $STUDENT_ACT (He${STUDENT_HE_ORDER})"
            echo "D sweep : ${D_LIST[*]}"
            echo "Seeds   : ${SEED_LIST[*]}"
            echo "=============================================="

            nohup python3 "$PLOT_SCRIPT" \
                --base_dir "$BASE_SAVE_DIR" \
                --teacher_activation "$TEACHER_ACT" \
                --teacher_hermite_order "$TEACHER_HE_ORDER" \
                --student_activation "$STUDENT_ACT" \
                --student_hermite_order "$STUDENT_HE_ORDER" \
                --d_list "${D_LIST[@]}" \
                --seed_list "${SEED_LIST[@]}" \
                --compare \
                > "$BASE_LOG_DIR/plot_${RUN_NAME}.log" 2>&1 &

            nohup python3 "$PLOT_TIME_SCRIPT" \
                --base_dir "$BASE_SAVE_DIR" \
                --teacher_activation "${TEACHER_ACT}_He${TEACHER_HE_ORDER}" \
                --student_activation "${STUDENT_ACT}_He${STUDENT_HE_ORDER}" \
                > "$BASE_LOG_DIR/time_to_overlap_${RUN_NAME}.log" 2>&1 &

        # ------------------------------
        # Full mode: train + plot
        # ------------------------------
        else
            for D in "${D_LIST[@]}"; do
                RUN_NAME="T_${TEACHER_ACT}_He${TEACHER_HE_ORDER}_S_${STUDENT_ACT}_He${STUDENT_HE_ORDER}_D${D}"
                LOG_FILE="$BASE_LOG_DIR/train_${RUN_NAME}.log"

                # 1) Launch training (Stage 2: targets are the true labels y)
                nohup python3 experiment.py \
                    --teacher_activation "$TEACHER_ACT" \
                    --teacher_hermite_order "$TEACHER_HE_ORDER" \
                    --student_activation "$STUDENT_ACT" \
                    --student_hermite_order "$STUDENT_HE_ORDER" \
                    --previous_epochs "$previous_epochs" \
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

                TRAIN_PID=$!
                echo "💡 Launched Stage-2 training for D=$D (PID: $TRAIN_PID)"

                # 2) Launch plotting for this run
                nohup python3 "$PLOT_SCRIPT" \
                    --base_dir "$BASE_SAVE_DIR" \
                    --teacher_activation "$TEACHER_ACT" \
                    --teacher_hermite_order "$TEACHER_HE_ORDER" \
                    --student_activation "$STUDENT_ACT" \
                    --student_hermite_order "$STUDENT_HE_ORDER" \
                    --d_list "$D" \
                    --seed_list "${SEED_LIST[@]}" \
                    --compare \
                    > "$BASE_LOG_DIR/plot_${RUN_NAME}.log" 2>&1 &
            done

            # 3) Time-to-overlap plot for this teacher/student pair (across D)
            nohup python3 "$PLOT_TIME_SCRIPT" \
                --base_dir "$BASE_SAVE_DIR" \
                --teacher_activation "${TEACHER_ACT}_He${TEACHER_HE_ORDER}" \
                --student_activation "${STUDENT_ACT}_He${STUDENT_HE_ORDER}" \
                > "$BASE_LOG_DIR/time_to_overlap_${TEACHER_ACT}_He${TEACHER_HE_ORDER}_S_${STUDENT_ACT}_He${STUDENT_HE_ORDER}.log" 2>&1 &
        fi
    done
done


# ============================================================
# ✅ Meta-comparison plot across all teacher/student pairs
# ============================================================
echo "🖌️ Generating combined time-to-overlap plot for all teacher/student activations (Stage-2)..."
nohup python3 "$COMPARE_SCRIPT" \
    --base_dir "$BASE_SAVE_DIR" \
    > "$BASE_LOG_DIR/compare_time_to_overlap_all.log" 2>&1 &

echo "✅ Stage-2 runs launched! (training skipped: $PLOT_ONLY)"
