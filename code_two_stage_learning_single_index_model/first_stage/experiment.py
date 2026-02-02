import os
import math
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import matplotlib.pyplot as plt


# ============================================================
# Two-stage learning — STAGE 1 (pre-training with squared labels)
# ============================================================
#
# Single-index teacher–student setup (K=1 by default).
#
# Teacher:
#   - draw x ~ N(0, I_N)
#   - teacher direction ω★ (named T in code), ||T||=1
#   - teacher activation  φ(·)
#   - "raw" teacher label:        y = φ(x · ω★)
#
# Stage-1 twist (two-stage learning):
#   - we train the student on SQUARED labels:
#         y_stage1 = y^2 = [φ(x · ω★)]^2
#
# Student:
#   - student activation σ(·)
#   - student direction w (named S in code)
#
# Pretraining / initialization knob "d" (often your μ):
#   - if d == "V_only": student is just V and we train V directly
#   - else: student weights are initialized/structured as
#         S = d * T + U V
#     where U V is a rank-r LoRA correction.
#
# The script runs multiple seeds in parallel and aggregates trajectories.
# ============================================================

torch.set_num_threads(1)


# ------------------------------------------------------------
# Activations
# ------------------------------------------------------------
def hermite_He(x: torch.Tensor, n: int) -> torch.Tensor:
    """Probabilists' Hermite polynomial He_n(x), implemented up to n=6."""
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x
    if n == 2:
        return x**2 - 1
    if n == 3:
        return x**3 - 3 * x
    if n == 4:
        return x**4 - 6 * x**2 + 3
    if n == 5:
        return x**5 - 10 * x**3 + 15 * x
    if n == 6:
        return x**6 - 15 * x**4 + 45 * x**2 - 15
    raise ValueError("Hermite order not implemented (max 6 for now).")


def build_activation(name: str, order: int | None = None, next_freq: int = 1):
    """
    Return an activation function (callable) given a name.

    For Hermite:
      - 'hermite'  : He_order
      - 'hermite+' : He_order + He_(order + next_freq)
    """
    if name == "linear":
        return lambda x: x
    if name == "square":
        return lambda x: x**2
    if name == "erf":
        return torch.erf
    if name == "relu":
        return torch.relu
    if name == "sigmoid":
        return torch.sigmoid

    if name == "hermite":
        if order is None:
            raise ValueError("Hermite activation requires *_hermite_order")
        return lambda x: hermite_He(x, order)

    if name == "hermite+":
        if order is None:
            raise ValueError("Hermite+ requires *_hermite_order")
        return lambda x: hermite_He(x, order) + hermite_He(x, order + next_freq)

    raise ValueError(f"Unknown activation {name}")


# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------
def save_overlap_plot(overlap_history, seed, d, result_dir):
    """Save overlap trajectory |<T, V>| over epochs."""
    plt.figure()
    plt.plot(overlap_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Overlap |⟨ω★, V⟩|")
    plt.title(f"Overlap vs Epoch (seed={seed}, d={d})")
    plt.grid(True)
    save_path = os.path.join(result_dir, f"plot_overlap_v_teacher_seed_{seed}_d_{d}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📁 Saved overlap plot to: {save_path}")


# ------------------------------------------------------------
# Utility: align variable-length trajectories across seeds
# ------------------------------------------------------------
def align_arrays(list_of_arrays, mode="truncate"):
    """
    Align a list of 1D arrays to a common length.
      - truncate: cut all to shortest length
      - pad:      extend shorter arrays by repeating last value
    """
    if not list_of_arrays:
        return list_of_arrays

    if mode == "truncate":
        min_len = min(len(a) for a in list_of_arrays)
        return [a[:min_len] for a in list_of_arrays]

    if mode == "pad":
        max_len = max(len(a) for a in list_of_arrays)

        def pad(a, target):
            return np.pad(a, (0, target - len(a)), mode="edge")

        return [pad(a, max_len) for a in list_of_arrays]

    raise ValueError(f"Unknown mode '{mode}', use 'truncate' or 'pad'.")


# ------------------------------------------------------------
# Single seed run (one d, one seed)
# ------------------------------------------------------------
def single_seed_run(seed, args, T_cpu, X_test_cpu, y_test_cpu, d, result_dir):
    """
    One training run for a fixed random seed and fixed d.

    Training objective (STAGE 1):
        y = φ(X·T)            (teacher raw label)
        target = y^2          (squared label)
        y_hat = σ(X·S)        (student prediction)
        loss = 1/2 E[(y_hat - target)^2]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build activations for teacher and student
    teacher_act = build_activation(
        args.teacher_activation,
        args.teacher_hermite_order,
        args.teacher_next_freq,
    )
    student_act = build_activation(
        args.student_activation,
        args.student_hermite_order,
        args.student_next_freq,
    )

    N, K, r = args.N, args.K, args.r
    B, lr, epochs = args.batch_size, args.learning_rate, args.epochs

    # Optional LR scaling for Hermite student activation
    if args.student_activation == "hermite":
        lr = lr / (math.factorial(args.student_hermite_order) * args.student_hermite_order)
    elif args.student_activation == "hermite+":
        # NOTE: you likely meant args.student_next_freq here, not args.next_freq
        n = args.student_hermite_order + args.student_next_freq
        lr = lr / (math.factorial(n) * n)

    # Move fixed tensors to device inside subprocess
    T = T_cpu.to(device)
    X_test = X_test_cpu.to(device)
    y_test = y_test_cpu.to(device)

    # =========================================================
    # Student parametrization
    # =========================================================
    if d == "V_only":
        # Pure training of V (no pretrained term, no U)
        V = nn.Parameter(torch.randn(K, N, device=device))
        with torch.no_grad():
            V /= V.norm()
        U = None
        optimizer = optim.SGD([V], lr=lr)

        def forward_weights():
            return V

    else:
        # Pretrained term is d * T, plus a rank-r LoRA correction U@V
        D = torch.ones(K, N, device=device) * float(d)

        U = nn.Parameter(torch.ones(K, r, device=device) / math.sqrt(N**4))
        V = nn.Parameter(torch.randn(r, N, device=device))
        with torch.no_grad():
            V /= (V.norm(dim=1, keepdim=True) + 1e-12)

        optimizer = optim.SGD([U, V], lr=lr)

        def forward_weights():
            return D * T + U @ V

    # =========================================================
    # Resume from disk if requested
    # =========================================================
    safe_d = d if isinstance(d, str) else f"{d}"
    file_path = os.path.join(result_dir, f"d_{safe_d}_seed_{seed}.npz")

    if args.previous_epochs and os.path.exists(file_path):
        data = np.load(file_path)
        train_loss_epoch = list(data["train"])
        test_loss_epoch = list(data["test"])
        overlap_v_teacher_epoch = list(data["overlap_v_teacher"])
        overlap_u_teacher_epoch = list(data["overlap_u_teacher"])
        overlap_uv_teacher_epoch = list(data["overlap_uv_teacher"])
        overlap_student_teacher_epoch = list(data["overlap_student_teacher"])

        if d != "V_only":
            U = nn.Parameter(torch.tensor(data["U"], device=device))
            V = nn.Parameter(torch.tensor(data["V"], device=device))
            optimizer = optim.SGD([U, V], lr=lr)
        else:
            V = nn.Parameter(torch.tensor(data["V"], device=device))
            U = None
            optimizer = optim.SGD([V], lr=lr)

        print(f"✅ Loaded previous model for seed={seed}, d={d}")
    else:
        print("ℹ️ No checkpoint found — starting fresh.")
        train_loss_epoch, test_loss_epoch = [], []
        overlap_v_teacher_epoch, overlap_u_teacher_epoch = [], []
        overlap_uv_teacher_epoch, overlap_student_teacher_epoch = [], []

    # =========================================================
    # Early stopping parameters
    # =========================================================
    early_stop_patience = 8000
    high_overlap_threshold = float(d) + 0.01 if d != "V_only" else 0.98
    max_epochs = getattr(args, "max_epochs", 13000)
    no_improve_count = 0

    # =========================================================
    # Training loop
    # =========================================================
    current_epoch = 0
    total_epochs = epochs

    while current_epoch < total_epochs:
        if current_epoch >= max_epochs:
            print(f"🛑 Reached max_epochs={max_epochs}, stopping.")
            break

        # Sample batch
        X = torch.randn(B, N, device=device)

        # Teacher raw label
        y = teacher_act(X @ T.T)  # shape (B, K) for K=1

        # Student prediction
        S = forward_weights()
        y_pred = student_act(X @ S.T)

        # -------- Stage 1 loss: squared labels --------
        target = y**2
        loss = 0.5 * ((y_pred - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Normalize V for stability
        with torch.no_grad():
            if d == "V_only":
                V /= V.norm()
            else:
                V /= (V.norm(dim=1, keepdim=True) + 1e-12)

        train_loss_epoch.append(loss.item())

        # ---- Validation / overlaps ----
        with torch.no_grad():
            S_test = forward_weights()

            # NOTE: your original code used X_test @ S.T (training S) here,
            # but S_test is the one you want on test.
            y_pred_test = student_act(X_test @ S_test.T)
            test_loss = (0.5 * (y_pred_test - y_test) ** 2).mean()

            overlap_v_teacher = torch.abs(T @ V.T).mean().item()

            if d != "V_only":
                overlap_uv_teacher = (U * (T @ V.T)).mean().item()
                overlap_u_teacher = torch.abs(U).mean().item()
            else:
                overlap_uv_teacher = overlap_v_teacher
                overlap_u_teacher = 0.0

            overlap_student_teacher = (T @ S_test.T).mean().item()

        test_loss_epoch.append(test_loss.item())
        overlap_v_teacher_epoch.append(overlap_v_teacher)
        overlap_u_teacher_epoch.append(overlap_u_teacher)
        overlap_uv_teacher_epoch.append(overlap_uv_teacher)
        overlap_student_teacher_epoch.append(overlap_student_teacher)

        # ---- Save plot occasionally ----
        if (current_epoch % 100 == 0) and (current_epoch > 0):
            save_overlap_plot(overlap_v_teacher_epoch, seed, safe_d, result_dir)

        # ---- Early stopping logic ----
        if overlap_v_teacher > high_overlap_threshold:
            no_improve_count += 1
        else:
            no_improve_count = 0

        if no_improve_count >= early_stop_patience and current_epoch > 100:
            print(f"✅ Early stopping at epoch {current_epoch} (overlap_v_teacher > {high_overlap_threshold:.3f})")
            print(f"Current overlap_v_teacher: {overlap_v_teacher:.4f}")
            break

        # ---- Extend if needed ----
        if current_epoch == epochs - 1 and overlap_v_teacher <= high_overlap_threshold:
            print(f"⚠️ Extending training up to {max_epochs} (overlap_v_teacher={overlap_v_teacher:.3f})")
            total_epochs = min(max_epochs, epochs + (max_epochs - epochs))

        # ---- Save checkpoint ----
        os.makedirs(result_dir, exist_ok=True)
        np.savez(
            file_path,
            train=np.array(train_loss_epoch, dtype=float),
            test=np.array(test_loss_epoch, dtype=float),
            overlap_v_teacher=np.array(overlap_v_teacher_epoch, dtype=float),
            overlap_u_teacher=np.array(overlap_u_teacher_epoch, dtype=float),
            overlap_uv_teacher=np.array(overlap_uv_teacher_epoch, dtype=float),
            overlap_student_teacher=np.array(overlap_student_teacher_epoch, dtype=float),
            d=d,
            U=U.detach().cpu().numpy() if U is not None else np.array([]),
            V=V.detach().cpu().numpy(),
            seed=seed,
        )

        current_epoch += 1

    return seed


# ------------------------------------------------------------
# Experiment runner (parallel over seeds, sweep d)
# ------------------------------------------------------------
def run_experiment(args):
    print("Parent process - using spawn method for multiprocessing.")

    N, K, test_size = args.N, args.K, args.test_size
    torch.manual_seed(0)

    # Teacher direction ω★
    T = torch.randn(K, N)
    T /= T.norm()

    # Test set + teacher test labels (RAW teacher labels, not squared here)
    X_test = torch.randn(test_size, N)
    teacher_act = build_activation(args.teacher_activation, args.teacher_hermite_order, args.teacher_next_freq)
    y_test = teacher_act(X_test @ T.T)

    # Build a readable experiment name
    def act_tag(prefix, act, order, next_freq):
        if act not in ("hermite", "hermite+"):
            return f"{prefix}_{act}"
        if act == "hermite+":
            return f"{prefix}_hermite_He{order}+He{order+next_freq}"
        return f"{prefix}_hermite_He{order}"

    exp_name = (
        act_tag("T", args.teacher_activation, args.teacher_hermite_order, args.teacher_next_freq)
        + "_"
        + act_tag("S", args.student_activation, args.student_hermite_order, args.student_next_freq)
    )

    result_dir = os.path.join(
        f"results_lr{args.learning_rate}_batch_size{args.batch_size}_input_dimension{args.N}",
        exp_name,
    )
    os.makedirs(result_dir, exist_ok=True)

    # Parse d list (keep "V_only" string)
    d_list_clean = []
    for el in args.d_list:
        if isinstance(el, str) and el == "V_only":
            d_list_clean.append("V_only")
        else:
            d_list_clean.append(float(el))

    seed_list = [int(s) for s in args.seed_list]
    args.n_realizations = len(seed_list)

    # Sweep d
    for d in d_list_clean:
        print(f"\nRunning d={d} across seeds: {seed_list} ...")

        max_workers = 1 if torch.cuda.is_available() else min(args.n_realizations, os.cpu_count())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(single_seed_run, seed, args, T, X_test, y_test, d, result_dir)
                for seed in seed_list
            ]
            for f in as_completed(futures):
                print(f"  ✅ Seed {f.result()} finished for d={d}")

        # Aggregate across seeds
        all_train, all_test = [], []
        all_overlap_student_teacher, all_overlap_v_teacher = [], []
        all_overlap_u_teacher, all_overlap_uv_teacher = [], []

        for seed in seed_list:
            safe_d = d if isinstance(d, str) else f"{d}"
            data_path = os.path.join(result_dir, f"d_{safe_d}_seed_{seed}.npz")
            if not os.path.exists(data_path):
                continue

            data = np.load(data_path)
            all_train.append(data["train"])
            all_test.append(data["test"])
            all_overlap_student_teacher.append(data["overlap_student_teacher"])
            all_overlap_v_teacher.append(data["overlap_v_teacher"])
            all_overlap_uv_teacher.append(data["overlap_uv_teacher"])
            all_overlap_u_teacher.append(data["overlap_u_teacher"])

        MODE = "pad"
        all_train = align_arrays(all_train, mode=MODE)
        all_test = align_arrays(all_test, mode=MODE)
        all_overlap_v_teacher = align_arrays(all_overlap_v_teacher, mode=MODE)
        all_overlap_u_teacher = align_arrays(all_overlap_u_teacher, mode=MODE)
        all_overlap_uv_teacher = align_arrays(all_overlap_uv_teacher, mode=MODE)
        all_overlap_student_teacher = align_arrays(all_overlap_student_teacher, mode=MODE)

        train_mean, train_std = np.mean(all_train, axis=0), np.std(all_train, axis=0)
        test_mean, test_std = np.mean(all_test, axis=0), np.std(all_test, axis=0)
        overlap_v_teacher_mean, overlap_v_teacher_std = np.mean(all_overlap_v_teacher, axis=0), np.std(all_overlap_v_teacher, axis=0)
        overlap_u_teacher_mean, overlap_u_teacher_std = np.mean(all_overlap_u_teacher, axis=0), np.std(all_overlap_u_teacher, axis=0)
        overlap_uv_teacher_mean, overlap_uv_teacher_std = np.mean(all_overlap_uv_teacher, axis=0), np.std(all_overlap_uv_teacher, axis=0)
        overlap_student_teacher_mean, overlap_student_teacher_std = np.mean(all_overlap_student_teacher, axis=0), np.std(all_overlap_student_teacher, axis=0)

        safe_d = d if isinstance(d, str) else f"{d}"
        np.savez(
            os.path.join(result_dir, f"d_{safe_d}_aggregate.npz"),
            train_mean=train_mean,
            train_std=train_std,
            test_mean=test_mean,
            test_std=test_std,
            overlap_student_teacher_mean=overlap_student_teacher_mean,
            overlap_student_teacher_std=overlap_student_teacher_std,
            overlap_v_teacher_mean=overlap_v_teacher_mean,
            overlap_v_teacher_std=overlap_v_teacher_std,
            overlap_u_teacher_mean=overlap_u_teacher_mean,
            overlap_u_teacher_std=overlap_u_teacher_std,
            overlap_uv_teacher_mean=overlap_uv_teacher_mean,
            overlap_uv_teacher_std=overlap_uv_teacher_std,
            d=d,
            teacher_activation=args.teacher_activation,
            teacher_hermite_order=args.teacher_hermite_order,
            teacher_next_freq=args.teacher_next_freq,
            student_activation=args.student_activation,
            student_hermite_order=args.student_hermite_order,
            student_next_freq=args.student_next_freq,
        )

        print(f"📦 Aggregated results saved for d={d} ({exp_name})")

    print(f"\n✅ All results saved in: {result_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage learning (Stage 1): train on squared teacher labels")

    # Teacher activation φ
    parser.add_argument("--teacher_activation", type=str, default="linear",
                        choices=["linear", "square", "erf", "relu", "sigmoid", "hermite", "hermite+"])
    parser.add_argument("--teacher_hermite_order", type=int, default=None)
    parser.add_argument("--teacher_next_freq", type=int, default=1)

    # Student activation σ
    parser.add_argument("--student_activation", type=str, default="linear",
                        choices=["linear", "square", "erf", "relu", "sigmoid", "hermite", "hermite+"])
    parser.add_argument("--student_hermite_order", type=int, default=None)
    parser.add_argument("--student_next_freq", type=int, default=1)

    # Core dimensions / hyperparams
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--r", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--epochs_save_Step", type=int, default=2, help="Interval for saving plots/weights")

    # Run control
    parser.add_argument("--previous_epochs", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--test_size", type=int, default=200)

    # Sweep + seeds
    parser.add_argument("--d_list", nargs="+", default=["V_only", 0.0, 0.2])
    parser.add_argument("--seed_list", nargs="+", default=[0])

    return parser.parse_args()


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    run_experiment(args)
