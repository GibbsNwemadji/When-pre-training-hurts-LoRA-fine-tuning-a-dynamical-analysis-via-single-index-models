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
# Single-index teacher–student with μ-aligned preactivation
# ============================================================
#
# This script implements a SINGLE-INDEX model (K = 1 by default).
#
# Teacher generates labels using a direction ω★ (called T in code):
#     y = σ( x · ω★ )
#
# Student is initialized with a preactivation direction that mixes:
#   - the teacher direction ω★
#   - an independent random direction ξ
#
# Pre-activation direction (correlated initialization):
#     w0(μ) = μ ω★ + (1-μ) ξ
#
# In this code:
#   - T is ω★   (teacher direction)
#   - Xi is ξ   (random direction)
#   - d plays the role of μ (alignment strength)
#
# Student weights S are:
#   - either trained directly (V_only): S = V
#   - or LoRA-style:  S = w0(μ) + U V
#
# where U V is a rank-r adaptation term.
# ============================================================


torch.set_num_threads(1)


# ------------------------------------------------------------
# Activations
# ------------------------------------------------------------
def hermite_He(x: torch.Tensor, n: int) -> torch.Tensor:
    """Probabilists' Hermite polynomials He_n(x), implemented up to n=6."""
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


def get_activation(name: str, args):
    """Return a callable activation σ."""
    if name == "linear":
        return lambda x: x
    if name == "square":
        return lambda x: x**2
    if name == "erf":
        return lambda x: torch.erf(x)
    if name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    if name == "relu":
        return torch.relu

    if name == "hermite":
        if args.hermite_order is None:
            raise ValueError("Hermite activation selected but no order provided for hermite.")
        return lambda x: hermite_He(x, args.hermite_order)

    if name == "hermite+":
        if args.hermite_order is None:
            raise ValueError("Hermite activation selected but no order provided for hermite+.")
        n0 = args.hermite_order
        n1 = args.hermite_order + args.next_freq
        return lambda x: hermite_He(x, n0) + hermite_He(x, n1)

    raise ValueError(f"Unknown activation: {name}")


# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------
def save_overlap_plot(overlap_history, seed, mu, result_dir):
    """Save the overlap(V, teacher) trajectory."""
    plt.figure()
    plt.plot(overlap_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.xscale("log")
    plt.ylabel("Overlap ⟨V, ω★⟩")
    plt.title(f"Overlap vs Epoch (seed={seed}, μ={mu})")
    plt.grid(True)

    save_path = os.path.join(result_dir, f"plot_overlap_v_teacher_seed_{seed}_mu_{mu}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📁 Saved overlap plot to: {save_path}")


# ------------------------------------------------------------
# Utility: align variable-length trajectories across seeds
# ------------------------------------------------------------
def align_arrays(list_of_arrays, mode="truncate"):
    """
    Align a list of 1D arrays to a common length.
      - truncate: cut all to the shortest length
      - pad:      extend shorter arrays by repeating their last value
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
# Single seed run
# ------------------------------------------------------------
def single_seed_run(seed, args, T_cpu, Xi_cpu, X_test_cpu, y_test_cpu, mu, result_dir):
    """
    One training run for a fixed random seed and a fixed μ.

    Teacher:
      y = σ(X @ T^T)

    Student:
      - "V_only": S = V (train V directly)
      - otherwise:
          w0(μ) = μ T + (1-μ) Xi
          S     = w0(μ) + U @ V
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activation = get_activation(args.activation, args)
    N, K, r = args.N, args.K, args.r
    B, lr, epochs = args.batch_size, args.learning_rate, args.epochs

    # Optional LR scaling for Hermite activations
    if args.activation == "hermite":
        lr = lr / (math.factorial(args.hermite_order) * args.hermite_order)
    elif args.activation == "hermite+":
        n = args.hermite_order + args.next_freq
        lr = lr / (math.factorial(n) * n)

    # Move fixed tensors to device INSIDE subprocess
    T = T_cpu.to(device)          # teacher direction ω★
    Xi = Xi_cpu.to(device)        # random direction ξ
    X_test = X_test_cpu.to(device)
    y_test = y_test_cpu.to(device)

    # =========================================================
    # Student parameters
    # =========================================================
    if mu == "V_only":
        # Train student direction directly: S = V
        V = nn.Parameter(torch.randn(K, N, device=device))
        with torch.no_grad():
            V /= V.norm()
        U = None
        optimizer = optim.SGD([V], lr=lr)

        def forward_weights():
            return V
    else:
        # μ-aligned initialization (single-index core idea)
        # w0(μ) = μ T + (1-μ) Xi
        w0 = float(mu) * T + (1.0 - float(mu)) * Xi

        # LoRA adaptation (rank r): S = w0 + U V
        U = nn.Parameter(torch.ones(K, r, device=device) / math.sqrt(N**4))
        V = nn.Parameter(torch.randn(r, N, device=device))
        with torch.no_grad():
            V /= (V.norm(dim=1, keepdim=True) + 1e-12)

        optimizer = optim.SGD([U, V], lr=lr)

        def forward_weights():
            return w0 + U @ V

    # =========================================================
    # Resume if checkpoint exists
    # =========================================================
    safe_mu = mu if isinstance(mu, str) else f"{mu}"
    file_path = os.path.join(result_dir, f"mu_{safe_mu}_seed_{seed}.npz")

    if args.previous_epochs and os.path.exists(file_path):
        data = np.load(file_path)
        train_loss_epoch = list(data["train"])
        test_loss_epoch = list(data["test"])
        overlap_v_teacher_epoch = list(data["overlap_v_teacher"])
        overlap_u_teacher_epoch = list(data["overlap_u_teacher"])
        overlap_uv_teacher_epoch = list(data["overlap_uv_teacher"])
        overlap_student_teacher_epoch = list(data["overlap_student_teacher"])

        if mu != "V_only":
            U = nn.Parameter(torch.tensor(data["U"], device=device))
            V = nn.Parameter(torch.tensor(data["V"], device=device))
            optimizer = optim.SGD([U, V], lr=lr)
        else:
            V = nn.Parameter(torch.tensor(data["V"], device=device))
            U = None
            optimizer = optim.SGD([V], lr=lr)

        print(f"✅ Loaded previous model for seed={seed}, μ={mu}")
    else:
        print("ℹ️ No previous checkpoint found — starting fresh.")
        train_loss_epoch, test_loss_epoch = [], []
        overlap_v_teacher_epoch, overlap_u_teacher_epoch = [], []
        overlap_uv_teacher_epoch, overlap_student_teacher_epoch = [], []

    # =========================================================
    # Training loop
    # =========================================================
    high_overlap_threshold = 0.98
    max_epochs = getattr(args, "max_epochs", 8000)

    current_epoch = 0
    total_epochs = epochs

    while current_epoch < total_epochs:
        if current_epoch >= max_epochs:
            print(f"🛑 Reached max_epochs={max_epochs}, stopping.")
            break

        # Synthetic batch
        X = torch.randn(B, N, device=device)

        # Teacher targets and student predictions
        y = activation(X @ T.T)             # (B, K) in general
        S = forward_weights()
        y_pred = activation(X @ S.T)

        loss = 0.5 * ((y_pred - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Normalize V (keep direction controlled)
        with torch.no_grad():
            if mu == "V_only":
                V /= V.norm()
            else:
                V /= (V.norm(dim=1, keepdim=True) + 1e-12)

        train_loss_epoch.append(loss.item())

        # ---- Validation + overlaps ----
        with torch.no_grad():
            S_test = forward_weights()
            y_pred_test = activation(X_test @ S_test.T)
            test_loss = (0.5 * (y_pred_test - y_test) ** 2).mean()

            # Overlap between V directions and teacher
            overlap_v_teacher = torch.abs(T @ V.T).mean().item()

            if mu != "V_only":
                overlap_uv_teacher = (U * (T @ V.T)).mean().item()
                overlap_u_teacher = torch.abs(U).mean().item()
            else:
                overlap_uv_teacher = overlap_v_teacher
                overlap_u_teacher = 0.0

            # In single-index K=1 this is just ⟨T, S⟩
            overlap_student_teacher = (T @ S_test.T).mean().item()

        test_loss_epoch.append(test_loss.item())
        overlap_v_teacher_epoch.append(overlap_v_teacher)
        overlap_u_teacher_epoch.append(overlap_u_teacher)
        overlap_uv_teacher_epoch.append(overlap_uv_teacher)
        overlap_student_teacher_epoch.append(overlap_student_teacher)

        # ---- Save plot occasionally ----
        if (current_epoch % 100 == 0) and (current_epoch > 0):
            save_overlap_plot(overlap_v_teacher_epoch, seed, safe_mu, result_dir)

        # ---- Extend training if still not escaped ----
        if current_epoch == epochs - 1 and overlap_v_teacher <= high_overlap_threshold:
            print(f"⚠️ Extending training up to {max_epochs} (overlap_v_teacher={overlap_v_teacher:.3f})")
            total_epochs = min(max_epochs, epochs + (max_epochs - epochs))

        # ---- Save checkpoint every epoch ----
        os.makedirs(result_dir, exist_ok=True)
        np.savez(
            file_path,
            train=np.array(train_loss_epoch, dtype=float),
            test=np.array(test_loss_epoch, dtype=float),
            overlap_v_teacher=np.array(overlap_v_teacher_epoch, dtype=float),
            overlap_u_teacher=np.array(overlap_u_teacher_epoch, dtype=float),
            overlap_uv_teacher=np.array(overlap_uv_teacher_epoch, dtype=float),
            overlap_student_teacher=np.array(overlap_student_teacher_epoch, dtype=float),
            mu=mu,
            U=U.detach().cpu().numpy() if U is not None else np.array([]),
            V=V.detach().cpu().numpy(),
            seed=seed,
        )

        current_epoch += 1

    return seed


# ------------------------------------------------------------
# Main experiment (parallel over seeds, sweep over μ)
# ------------------------------------------------------------
def run_experiment(args):
    print("Parent process - using spawn method for multiprocessing.")

    N, K, test_size = args.N, args.K, args.test_size
    torch.manual_seed(0)

    # Teacher direction ω★ (normalized)
    T = torch.randn(K, N)
    T /= T.norm()

    # Random direction ξ (normalized)
    Xi = torch.randn(K, N)
    Xi /= Xi.norm()

    # Test set
    X_test = torch.randn(test_size, N)
    activation_parent = get_activation(args.activation, args)
    y_test = activation_parent(X_test @ T.T)

    # Result folder naming
    if args.activation not in ("hermite", "hermite+"):
        act_name = f"{args.activation}"
    elif args.activation == "hermite+":
        act_name = f"hermite_He{args.hermite_order}+He{args.hermite_order+args.next_freq}"
    else:
        act_name = f"hermite_He{args.hermite_order}"

    result_dir = os.path.join(
        f"results_lr{args.learning_rate}_batch_size{args.batch_size}_input_dimension{args.N}",
        act_name,
    )
    os.makedirs(result_dir, exist_ok=True)

    # μ sweep list (keeps "V_only" as string)
    mu_list = []
    for el in args.d_list:
        if isinstance(el, str) and el == "V_only":
            mu_list.append("V_only")
        else:
            mu_list.append(float(el))

    seed_list = [int(s) for s in args.seed_list]
    args.n_realizations = len(seed_list)

    for mu in mu_list:
        print(f"\nRunning μ={mu} across seeds: {seed_list} ...")

        # If CUDA is available, keep a single worker to avoid GPU contention
        max_workers = 1 if torch.cuda.is_available() else min(args.n_realizations, os.cpu_count())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(single_seed_run, seed, args, T, Xi, X_test, y_test, mu, result_dir)
                for seed in seed_list
            ]
            for f in as_completed(futures):
                print(f"  ✅ Seed {f.result()} finished for μ={mu}")

        # Aggregate across seeds
        all_train, all_test = [], []
        all_overlap_student_teacher, all_overlap_v_teacher = [], []
        all_overlap_u_teacher, all_overlap_uv_teacher = [], []

        for seed in seed_list:
            safe_mu = mu if isinstance(mu, str) else f"{mu}"
            data_path = os.path.join(result_dir, f"mu_{safe_mu}_seed_{seed}.npz")
            if not os.path.exists(data_path):
                continue

            data = np.load(data_path)
            all_train.append(data["train"])
            all_test.append(data["test"])
            all_overlap_student_teacher.append(data["overlap_student_teacher"])
            all_overlap_v_teacher.append(data["overlap_v_teacher"])
            all_overlap_u_teacher.append(data["overlap_u_teacher"])
            all_overlap_uv_teacher.append(data["overlap_uv_teacher"])

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

        safe_mu = mu if isinstance(mu, str) else f"{mu}"
        np.savez(
            os.path.join(result_dir, f"mu_{safe_mu}_aggregate.npz"),
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
            mu=mu,
            activation=args.activation,
            hermite_order=args.hermite_order,
        )

        print(f"📦 Aggregated results saved for μ={mu} ({act_name})")

    print(f"\n✅ All results saved in: {result_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Single-index teacher–student with μ-aligned initialization (parallelized)")
    parser.add_argument(
        "--activation",
        type=str,
        default="linear",
        choices=["linear", "square", "erf", "relu", "sigmoid", "hermite", "hermite+"],
    )
    parser.add_argument("--hermite_order", type=int, default=None)
    parser.add_argument("--next_freq", type=int, default=1)

    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--K", type=int, default=1)  # single-index when K=1
    parser.add_argument("--r", type=int, default=1)

    parser.add_argument("--previous_epochs", type=lambda x: x.lower() == "true", default=True)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--epochs_save_Step", type=int, default=2)

    parser.add_argument("--n_realizations", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=1000)

    # We keep the original arg name d_list for compatibility, but it represents μ here
    parser.add_argument("--d_list", nargs="+", default=["V_only", 0.0, 0.1, 0.5, 0.9])
    parser.add_argument("--seed_list", nargs="+", default=[0, 1])

    return parser.parse_args()


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    run_experiment(args)
