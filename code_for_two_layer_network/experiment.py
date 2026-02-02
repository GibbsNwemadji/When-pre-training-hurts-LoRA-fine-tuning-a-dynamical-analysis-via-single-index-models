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


# ------------------------------------------------------------
# Global settings
# ------------------------------------------------------------
torch.set_num_threads(1)


# ============================================================
# Activations
# ============================================================
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
    """
    Return a callable activation function.
    Supported: linear, square, erf, sigmoid, relu, hermite, hermite+.
    """
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


# ============================================================
# Two-layer readout and scaling
# ============================================================
def activation_scaling(activation_name: str) -> str:
    """
    Returns which normalization to use in the 2-layer readout:
      - "sqrtK": mean-zero activations -> 1/sqrt(K)
      - "K":     non-zero-mean activations -> 1/K
    """
    mean_zero_activations = {
        "linear",
        "erf",
        "hermite",    # assuming odd Hermite order in usage
        "hermite+",   # assuming combination preserves odd symmetry
    }
    return "sqrtK" if activation_name in mean_zero_activations else "K"


def two_layer_readout(X: torch.Tensor, W: torch.Tensor, activation, K: int, activation_name: str) -> torch.Tensor:
    """
    Two-layer scalar readout:
      pre = X W^T  in R^{B x K}
      y   = (1/scale) sum_j sigma(pre_j)

    X: (B, N)
    W: (K, N)
    returns: (B, 1)
    """
    pre = X @ W.T          # (B, K)
    h = activation(pre)    # (B, K)

    if activation_scaling(activation_name) == "K":
        return h.sum(dim=1, keepdim=True) / K
    return h.sum(dim=1, keepdim=True) / math.sqrt(K)


# ============================================================
# Plotting utilities
# ============================================================
def save_overlap_plot(overlap_history, seed, d, result_dir):
    """Save overlap(V, teacher) history as a PNG."""
    plt.figure()
    plt.plot(overlap_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.xscale("log")
    plt.ylabel("Overlap V-Teacher")
    plt.title(f"Overlap vs Epoch (seed={seed}, d={d})")
    plt.grid(True)

    save_path = os.path.join(result_dir, f"plot_overlap_v_teacher_seed_{seed}_d_{d}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📁 Saved overlap plot to: {save_path}")


def save_metric_plot(
    metric_history,
    seed,
    d,
    result_dir,
    metric_name="Test loss",
    filename_prefix="test_loss",
    yscale="log",
):
    """Generic metric plot saver."""
    plt.figure()
    plt.plot(metric_history, linewidth=2)
    plt.xlabel("Epoch")
    if yscale is not None:
        plt.yscale(yscale)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Epoch (seed={seed}, d={d})")
    plt.grid(True)

    save_path = os.path.join(result_dir, f"plot_{filename_prefix}_seed_{seed}_d_{d}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📁 Saved {metric_name} plot to: {save_path}")


# ============================================================
# Aggregation utilities
# ============================================================
def align_arrays(list_of_arrays, mode="truncate"):
    """
    Align arrays to the same length:
      - mode="truncate": cut to shortest
      - mode="pad":      pad shorter arrays with last value to match longest
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


# ============================================================
# Single-seed training run
# ============================================================
def single_seed_run(seed, args, T_cpu, Xi_cpu, X_test_cpu, y_test_cpu, d, result_dir):
    """
    Run one training trajectory for a fixed seed and a fixed d.

    Teacher weights: T in R^{KxN}
    Student weights:
      - if d == "V_only": S = V (train V directly in R^{KxN})
      - else:            S = D T + U V (LoRA rank-r adaptation)
    """
    # ----- RNG + device -----
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- hyperparameters -----
    activation = get_activation(args.activation, args)
    N, K, r = args.N, args.K, args.r
    B, lr, epochs = args.batch_size, args.learning_rate, args.epochs

    # Optional LR rescaling for Hermite activations
    if args.activation == "hermite":
        lr = lr / (math.factorial(args.hermite_order) * args.hermite_order)
    elif args.activation == "hermite+":
        n = args.hermite_order + args.next_freq
        lr = lr / (math.factorial(n) * n)

    # ----- move fixed tensors to device inside subprocess -----
    T = T_cpu.to(device)
    Xi = Xi_cpu.to(device)
    X_test = X_test_cpu.to(device)
    y_test = y_test_cpu.to(device)

    # =========================================================
    # Student parametrization
    # =========================================================
    if d == "V_only":
        # Train V directly: S = V
        V = nn.Parameter(torch.randn(K, N, device=device))
        with torch.no_grad():
            V /= V.norm()
        U = None
        optimizer = optim.SGD([V], lr=lr)

        def forward_weights():
            return V
    else:
        # LoRA: S = D T + U V
        D = torch.eye(K, device=device)
        if K == 1:
            D[:] = float(d)
        else:
            for i in range(2):
                D[i, i] = float(d)

        U = nn.Parameter(torch.ones(K, r, device=device) / math.sqrt(N**4))
        V = nn.Parameter(torch.randn(r, N, device=device))
        with torch.no_grad():
            V /= (V.norm(dim=1, keepdim=True) + 1e-12)

        optimizer = optim.SGD([U, V], lr=lr)

        def forward_weights():
            return D @ T + U @ V

    # =========================================================
    # Resume from disk (if enabled)
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
            optimizer = optim.SGD([U, V], lr=lr)  # reinit optimizer
        else:
            V = nn.Parameter(torch.tensor(data["V"], device=device))
            U = None
            optimizer = optim.SGD([V], lr=lr)

        print(f"✅ Loaded previous model for seed={seed}, d={d}")
    else:
        print("ℹ️ No previous run found — starting from scratch.")
        train_loss_epoch = []
        test_loss_epoch = []
        overlap_v_teacher_epoch = []
        overlap_u_teacher_epoch = []
        overlap_uv_teacher_epoch = []
        overlap_student_teacher_epoch = []

    # =========================================================
    # Training loop
    # =========================================================
    high_overlap_threshold = 0.98
    max_epochs = getattr(args, "max_epochs", 15000)

    current_epoch = 0
    total_epochs = epochs

    while current_epoch < total_epochs:
        if current_epoch >= max_epochs:
            print(f"🛑 Reached max_epochs={max_epochs}, stopping.")
            break

        # ----- sample fresh minibatch -----
        X = torch.randn(B, N, device=device)

        # Teacher labels
        y = two_layer_readout(X, T, activation, K, args.activation)

        # Student prediction
        S = forward_weights()
        y_pred = two_layer_readout(X, S, activation, K, args.activation)

        # MSE /2
        loss = 0.5 * ((y_pred - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # keep V normalized (vector or row-wise)
        with torch.no_grad():
            V /= V.norm() if d == "V_only" else (V.norm(dim=1, keepdim=True) + 1e-12)

        train_loss_epoch.append(loss.item())

        # ----- validation + overlaps -----
        with torch.no_grad():
            S_test = forward_weights()
            y_pred_test = two_layer_readout(X_test, S_test, activation, K, args.activation)
            test_loss = (0.5 * (y_pred_test - y_test) ** 2).mean()

            # Overlaps
            A = T @ V.T                         # (K,r) or (K,K) if V_only with r=K
            overlap_v_teacher = torch.abs(A).mean().item()

            if d != "V_only":
                overlap_uv_teacher = (U * A).mean().item()
                overlap_u_teacher = torch.abs(U).mean().item()
            else:
                overlap_uv_teacher = overlap_v_teacher
                overlap_u_teacher = 0.0

            M = T @ S_test.T                    # (K,K)
            overlap_student_teacher = M.diag()[:r].mean().item()

        test_loss_epoch.append(test_loss.item())
        overlap_v_teacher_epoch.append(overlap_v_teacher)
        overlap_u_teacher_epoch.append(overlap_u_teacher)
        overlap_uv_teacher_epoch.append(overlap_uv_teacher)
        overlap_student_teacher_epoch.append(overlap_student_teacher)

        # ----- occasional plots -----
        if (current_epoch % 100 == 0) and (current_epoch > 0):
            save_overlap_plot(overlap_v_teacher_epoch, seed, safe_d, result_dir)
            save_metric_plot(
                test_loss_epoch,
                seed,
                safe_d,
                result_dir,
                metric_name="Test loss",
                filename_prefix="test_loss",
                yscale="log",
            )

        # ----- extend training if still not escaped by the end -----
        if current_epoch == epochs - 1 and overlap_v_teacher <= high_overlap_threshold:
            print(f"⚠️ Extending training up to {max_epochs} (overlap_v_teacher={overlap_v_teacher:.3f})")
            total_epochs = min(max_epochs, epochs + (max_epochs - epochs))

        # ----- save checkpoint each epoch -----
        os.makedirs(result_dir, exist_ok=True)
        final_path = os.path.join(result_dir, f"d_{safe_d}_seed_{seed}.npz")

        np.savez(
            final_path,
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


# ============================================================
# Main experiment driver (parallel over seeds, loop over d)
# ============================================================
def run_experiment(args):
    print("Parent process - using spawn method for multiprocessing.")

    # ------------------------------------------------------------
    # Create teacher directions on CPU (avoid CUDA init in parent)
    # ------------------------------------------------------------
    N, K, test_size = args.N, args.K, args.test_size
    torch.manual_seed(0)

    T = torch.randn(K, N)
    Q, _ = torch.linalg.qr(T.T)
    T = Q[:, :K].T.contiguous()

    Xi = torch.randn(K, N)
    Qx, _ = torch.linalg.qr(Xi.T)
    Xi = Qx[:, :K].T.contiguous()

    # Test set on CPU
    activation_parent = get_activation(args.activation, args)
    X_test = torch.randn(test_size, N)
    y_test = two_layer_readout(X_test, T, activation_parent, K, args.activation)

    # ------------------------------------------------------------
    # Result directory naming
    # ------------------------------------------------------------
    if args.activation not in ("hermite", "hermite+"):
        act_name = f"{args.activation}"
    elif args.activation == "hermite+":
        act_name = f"hermite_He{args.hermite_order}+He{args.hermite_order+args.next_freq}"
    else:
        act_name = f"hermite_He{args.hermite_order}"

    result_dir = os.path.join(args.base_dir, act_name)
    os.makedirs(result_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Clean d_list and seeds
    # ------------------------------------------------------------
    d_list_clean = []
    for el in args.d_list:
        if isinstance(el, str) and el == "V_only":
            d_list_clean.append("V_only")
        else:
            d_list_clean.append(float(el))

    seed_list = [int(s) for s in args.seed_list]
    args.n_realizations = len(seed_list)

    # ------------------------------------------------------------
    # Loop over d, parallelize over seeds
    # ------------------------------------------------------------
    for d in d_list_clean:
        print(f"\nRunning d={d} across seeds: {seed_list} ...")

        # If CUDA is available, run 1 worker (avoid multi-process CUDA contention)
        max_workers = 1 if torch.cuda.is_available() else min(args.n_realizations, os.cpu_count())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(single_seed_run, seed, args, T, Xi, X_test, y_test, d, result_dir)
                for seed in seed_list
            ]
            for f in as_completed(futures):
                print(f"  ✅ Seed {f.result()} finished for d={d}")

        # --------------------------------------------------------
        # Aggregate results across seeds
        # --------------------------------------------------------
        all_train = []
        all_test = []
        all_overlap_student_teacher = []
        all_overlap_v_teacher = []
        all_overlap_u_teacher = []
        all_overlap_uv_teacher = []

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

        # Align and compute mean/std
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

        # Save aggregated results
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
            activation=args.activation,
            hermite_order=args.hermite_order,
        )

        print(f"📦 Aggregated results saved for d={d} ({act_name})")

    print(f"\n✅ All results saved in: {result_dir}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Teacher-Student LoRA experiment (parallelized)")
    parser.add_argument(
        "--activation",
        type=str,
        default="linear",
        choices=["linear", "square", "erf", "relu", "sigmoid", "hermite", "hermite+"],
    )
    parser.add_argument("--hermite_order", type=int, default=None)
    parser.add_argument("--next_freq", type=int, default=1)

    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--r", type=int, default=1)

    parser.add_argument("--base_dir", type=str, default="results", help="Base results directory")
    parser.add_argument("--previous_epochs", type=lambda x: x.lower() == "true", default=True)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--epochs_save_Step", type=int, default=2)

    parser.add_argument("--n_realizations", type=int, default=10)
    parser.add_argument("--test_size", type=int, default=1000)

    parser.add_argument("--d_list", nargs="+", default=["V_only", 0.0, 0.1, 0.5, 0.9])
    parser.add_argument("--seed_list", nargs="+", default=[0, 1])

    return parser.parse_args()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    run_experiment(args)