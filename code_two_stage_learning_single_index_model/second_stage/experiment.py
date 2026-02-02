import torch
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# ============================================================
# Activations
# ============================================================
def build_activation(name, order=None, next_freq=1):
    if name == "linear":
        return lambda x: x
    if name == "square":
        return lambda x: x**2
    if name == "erf":
        return torch.erf
    if name == "relu":
        return torch.relu

    if name == "hermite":
        if order is None:
            raise ValueError("Hermite activation requires --*_hermite_order")
        return lambda x: hermite_He(x, order)

    if name == "hermite+":
        if order is None:
            raise ValueError("Hermite+ requires --*_hermite_order")
        return lambda x: hermite_He(x, order) + hermite_He(x, order + next_freq)

    raise ValueError(f"Unknown activation {name}")

def hermite_He(x, n):
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x
    if n == 2:
        return x**2 - 1
    if n == 3:
        return x**3 - 3*x
    if n == 4:
        return x**4 - 6*x**2 + 3
    if n == 5:
        return x**5 - 10*x**3 + 15*x
    if n == 6:
        return x**6 - 15*x**4 + 45*x**2 - 15
    raise ValueError("Hermite order not implemented (max 6 for now).")

# ============================================================
# Plotting
# ============================================================
def save_overlap_plot(overlap_history, seed, d, result_dir):
    plt.figure()
    plt.plot(overlap_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Overlap V-Teacher")
    plt.title(f"Overlap vs Epoch (seed={seed}, d={d})")
    plt.grid(True)
    save_path = os.path.join(result_dir, f"plot_overlap_v_teacher_seed_{seed}_d_{d}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📁 Saved overlap plot to: {save_path}")

# ============================================================
# Utility: align arrays (aggregation)
# ============================================================
def align_arrays(list_of_arrays, mode="truncate"):
    """
    Align arrays so they share the same length.
    mode='truncate' -> cut to shortest
    mode='pad'      -> pad to longest using edge values
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
# Single seed run
# ============================================================
def single_seed_run(seed, args, T_cpu, X_test_cpu, y_test_cpu, d, result_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Optional LR rescaling for Hermite student (kept as-is from your spirit)
    if args.student_activation == "hermite":
        lr = lr / (math.factorial(args.student_hermite_order) * args.student_hermite_order)
    elif args.student_activation == "hermite+":
        # note: using student_next_freq (NOT args.next_freq)
        he2 = args.student_hermite_order + args.student_next_freq
        lr = lr / (math.factorial(he2) * he2)

    # Move tensors to device inside subprocess
    T = T_cpu.to(device)
    X_test = X_test_cpu.to(device)
    y_test = y_test_cpu.to(device)

    # -----------------------
    # Student parameterization
    # -----------------------
    if d == "V_only":
        U = None
        V = nn.Parameter(torch.randn(K, N, device=device))
        with torch.no_grad():
            V /= (V.norm() + 1e-12)
        optimizer = optim.SGD([V], lr=lr)

        def forward_weights():
            return V

    else:
        # D is an elementwise mask here (same as your code)
        D = torch.ones(K, N, device=device)
        if K == 1:
            D[:] = float(d)
        else:
            # keep your choice: apply d on first r diagonal entries if possible
            for i in range(min(r, K, N)):
                D[i, i] = float(d)

        U = nn.Parameter(torch.ones(K, r, device=device) / math.sqrt(N**4))
        V = nn.Parameter(torch.randn(r, N, device=device))
        with torch.no_grad():
            V /= (V.norm() + 1e-12)

        optimizer = optim.SGD([U, V], lr=lr)

        def forward_weights():
            # pre-trained component + LoRA correction
            return D * T + U @ V

    # -----------------------
    # Resume previous run
    # -----------------------
    safe_d = d if isinstance(d, str) else f"{d}"
    file_path = os.path.join(result_dir, f"d_{safe_d}_seed_{seed}.npz")

    train_loss_epoch = []
    test_loss_epoch = []
    overlap_v_teacher_epoch = []
    overlap_u_teacher_epoch = []
    overlap_uv_teacher_epoch = []
    overlap_student_teacher_epoch = []

    if args.previous_epochs and os.path.exists(file_path):
        data = np.load(file_path)
        train_loss_epoch = list(data["train"])
        test_loss_epoch = list(data["test"])
        overlap_v_teacher_epoch = list(data["overlap_v_teacher"])
        overlap_u_teacher_epoch = list(data["overlap_u_teacher"])
        overlap_uv_teacher_epoch = list(data["overlap_uv_teacher"])
        overlap_student_teacher_epoch = list(data["overlap_student_teacher"])

        if d != "V_only":
            U_np = data["U"]
            V_np = data["V"]
            U = nn.Parameter(torch.tensor(U_np, device=device))
            V = nn.Parameter(torch.tensor(V_np, device=device))
            optimizer = optim.SGD([U, V], lr=lr)
        else:
            V_np = data["V"]
            V = nn.Parameter(torch.tensor(V_np, device=device))
            U = None
            optimizer = optim.SGD([V], lr=lr)

        print(f"✅ Loaded previous model for seed={seed}, d={d}")
    else:
        print("🟡 No previous checkpoint found — starting from scratch.")

    # ============================================================
    # IMPORTANT: flip only once (per run)
    # ============================================================
    did_flip_U = False

    # -----------------------
    # Training loop
    # -----------------------
    current_epoch = 0
    total_epochs = epochs

    high_overlap_threshold = 0.98  # kept from your second-stage code

    while current_epoch < total_epochs:
        # Sample fresh training batch
        X = torch.randn(B, N, device=device)

        # Teacher labels
        y = teacher_act(X @ T.T)

        # Student prediction
        S = forward_weights()
        y_pred = student_act(X @ S.T)

        # Standard squared loss (second stage: ground-truth labels)
        loss = 0.5 * ((y_pred - y) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # keep V normalized
            V /= (V.norm() + 1e-12)

        train_loss_epoch.append(loss.item())

        # -----------------------
        # Validation / metrics
        # -----------------------
        with torch.no_grad():
            S_test = forward_weights()
            y_pred_test = student_act(X_test @ S_test.T)
            test_loss = (0.5 * (y_pred_test - y_test) ** 2).mean()

            # overlaps
            overlap_v_teacher = torch.abs(T @ V.T).mean().item()

            if d != "V_only":
                A = (T @ V.T)  # (K,r)
                overlap_uv_teacher = (U * A).mean().item()

                # ✅ Flip only ONCE if the signed overlap is negative
                if (not did_flip_U) and (overlap_uv_teacher < 0.0):
                    U.mul_(-1.0)  # in-place, keeps parameter tracked
                    did_flip_U = True
                    # update overlap after flipping (optional but nice)
                    overlap_uv_teacher = (U * A).mean().item()

                overlap_u_teacher = torch.abs(U).mean().item()
            else:
                overlap_uv_teacher = overlap_v_teacher
                overlap_u_teacher = 0.0

            overlap_student_teacher = (T @ S_test.T).mean().item()

        test_loss_epoch.append(test_loss.item())
        overlap_student_teacher_epoch.append(overlap_student_teacher)
        overlap_v_teacher_epoch.append(overlap_v_teacher)
        overlap_u_teacher_epoch.append(overlap_u_teacher)
        overlap_uv_teacher_epoch.append(overlap_uv_teacher)

        # Save plot occasionally
        if (current_epoch % 100 == 0) and (current_epoch > 0):
            save_overlap_plot(overlap_v_teacher_epoch, seed, safe_d, result_dir)

        # Save checkpoint every epoch (same spirit)
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
# Main experiment (parallel over seeds)
# ============================================================
def run_experiment(args):
    print("Parent process - using spawn method for multiprocessing.")
    N, K, test_size = args.N, args.K, args.test_size

    # Teacher direction (CPU)
    torch.manual_seed(0)
    T = torch.randn(K, N)
    T /= (T.norm() + 1e-12)

    # Test set (CPU)
    X_test = torch.randn(test_size, N)

    teacher_act = build_activation(
        args.teacher_activation,
        args.teacher_hermite_order,
        args.teacher_next_freq,
    )
    y_test = teacher_act(X_test @ T.T)

    # Naming folder by teacher+student activation choices
    if args.teacher_activation not in ["hermite", "hermite+"]:
        act_name = f"T_{args.teacher_activation}"
    elif args.teacher_activation == "hermite+":
        act_name = f"T_hermite_He{args.teacher_hermite_order}+He{args.teacher_hermite_order+args.teacher_next_freq}"
    else:
        act_name = f"T_hermite_He{args.teacher_hermite_order}"

    if args.student_activation not in ["hermite", "hermite+"]:
        act_name += f"_S_{args.student_activation}"
    elif args.student_activation == "hermite+":
        act_name += f"_S_hermite_He{args.student_hermite_order}+He{args.student_hermite_order+args.student_next_freq}"
    else:
        act_name += f"_S_hermite_He{args.student_hermite_order}"

    result_dir = os.path.join(
        f"results_lr{args.learning_rate}_batch_size{args.batch_size}_input_dimension{args.N}",
        act_name,
    )
    os.makedirs(result_dir, exist_ok=True)

    # d_list cleaning
    d_list_clean = []
    for el in args.d_list:
        if isinstance(el, str) and el == "V_only":
            d_list_clean.append("V_only")
        else:
            d_list_clean.append(float(el))

    seed_list = [int(s) for s in args.seed_list]

    for d in d_list_clean:
        print(f"\nRunning d={d} across seeds: {seed_list} ...")

        # If you use GPU, keep 1 worker (same spirit)
        max_workers = 1 if torch.cuda.is_available() else min(len(seed_list), os.cpu_count())

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
            if os.path.exists(data_path):
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
            student_activation=args.student_activation,
            student_hermite_order=args.student_hermite_order,
        )

        print(f"📦 Aggregated results saved for d={d} ({act_name})")

    print(f"\n✅ All results saved in: {result_dir}")

# ============================================================
# Args
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Teacher-Student LoRA experiment")

    # Teacher activation
    parser.add_argument(
        "--teacher_activation",
        type=str,
        default="linear",
        choices=["linear", "square", "erf", "relu", "hermite", "hermite+"],
    )
    parser.add_argument("--teacher_hermite_order", type=int, default=None)
    parser.add_argument("--teacher_next_freq", type=int, default=1)

    # Student activation
    parser.add_argument(
        "--student_activation",
        type=str,
        default="linear",
        choices=["linear", "square", "erf", "relu", "hermite", "hermite+"],
    )
    parser.add_argument("--student_hermite_order", type=int, default=None)
    parser.add_argument("--student_next_freq", type=int, default=1)

    # Core experiment params
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--previous_epochs", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--d_list", nargs="+", default=["V_only", 0.0, 0.2])
    parser.add_argument("--seed_list", nargs="+", default=[0])
    parser.add_argument(
        "--epochs_save_Step",
        type=int,
        default=2,
        help="Step interval for saving plots/weights",
    )

    return parser.parse_args()

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    run_experiment(args)