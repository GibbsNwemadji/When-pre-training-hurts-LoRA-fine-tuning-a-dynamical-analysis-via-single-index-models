import os
import glob
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.style.use("seaborn-v0_8-whitegrid")


# -----------------------
# Utility
# -----------------------

def load_all_seeds(result_dir, d):
    files = sorted(glob.glob(os.path.join(result_dir, f"d_{d}_seed_*.npz")))
    all_train, all_test = [], []
    for f in files:
        data = np.load(f)
        if "train" in data and "test" in data:
            all_train.append(data["train"])
            all_test.append(data["test"])
    return np.array(all_train), np.array(all_test)


"""def load_aggregate(result_dir, d):
    #safe_d = d if isinstance(d, str) else f"{d}"
    safe_d = d if isinstance(d, str) else f"{float(d):.0e}"
    path = os.path.join(result_dir, f"d_{safe_d}_aggregate.npz")
    print(f"the path for {d} is {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Aggregate not found for d={d} in {result_dir}")
    #print(f"the path for {d} is {path}")
    return np.load(path)"""


"""def load_aggregate(result_dir, d):
    # Try consistent scientific notation (e.g., 1e-05)
    try:
        #safe_d = f"{float(d):.0e}"
        safe_d = d
    except (ValueError, TypeError):
        safe_d = str(d)
    path = os.path.join(result_dir, f"d_{safe_d}_aggregate.npz")
    print(f"the path for {d} is {path}")

    # If file not found, try alternate format
    if not os.path.exists(path):
        print(f"⚠️ Not found for path {path}, trying alternate path")
        alt_d = str(d)
        alt_path = os.path.join(result_dir, f"d_{alt_d}_aggregate.npz")
        path = alt_path
        
        if not os.path.exists(alt_path):
            print(f"⚠️ Not found for path {path}, trying alternate path")
            alt_d = float(d)
            alt_path = os.path.join(result_dir, f"d_{alt_d}_aggregate.npz")
            path = alt_path
            
            if not os.path.exists(alt_path):
                raise FileNotFoundError(f"Aggregate not found for d={d} in {result_dir}")
        else:
            path = alt_path  # use the alternate file if found

    return np.load(path)"""


def load_aggregate(result_dir, d):
    d_variants = [d, str(d)]
    try:
        d_variants.append(f"{float(d):.0e}")
    except:
        pass

    for dv in d_variants:
        path = os.path.join(result_dir, f"d_{dv}_aggregate.npz")
        if os.path.exists(path):
            print(f"✅ Found: {path}")
            return np.load(path)
    raise FileNotFoundError(f"No aggregate file found for d={d} in {result_dir}")






# -----------------------
# Plot 1 — Dynamics for Activation
# -----------------------

def plot_dynamics_for_activation(result_dir, activation_name, d_list, save_pdf=True):
    fig, axes = plt.subplots(3, 2, figsize=(15, 8))
    #ax_train, ax_test = axes
    # Unpack:
    ax_train               = axes[0, 0]
    ax_test                = axes[0, 1]
    ax_overlap_v_teacher   = axes[1, 0]
    ax_overlap_student_teacher     = axes[1, 1]
    ax_overlap_u_teacher   = axes[2, 0]
    ax_overlap_uv_teacher   = axes[2, 1]
    ax_train.set_title(f"Train Loss ({activation_name})", fontsize=13, weight="bold")
    ax_test.set_title(f"Test Loss ({activation_name})", fontsize=13, weight="bold")
    ax_overlap_v_teacher.set_ylabel(fr"$m=v\cdot \omega$")
    ax_overlap_u_teacher.set_ylabel("u")
    ax_overlap_uv_teacher.set_ylabel(fr"$um$")
    ax_overlap_student_teacher.set_ylabel(fr"$um + \mu$")
    ax_train.set_yscale("log")
    ax_test.set_yscale("log")
    ax_train.set_ylabel("train error")
    ax_test.set_ylabel("test error")
    for ax, name in zip(axes.flatten(), ["Train Error", "Test Error", "Overlap v Teacher $m=v\cdot \omega$", fr"Overlap Student–Teacher: $\mu+um$", " u ", "um"]):
        ax.set_title(f"{name} ({activation_name})", fontsize=13, weight="bold")
        #ax.set_xscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    colors = plt.cm.viridis(np.linspace(0, 1, len(d_list)))
    #colors = plt.cm.tab20(np.linspace(0, 1, len(activations)))


    for color, d in zip(colors, d_list):
        try:
            data = load_aggregate(result_dir, d)
        except FileNotFoundError:
            print(f"⚠️ Aggregate missing for μ={d}")
            continue

        endf = -1
        train_mean, train_std = data["train_mean"][:endf], data["train_std"][:endf]
        test_mean, test_std = data["test_mean"][:endf], data["test_std"][:endf]
        overlap_v_teacher_mean, overlap_v_teacher_std = data["overlap_v_teacher_mean"][:endf], data["overlap_v_teacher_std"][:endf]
        overlap_u_teacher_mean, overlap_u_teacher_std = data["overlap_u_teacher_mean"][:endf], data["overlap_u_teacher_std"][:endf]
        overlap_uv_teacher_mean, overlap_uv_teacher_std = data["overlap_uv_teacher_mean"][:endf], data["overlap_uv_teacher_std"][:endf]
        overlap_student_teacher_mean, overlap_student_teacher_std = data["overlap_student_teacher_mean"][:endf], data["overlap_student_teacher_std"][:endf]
        
        
        #print(f"unique seed train_mean is {train_mean}")

        ax_train.plot(train_mean,'--', color=color, label=f"μ={d}", linewidth=2)
        ax_train.fill_between(range(len(train_mean)), train_mean - train_std, train_mean + train_std,
                              color=color, alpha=0.25)
        ax_test.plot(test_mean,'--', color=color, label=f"μ={d}", linewidth=2)
        ax_test.fill_between(range(len(test_mean)), test_mean - test_std, test_mean + test_std,
                             color=color, alpha=0.25)
        ax_overlap_student_teacher.plot(overlap_student_teacher_mean,'-', color=color, label=f"μ={d}", linewidth=2)
        ax_overlap_student_teacher.fill_between(range(len(overlap_student_teacher_mean)),overlap_student_teacher_mean - overlap_student_teacher_std, overlap_student_teacher_mean + overlap_student_teacher_std,
                             color=color, alpha=0.25)
        ax_overlap_v_teacher.plot(overlap_v_teacher_mean,'-', color=color, label=fr"$m=v\cdot \omega_\star$, μ={d}", linewidth=2)
        print(f"the last five element {overlap_v_teacher_mean[-5:]} for μ={d}")
        ax_overlap_v_teacher.fill_between(range(len(overlap_v_teacher_mean)), overlap_v_teacher_mean - overlap_v_teacher_std, overlap_v_teacher_mean + overlap_v_teacher_std,
                             color=color, alpha=0.25)
        ax_overlap_u_teacher.plot(overlap_u_teacher_mean,'-', color=color, label=f"u", linewidth=2)
        ax_overlap_u_teacher.fill_between(range(len(overlap_u_teacher_mean)), overlap_u_teacher_mean - overlap_u_teacher_std, overlap_u_teacher_mean + overlap_u_teacher_std,
                             color=color, alpha=0.25)
        ax_overlap_uv_teacher.plot(overlap_uv_teacher_mean,'-', color=color, label=fr"$u m= u v\cdot \omega_\star$, μ={d}", linewidth=2)
        ax_overlap_uv_teacher.fill_between(range(len(overlap_uv_teacher_mean)), overlap_uv_teacher_mean - overlap_uv_teacher_std, overlap_uv_teacher_mean + overlap_uv_teacher_std,
                             color=color, alpha=0.25)

        # Overlay seed trajectories for small d
        """try:
            d_val = float(d)
            if d_val < 0.3:
                all_train, all_test = load_all_seeds(result_dir, d)
                for traj in all_train:
                    ax_train.plot(traj,'--', color=color, alpha=0.015, linewidth=0.6)
                for traj in all_test:
                    ax_test.plot(traj,'--', color=color, alpha=0.015, linewidth=0.6)
        except:
            all_train, all_test = load_all_seeds(result_dir, d)
            for traj in all_train:
                ax_train.plot(traj,'--', color=color, alpha=0.015, linewidth=0.6)
            for traj in all_test:
                ax_test.plot(traj,'--', color=color, alpha=0.015, linewidth=0.6)"""

    ax_train.legend(title="μ values", fontsize=9, loc='center right')
    plt.tight_layout()

    if save_pdf:
        out_path = os.path.join(result_dir, f"{activation_name}_train_test_dynamics.eps")
        os.makedirs(result_dir, exist_ok=True)  # ✅ ensure directory exists
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"📄 Saved: {out_path}")
    plt.show()


# -----------------------
# Plot 2 — Compare Activations for Fixed d
# -----------------------

def compare_activations_for_d(base_dir, activations, d_fixed, save_pdf=True):
    fig, axes = plt.subplots(3, 2, figsize=(15, 8))
    #ax_train, ax_test = axes
    # Unpack:
    ax_train               = axes[0, 0]
    ax_test                = axes[0, 1]
    ax_overlap_v_teacher   = axes[1, 0]
    ax_overlap_student_teacher     = axes[1, 1]
    ax_overlap_u_teacher = axes[2, 0]
    ax_overlap_uv_teacher = axes[2, 1]
    ax_train.set_yscale("log")
    ax_test.set_yscale("log")
    ax_train.set_ylabel("train error")
    ax_test.set_ylabel("test error")
    ax_overlap_v_teacher.set_ylabel(r"$m=\boldsymbol{\omega} \cdot \boldsymbol{\omega}_\star$")
    #ax_overlap_v_teacher.set_ylim([0.9, 1.0])
    #ax_overlap_v_teacher.set_yscale("log")
    #ax_overlap_u_teacher.set_yscale("log")
    ax_overlap_u_teacher.set_ylabel("u")
    ax_overlap_uv_teacher.set_ylabel(fr"$um$")
    ax_overlap_student_teacher.set_ylabel(fr"$um + \mu$")
    
    
    for ax, name in zip(axes.flatten(), ["Train Error", "Test Error", r"Overlap $\boldsymbol{\omega}$-Teacher $m=\boldsymbol{\omega} \cdot \boldsymbol{\omega}_\star$", fr"Overlap Student–Teacher: $\mu+um$", " u ", "um"]):
        ax.set_title(f"{name}  (μ={d_fixed})", fontsize=13, weight="bold")
        ax.set_xlabel("Epoch")
        #ax.set_xscale("log")
        #ax.set_ylabel("Loss")
        #ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    #colors = plt.cm.plasma(np.linspace(0, 1, len(activations)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(activations)))


    for color, act in zip(colors, activations):
        act_dir = os.path.join(base_dir, act)
        try:
            data = load_aggregate(act_dir, d_fixed)
        except FileNotFoundError:
            print(f"⚠️ Missing results for activation '{act}' at μ={d_fixed}")
            continue
        endf = -1
        #train_mean, test_mean, overlap_student_teacher_mean, overlap_v_teacher_mean, overlap_u_teacher_mean, overlap_uv_teacher_mean = data["train_mean"][:endf], data["test_mean"][:endf], data["overlap_student_teacher_mean"][:endf], data["overlap_v_teacher_mean"][:endf], data["overlap_u_teacher_mean"][:endf],data["overlap_uv_teacher_mean"][:endf]
        train_mean = data["train_mean"][:endf]
        train_std  = data["train_std"][:endf]

        test_mean = data["test_mean"][:endf]
        test_std  = data["test_std"][:endf]

        overlap_student_teacher_mean = data["overlap_student_teacher_mean"][:endf]
        overlap_student_teacher_std  = data["overlap_student_teacher_std"][:endf]

        overlap_v_teacher_mean = data["overlap_v_teacher_mean"][:endf]
        overlap_v_teacher_std  = data["overlap_v_teacher_std"][:endf]

        overlap_u_teacher_mean = data["overlap_u_teacher_mean"][:endf]
        overlap_u_teacher_std  = data["overlap_u_teacher_std"][:endf]

        overlap_uv_teacher_mean = data["overlap_uv_teacher_mean"][:endf]
        overlap_uv_teacher_std  = data["overlap_uv_teacher_std"][:endf]

        #print(f" train_mean is {train_mean}")
        #ax_train.plot(train_mean,'--', color=color, label=act, linewidth=2)
        ax_train.plot(train_mean, '--', color=color, label=act, linewidth=2)
        ax_train.fill_between(
            np.arange(len(train_mean)),
            train_mean - train_std,
            train_mean + train_std,
            color=color,
            alpha=0.2
        )
        #ax_test.plot(test_mean,'--', color=color, label=act, linewidth=2)
        ax_test.plot(test_mean, '--', color=color, label=act, linewidth=2)
        ax_test.fill_between(
            np.arange(len(test_mean)),
            test_mean - test_std,
            test_mean + test_std,
            color=color,
            alpha=0.2
        )

        #ax_overlap_student_teacher.plot(overlap_student_teacher_mean,'-', color=color, label=act, linewidth=2)
        ax_overlap_student_teacher.plot(overlap_student_teacher_mean, '-', color=color, label=act, linewidth=2)
        ax_overlap_student_teacher.fill_between(
            np.arange(len(overlap_student_teacher_mean)),
            overlap_student_teacher_mean - overlap_student_teacher_std,
            overlap_student_teacher_mean + overlap_student_teacher_std,
            color=color,
            alpha=0.2
        )

    

        #ax_overlap_v_teacher.plot(overlap_v_teacher_mean,'-', color=color, label=act, linewidth=2)
        ax_overlap_v_teacher.plot(overlap_v_teacher_mean, '-', color=color, label=act, linewidth=2)
        ax_overlap_v_teacher.fill_between(
            np.arange(len(overlap_v_teacher_mean)),
            overlap_v_teacher_mean - overlap_v_teacher_std,
            overlap_v_teacher_mean + overlap_v_teacher_std,
            color=color,
            alpha=0.2
        )

        #ax_overlap_u_teacher.plot(overlap_u_teacher_mean,'-', color=color, label=act, linewidth=2)
        ax_overlap_u_teacher.plot(overlap_u_teacher_mean, '-', color=color, label=act, linewidth=2)
        ax_overlap_u_teacher.fill_between(
            np.arange(len(overlap_u_teacher_mean)),
            overlap_u_teacher_mean - overlap_u_teacher_std,
            overlap_u_teacher_mean + overlap_u_teacher_std,
            color=color,
            alpha=0.2
        )

        #ax_overlap_uv_teacher.plot(overlap_uv_teacher_mean,'-', color=color, label=act, linewidth=2)
        ax_overlap_uv_teacher.plot(overlap_uv_teacher_mean, '-', color=color, label=act, linewidth=2)
        ax_overlap_uv_teacher.fill_between(
            np.arange(len(overlap_uv_teacher_mean)),
            overlap_uv_teacher_mean - overlap_uv_teacher_std,
            overlap_uv_teacher_mean + overlap_uv_teacher_std,
            color=color,
            alpha=0.2
        )

        #ax_overlap_student_teacher.plot(overlap_student_teacher, '-', color=color, label=act, linewidth=2)
        #ax_overlap_v_teacher.plot(overlap_v_teacher, '-', color=color, label=act, linewidth=2)


    for ax in axes.flatten():
        ax.legend(title="Activation", fontsize=9, loc='center right', ncol=2)
    plt.tight_layout()

    if save_pdf:
        out_path = os.path.join(base_dir, f"activation_comparison_d{d_fixed}.eps")
        os.makedirs(base_dir, exist_ok=True)  # ✅ ensure directory exists
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        print(f"📄 Saved: {out_path}")
    plt.show()


# -----------------------
# Argument Parser
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Teacher-Student LoRA results")
    parser.add_argument("--base_dir", type=str, default="results", help="Base results directory")

    # Teacher
    parser.add_argument("--teacher_activation", type=str, default="linear",
                        choices=["linear", "square", "erf", "relu", "sigmoid", "hermite", "hermite+"])
    parser.add_argument("--teacher_hermite_order", type=int, default=None)
    parser.add_argument("--teacher_next_freq", type=int, default=1)

    # Student
    parser.add_argument("--student_activation", type=str, default="linear",
                        choices=["linear", "square", "erf", "relu", "sigmoid", "hermite", "hermite+"])
    parser.add_argument("--student_hermite_order", type=int, default=None)
    parser.add_argument("--student_next_freq", type=int, default=1)

    parser.add_argument("--d_list", nargs="+", default=[0.0, 0.1, 0.5, 0.9])
    parser.add_argument("--compare", action="store_true", help="Compare activations for a fixed d")
    parser.add_argument("--d_fixed", type=float, default=0.5, help="Fixed d for activation comparison")
    parser.add_argument("--seed_list", nargs="+", default=[0, 1])

    return parser.parse_args()

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    args = parse_args()

    # Construct result_dir
    #act_name = f"{args.activation}" if args.activation != "hermite" else f"hermite_He{args.hermite_order}"
    #if args.activation != "hermite" and args.activation != "hermite+":
    #    act_name = f"{args.activation}"
    #elif args.activation == "hermite+":
    #    act_name = f"hermite_He{args.hermite_order}+He{args.hermite_order+args.next_freq}"
    #else:
    #    act_name = f"hermite_He{args.hermite_order}"
    #result_dir = os.path.join(f"results_lr{args.learning_rate}_batch_size{args.batch_size}_input_dimension{args.N}", act_name)
    #result_dir = os.path.join(args.base_dir, act_name)
    
    def make_activation_name(act, hermite_order=None, next_freq=1):
        if act not in ["hermite", "hermite+"]:
            return act
        elif act == "hermite+":
            return f"hermite_He{hermite_order}+He{hermite_order + next_freq}"
        else:
            return f"hermite_He{hermite_order}"

    teacher_act_name = make_activation_name(args.teacher_activation,
                                        args.teacher_hermite_order,
                                        args.teacher_next_freq)

    student_act_name = make_activation_name(args.student_activation,
                                        args.student_hermite_order,
                                        args.student_next_freq)

    # Combine teacher + student for folder naming
    act_name = f"T_{teacher_act_name}S_{student_act_name}"
    result_dir = os.path.join(args.base_dir, act_name)
    
    
    
    # Normalize d_list
    d_list = []
    for d in args.d_list:
        try:
            d_list.append(float(d))
        except ValueError:
            d_list.append(d)

    # Plot dynamics
    plot_dynamics_for_activation(result_dir, act_name, d_list, save_pdf=True)

    # Optionally compare activations for a fixed d
    if args.compare:
        activations = [d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))]
        for d in d_list:
            compare_activations_for_d(args.base_dir, activations, d, save_pdf=True)




    # Optionally compare activations for a fixed d
    """if args.compare:
        # Detect all activations in base_dir
        activations = [d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))]
        
        # Loop over all d in d_list and plot comparison
        for d in args.d_list:
            d_val = d if isinstance(d, str) else f"{d}"
            try:
                d_val = d #float(d)
            except:
                d_val = d  # keep as string, e.g., "V_only"
            print(f"the activation for {d_val} is {args.base_dir} act is {activations}")
            compare_activations_for_d(args.base_dir, activations, d_val, save_pdf=True) """
