import os
import glob
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter


# -----------------------
# ICML 2026 style helper
# -----------------------

def set_icml_style():
    """
    Rough ICML / two-column conference style.

    - Single or double-column width
    - Small fonts (8pt)
    - Thin lines, light grid
    - Vector-friendly fonts
    """
    plt.rcParams.update({
        "figure.figsize": (6.75, 2.3),   # ~ two-column width, short height
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # embed fonts (good for Illustrator / ICML)
        "ps.fonttype": 42,
        "axes.grid": True,
    })
    # If your environment supports LaTeX and you want it:
    # plt.rcParams["text.usetex"] = True


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


def load_aggregate(result_dir, d):
    d_variants = [d, str(d)]
    try:
        d_variants.append(f"{float(d):.0e}")
    except Exception:
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
    # Apply ICML-like style
    set_icml_style()

    # 3 panels in a row, two-column style
    fig, axes = plt.subplots(1, 3, sharex=True)

    ax_test                = axes[0]
    ax_overlap_v_teacher   = axes[1]
    ax_overlap_student_teacher = axes[2]

    # Y labels
    ax_test.set_ylabel("MSE")
    ax_overlap_v_teacher.set_ylabel(r"$m = \boldsymbol{\omega} \cdot \boldsymbol{\omega}_\star$")
    ax_overlap_student_teacher.set_ylabel(r"$\mu + u m$")

    # X axis = time (log-scale)
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.7)

    # Panel titles (small, no bold)
    #ax_test.set_title("Test error")
    #ax_overlap_v_teacher.set_title(r"Overlap $m$")
    #ax_overlap_student_teacher.set_title(r"Overlap $\mu + u m$")

    # Color palette: tab10 (journal-friendly)
    colors = plt.cm.tab10(np.linspace(0, 1, len(d_list)))

    # For global legend
    legend_handles = []
    legend_labels = []

    for color, d in zip(colors, d_list):
        try:
            data = load_aggregate(result_dir, d)
        except FileNotFoundError:
            print(f"⚠️ Aggregate missing for μ={d}")
            continue

        endf = -1
        #endf = 21000

        test_mean, test_std = data["test_mean"][:endf], data["test_std"][:endf]
        overlap_v_teacher_mean, overlap_v_teacher_std = (
            data["overlap_v_teacher_mean"][:endf],
            data["overlap_v_teacher_std"][:endf],
        )
        overlap_student_teacher_mean, overlap_student_teacher_std = (
            data["overlap_student_teacher_mean"][:endf],
            data["overlap_student_teacher_std"][:endf],
        )

        x_axis = np.arange(len(test_mean))

        label = fr"$\mu={d}$"

        # Test error
        h_test = ax_test.plot(
            x_axis, test_mean, "-", color=color, label=label, linewidth=1.0
        )[0]
        ax_test.fill_between(
            x_axis,
            test_mean - test_std,
            test_mean + test_std,
            color=color,
            alpha=0.15,
        )

        # Overlap student–teacher
        ax_overlap_student_teacher.plot(
            x_axis,
            overlap_student_teacher_mean,
            "-",
            color=color,
            linewidth=1.0,
        )
        ax_overlap_student_teacher.fill_between(
            x_axis,
            overlap_student_teacher_mean - overlap_student_teacher_std,
            overlap_student_teacher_mean + overlap_student_teacher_std,
            color=color,
            alpha=0.15,
        )

        # Overlap v–teacher
        ax_overlap_v_teacher.plot(
            x_axis,
            overlap_v_teacher_mean,
            "-",
            color=color,
            linewidth=1.0,
        )
        ax_overlap_v_teacher.fill_between(
            x_axis,
            overlap_v_teacher_mean - overlap_v_teacher_std,
            overlap_v_teacher_mean + overlap_v_teacher_std,
            color=color,
            alpha=0.15,
        )

        legend_handles.append(h_test)
        legend_labels.append(label)

    # Only test error axis is log on y (if you want)
    ax_test.set_yscale("log")

    # X-label on last row only
    for ax in axes:
        pass
        #ax.set_xlabel("GD steps")

    # Global legend above all subplots
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=min(len(legend_labels), 5),
            frameon=False,
        )
        pass

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])

    if save_pdf:
        out_path = os.path.join(result_dir, f"{activation_name}_train_test_dynamics_icml.pdf")
        os.makedirs(result_dir, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
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
    
    
    for ax, name in zip(axes.flatten(), ["Train Error", "Test Error", r"Overlap $\boldsymbol{\omega}-$Teacher $m=\boldsymbol{\omega} \cdot \boldsymbol{\omega}_\star$", fr"Overlap Student–Teacher: $\mu+um$", " u ", "um"]):
        ax.set_title(f"{name}  (μ={d_fixed})", fontsize=13, weight="bold")
        ax.set_xlabel("Epoch")
        #ax.set_xscale("log")
        #ax.set_ylabel("Loss")
        #ax.set_yscale("log")
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    colors = plt.cm.plasma(np.linspace(0, 1, len(activations)))

    for color, act in zip(colors, activations):
        act_dir = os.path.join(base_dir, act)
        try:
            data = load_aggregate(act_dir, d_fixed)
        except FileNotFoundError:
            print(f"⚠️ Missing results for activation '{act}' at μ={d_fixed}")
            continue
        endf = -1
        #endf = 21000
        train_mean, test_mean, overlap_student_teacher_mean, overlap_v_teacher_mean, overlap_u_teacher_mean, overlap_uv_teacher_mean = data["train_mean"][:endf], data["test_mean"][:endf], data["overlap_student_teacher_mean"][:endf], data["overlap_v_teacher_mean"][:endf], data["overlap_u_teacher_mean"][:endf],data["overlap_uv_teacher_mean"][:endf]
        #print(f" train_mean is {train_mean}")
        ax_train.plot(train_mean,'--', color=color, label=act, linewidth=2)
        ax_test.plot(test_mean,'--', color=color, label=act, linewidth=2)
        ax_overlap_student_teacher.plot(overlap_student_teacher_mean,'-', color=color, label=act, linewidth=2)
        ax_overlap_v_teacher.plot(overlap_v_teacher_mean,'-', color=color, label=act, linewidth=2)
        ax_overlap_u_teacher.plot(overlap_u_teacher_mean,'-', color=color, label=act, linewidth=2)
        ax_overlap_uv_teacher.plot(overlap_uv_teacher_mean,'-', color=color, label=act, linewidth=2)
        #ax_overlap_student_teacher.plot(overlap_student_teacher, '-', color=color, label=act, linewidth=2)
        #ax_overlap_v_teacher.plot(overlap_v_teacher, '-', color=color, label=act, linewidth=2)


    for ax in axes.flatten():
        ax.legend(title="Activation", fontsize=9, loc='center right', ncol=2)
    plt.tight_layout()

    if save_pdf:
        out_path = os.path.join(base_dir, f"activation_comparison_d{d_fixed}.pdf")
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
    parser.add_argument("--activation", type=str, default="linear",
                        choices=["linear", "square", "erf", "relu", "sigmoid", "hermite", "hermite+"])
    parser.add_argument("--hermite_order", type=int, default=None)
    parser.add_argument("--d_list", nargs="+", default=["V_only", 0.0, 0.1, 0.5, 0.9])
    parser.add_argument("--compare", action="store_true", help="Compare activations for a fixed d")
    parser.add_argument("--d_fixed", type=float, default=0.5, help="Fixed d for activation comparison")
    parser.add_argument("--seed_list", nargs="+", default=[0, 1])
    parser.add_argument("--next_freq", type=int, default=1)

    return parser.parse_args()


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    args = parse_args()

    # Construct result_dir
    #act_name = f"{args.activation}" if args.activation != "hermite" else f"hermite_He{args.hermite_order}"
    if args.activation != "hermite" and args.activation != "hermite+":
        act_name = f"{args.activation}"
    elif args.activation == "hermite+":
        act_name = f"hermite_He{args.hermite_order}+He{args.hermite_order+args.next_freq}"
    else:
        act_name = f"hermite_He{args.hermite_order}"
    #result_dir = os.path.join(f"results_lr{args.learning_rate}_batch_size{args.batch_size}_input_dimension{args.N}", act_name)
    result_dir = os.path.join(args.base_dir, act_name)

    # Plot dynamics for activation
    plot_dynamics_for_activation(result_dir, act_name, args.d_list, save_pdf=True)

    # Optionally compare activations for a fixed d
    if args.compare:
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
            compare_activations_for_d(args.base_dir, activations, d_val, save_pdf=True)

