#!/usr/bin/env python3
# ============================================================
# 📈 Time-to-Overlap=1 Plotter + Auto-Run plot_time_to_overlap.py
# ============================================================

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import subprocess
import re

plt.style.use("seaborn-v0_8-whitegrid")


# ============================================================
# Utility Functions
# ============================================================

def extract_d_from_filename(filename: str):
    """Extracts the numeric or string d value from filename like 'd_0.1_aggregate.npz'"""
    base = os.path.basename(filename)
    part = base.replace("d_", "").replace("_aggregate.npz", "")
    try:
        return float(part)
    except ValueError:
        return part  # e.g. "V_only"


def find_first_time_overlap_ge1(overlap_array, d):
    indices = np.where(
        (overlap_array >= d - 0.05) & (overlap_array <= d + 0.05)
    )[0]

    print(f"Last five elements: {overlap_array[-5:]}")
    print(f"Indices: {indices}")

    if len(indices) == 0:
        return None

    return np.mean(indices), np.std(indices)

def compute_time_to_overlap(result_dir):
    """Loops through all 'd_*_aggregate.npz' in result_dir,
    extracts overlap_student_teacher_mean, and finds when it ≥ 1."""
    #files = sorted(glob.glob(os.path.join(result_dir, "d_*_aggregate.npz")))
    #result_dir = "path/to/results"
    all_files = glob.glob(os.path.join(result_dir, "d_*_aggregate.npz"))

    # Filter files to ensure the part after "d_" is a valid number
    files = sorted(f for f in all_files if re.match(r".*d_\d+(\.\d+)?_aggregate\.npz$", f))
    if not files:
        raise FileNotFoundError(f"No aggregate files found in {result_dir}")

    d_values, times, std_times = [], [], []
    for fpath in files:
        d = extract_d_from_filename(fpath)
        data = np.load(fpath)

        if "overlap_student_teacher_mean" not in data:
            print(f"⚠️ Missing key in {fpath}, skipping...")
            continue

        overlap = data["overlap_v_teacher_mean"]
        overlap_std = data["overlap_v_teacher_std"]
        print(f"for {d} the last five element is {overlap[-5:]} with directory {fpath}")
        t,std_t = find_first_time_overlap_ge1(overlap,d)
        print(f"the value uses for {d} is {t}")

        if t is not None:
            d_values.append(d)
            times.append(t)
            std_times.append(std_t)
            print(f"✅ d={d} → reached ≥1 at t={t}")
        else:
            print(f"❌ d={d} never reached ≥1")

    return np.array(d_values), np.array(times), np.array(std_times)


def save_and_plot_time_to_overlap(result_dir, activation_name):
    """Compute and save the 'time to overlap=1' data, then plot it."""
    d_vals, t_vals, std_vals = compute_time_to_overlap(result_dir)
    sorted_idx = np.argsort(d_vals)
    d_vals, t_vals, std_vals = d_vals[sorted_idx], t_vals[sorted_idx], std_vals[sorted_idx]

    # Save
    out_data_path = os.path.join(result_dir, f"{activation_name}_time_to_overlap1.npz")
    np.savez(out_data_path, d_values=d_vals, time_to_overlap=t_vals, std_time_to_overlap=std_vals)
    print(f"💾 Saved → {out_data_path}")

    # Plot
    plt.figure(figsize=(7, 5))
    #plt.plot(d_vals, t_vals, "o", lw=2, color="royalblue")
    plt.errorbar(
    d_vals,
    t_vals,
    yerr=std_vals,
    fmt="o",
    lw=2,
    capsize=4)
    plt.xlabel(" μ ")
    plt.ylabel("Time index where overlap ≥ 1")
    plt.title(f"Time to reach overlap ≥ 1 ({activation_name})")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=False))
    plt.tight_layout()

    out_fig_path = os.path.join(result_dir, f"{activation_name}_time_to_overlap1.png")
    plt.savefig(out_fig_path, dpi=300)
    print(f"📊 Plot saved → {out_fig_path}")
    plt.show()


def run_plot_fix_activation(base_dir, activation_name):
    """Automatically runs plot_time_to_overlap.py with appropriate arguments"""
    cmd = ["python3", "plot_time_to_overlap.py", "--base_dir", base_dir, "--activation", activation_name]

    # Try to detect hermite order
    match = re.search(r"He(\d+)", activation_name)
    if match:
        cmd.extend(["--hermite_order", match.group(1)])

    print(f"\n🚀 Launching plot_time_to_overlap.py for {activation_name} ...")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print("✅ plot_time_to_overlap.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ plot_time_to_overlap.py failed with error code {e.returncode}")


# ============================================================
# CLI Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Compute and plot time-to-overlap=1 and auto-run plot_time_to_overlap.py")
    parser.add_argument("--base_dir", type=str, default="results", help="Base results directory")
    #parser.add_argument("--activation", type=str, required=True, help="Activation subdirectory name (e.g. relu, hermite_He1)")
    parser.add_argument("--teacher_activation", type=str, required=True, help="Activation subdirectory name (e.g. relu, hermite_He1)")
    parser.add_argument("--student_activation", type=str, required=True, help="Activation subdirectory name (e.g. relu, hermite_He1)")

    parser.add_argument("--next_freq", type=int, default=1)
    parser.add_argument("--hermite_order", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path=f"T_{args.teacher_activation}S_{args.student_activation}"
    result_dir = os.path.join(args.base_dir, path)
    """if args.activation=="hermite":
        
        path=f"hermite_He{args.hermite_order}"
        result_dir = os.path.join(args.base_dir, path)
        args.activation=path
    elif args.activation=="hermite+":
        path=f"hermite_He{args.hermite_order}+He{args.hermite_order+args.next_freq}"
        result_dir = os.path.join(args.base_dir, path)
        args.activation=path"""
        
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Activation directory not found: {result_dir}")

    save_and_plot_time_to_overlap(result_dir, path)
    #run_plot_fix_activation(args.base_dir, args.activation)
