#!/usr/bin/env python3
# ============================================================
# 📊 Compare Time-to-Overlap Curves Across Activations
# ============================================================
# Robust version: skips activations with missing or empty files
# ============================================================

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import re

plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# 🔍 Helper functions
# ============================================================

def find_time_to_overlap_files(base_dir):
    """Return {activation_name: npz_path} for all detected activations."""
    matches = {}
    for subdir in sorted(os.listdir(base_dir)):
        act_dir = os.path.join(base_dir, subdir)
        if not os.path.isdir(act_dir):
            continue
        npz_files = glob.glob(os.path.join(act_dir, "*_time_to_overlap1.npz"))
        if npz_files:
            matches[subdir] = npz_files[0]
    return matches


def load_time_to_overlap(npz_path):
    """Load d_values and time_to_overlap arrays safely."""
    if not os.path.exists(npz_path):
        print(f"⚠️ File not found, skipping: {npz_path}")
        return None, None
    try:
        data = np.load(npz_path)
        d_vals = data["d_values"]
        t_vals = data["time_to_overlap"]
        idx = np.argsort(d_vals)
        return d_vals[idx], t_vals[idx]
    except (EOFError, KeyError) as e:
        print(f"⚠️ Skipping corrupted file {npz_path}: {e}")
        return None, None


# ============================================================
# 🧩 Main plotting routine
# ============================================================

def plot_all_time_to_overlap(base_dir, save=True):
    files = find_time_to_overlap_files(base_dir)
    if not files:
        print(f"⚠️ No *_time_to_overlap1.npz found under {base_dir}")
        return

    plt.figure(figsize=(8, 6))

    # dictionary to store results for saving later
    save_dict = {}

    for activation, path in files.items():
        # Extract the Hermite degree k
        # Expected format: "hermite_He1", "hermite_He3", ...
       
        # Extract all Hermite indices (e.g. He1+He2 -> [1, 2])
        try:
            k_list = [int(x) for x in re.findall(r"He(\d+)", activation)]
            if not k_list:
                raise ValueError
        except Exception as e:
            print(f"⚠️ Could not parse Hermite indices from activation: {activation}")
            k_list = []

        # Build a combined label, e.g. "He1+He2"
        label = "+".join([f"He{k}" for k in k_list]) if k_list else activation

        # Load data once (since this is a combined activation)
        print(f"the path were they extract the data is {path}")
        d_vals, t_vals = load_time_to_overlap(path)
        print(f"the actual data are: d_vals={d_vals}, t_vals={t_vals}")
        if d_vals is not None and t_vals is not None:
            # Store in dictionary using combined key
            key_prefix = "_".join([str(k) for k in k_list])
            save_dict[f"{key_prefix}_d"] = d_vals
            save_dict[f"{key_prefix}_t"] = t_vals

            # Plot with combined label
            #print(f"{activation}: d_vals={d_vals}, t_vals={t_vals}")
            plt.plot(d_vals, t_vals, "-o", lw=2, label=label)
        else:
            print(f"⚠️ Skipping {activation}: missing data")

    plt.xlabel("d")
    plt.yscale("log")
    plt.ylabel("Time to overlap")
    plt.legend(loc="center left")
    plt.tight_layout()

    out_path = os.path.join(base_dir, "compare_time_to_overlap_all.png")
    if save:
        plt.savefig(out_path, dpi=300)
        print(f"✅ Plot saved → {out_path}")
    plt.show()
    
    output_file = os.path.join(base_dir, "time_to_overlap_by_k.npz")
    np.savez(output_file, **save_dict)
    print(f"✅ Saved grouped Hermite data to: {output_file}")


# ============================================================
# 🚀 CLI entry
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare time-to-overlap curves across activations")
    parser.add_argument("--base_dir", type=str, required=True, help="Directory containing activation subfolders")
    args = parser.parse_args()

    print(f"📁 Scanning base directory: {args.base_dir}")
    plot_all_time_to_overlap(args.base_dir)
