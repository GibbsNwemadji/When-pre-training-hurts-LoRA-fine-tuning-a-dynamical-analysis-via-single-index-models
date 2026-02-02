import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


# -----------------------
# ICML-ish style helper
# -----------------------
def set_icml_style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
    })


# -----------------------
# Parse folder metadata
# -----------------------
DIR_RE = re.compile(
    r"results_lr(?P<lr>[\d\.eE+-]+)_width(?P<width>\d+)_rank(?P<rank>\d+)batch_size(?P<bs>\d+)_input_dimension(?P<N>\d+)$"
)

def parse_meta_from_dir(dirname):
    base = os.path.basename(dirname.rstrip("/"))
    m = DIR_RE.match(base)
    if not m:
        return None
    d = m.groupdict()
    d["lr"] = float(d["lr"])
    d["width"] = int(d["width"])
    d["rank"] = int(d["rank"])
    d["bs"] = int(d["bs"])
    d["N"] = int(d["N"])
    return d


def list_activation_dirs(rank_dir):
    acts = []
    for a in sorted(os.listdir(rank_dir)):
        p = os.path.join(rank_dir, a)
        if os.path.isdir(p) and not a.startswith("."):
            acts.append(a)
    return acts


def load_aggregate(act_dir, d):
    d_variants = [d, str(d)]
    try:
        d_variants.append(f"{float(d):.0e}")
    except Exception:
        pass

    for dv in d_variants:
        path = os.path.join(act_dir, f"d_{dv}_aggregate.npz")
        if os.path.exists(path):
            return np.load(path)
    raise FileNotFoundError(f"No aggregate file for μ={d} in {act_dir}")


def prep_axes(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.7)
    ax.set_xlabel("Epoch (log)")
    ax.set_ylabel("Test MSE (log)")


# -----------------------
# Discover structure
# -----------------------
def discover_runs(base_parent, lr, width, bs, N):
    """
    Returns:
      rank_to_dir: dict rank -> directory
      activation_set: sorted list of all activations found across ranks
      ranks_sorted: sorted list of ranks found
    """
    rank_dirs = []
    for p in sorted(glob.glob(os.path.join(base_parent, "results_lr*_width*_rank*batch_size*_input_dimension*"))):
        meta = parse_meta_from_dir(p)
        if meta is None:
            continue
        if abs(meta["lr"] - lr) > 1e-12:
            continue
        if meta["width"] != width or meta["bs"] != bs or meta["N"] != N:
            continue
        rank_dirs.append((meta["rank"], p))

    if not rank_dirs:
        raise FileNotFoundError(
            f"No directories found in {base_parent} matching "
            f"lr={lr}, width={width}, bs={bs}, N={N}."
        )

    rank_dirs = sorted(rank_dirs, key=lambda x: x[0])
    rank_to_dir = {r: p for r, p in rank_dirs}

    activation_set = set()
    for r, p in rank_dirs:
        for act in list_activation_dirs(p):
            activation_set.add(act)

    return rank_to_dir, sorted(list(activation_set)), [r for r, _ in rank_dirs]


# -----------------------
# Plot: rows = activations, cols = ranks
# -----------------------
def plot_activation_by_rank_grid(base_parent, lr, width, bs, N, d_list, save_pdf=True):
    set_icml_style()

    rank_to_dir, activations, ranks = discover_runs(base_parent, lr, width, bs, N)

    nrows = len(activations)
    ncols = len(ranks)

    # figure size that scales with grid
    fig_w = max(2.6 * ncols, 6.8)
    fig_h = max(2.0 * nrows, 2.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(d_list)))

    for i, act in enumerate(activations):
        for j, r in enumerate(ranks):
            ax = axes[i, j]
            prep_axes(ax)

            act_dir = os.path.join(rank_to_dir[r], act)

            # Titles: top row shows rank, first col shows activation
            if i == 0:
                ax.set_title(f"rank r={r}")
            if j == 0:
                ax.text(-0.35, 0.5, f"{act}", rotation=90,
                        va="center", ha="center", transform=ax.transAxes)

            if not os.path.isdir(act_dir):
                ax.text(0.5, 0.5, "missing act", ha="center", va="center", transform=ax.transAxes)
                continue

            handles, labels = [], []
            for c, dval in zip(colors, d_list):
                try:
                    data = load_aggregate(act_dir, dval)
                except FileNotFoundError:
                    continue

                test_mean = data["test_mean"]
                test_std  = data["test_std"]

                x = np.arange(1, len(test_mean) + 1)
                h = ax.plot(x, test_mean, "-", color=c)[0]
                ax.fill_between(x, test_mean - test_std, test_mean + test_std, color=c, alpha=0.15)

                handles.append(h)
                labels.append(fr"$μ={dval}$")

            # only put legend once per row (rightmost plot) to reduce clutter
            if handles and j == ncols - 1:
                ax.legend(handles, labels, frameon=False, loc="best", ncol=1)

    """fig.suptitle(
        f"Test loss grid | lr={lr}, width(K)={width}, bs={bs}, N={N}\n"
        f"Rows: activation | Cols: rank | Curves: d in {list(d_list)}",
        y=1.01
    )"""
    fig.tight_layout()

    if save_pdf:
        out = os.path.join(base_parent, f"grid_testloss_rowsACT_colsRANK_lr{lr}_width{width}_bs{bs}_N{N}.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"📄 Saved: {out}")

    plt.show()


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_parent", type=str, default=".", help="Folder containing results_lr... directories")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--d_list", nargs="+", default=["0.1", "0.5", "0.9"])
    p.add_argument("--no_save", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_activation_by_rank_grid(
        base_parent=args.base_parent,
        lr=args.lr,
        width=args.width,
        bs=args.batch_size,
        N=args.N,
        d_list=args.d_list,
        save_pdf=(not args.no_save),
    )
