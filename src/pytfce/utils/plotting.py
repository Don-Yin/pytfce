"""Publication-quality plots for the pTFCE optimization suite.

All figures use `-` separators in filenames and are saved to results/plots/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "unenhanced": "#9E9E9E",
    "R_TFCE": "#90CAF9",
    "R_pTFCE": "#607D8B",
    "Py_TFCE": "#2196F3",
    "Py_pTFCE": "#E91E63",
    "pTFCE_baseline": "#E91E63",
}

BASELINE_ORDER = ["R_TFCE", "R_pTFCE", "Py_TFCE", "Py_pTFCE"]
METHOD_ORDER = BASELINE_ORDER


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_all_plots(results_dir: Path) -> None:
    """Generate all publication figures from experiment results."""
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    base = _load_json(results_dir / "baseline" / "diagnostics.json")
    py_tfce = _load_json(results_dir / "py-tfce-perm" / "diagnostics.json")
    r_ptfce = _load_json(results_dir / "r-ptfce" / "diagnostics.json")
    r_tfce = _load_json(results_dir / "r-tfce-perm" / "diagnostics.json")

    if base:
        _fig01_spatial(results_dir, plots_dir)
        _fig02_method_progression(base, py_tfce, r_ptfce, r_tfce, plots_dir)
        _fig03_amplitude_ablation(base, plots_dir)
        _fig04_shape_ablation(base, plots_dir)
        _fig05_runtime(base, py_tfce, r_ptfce, r_tfce, plots_dir)
        _fig06_null_calibration(base, plots_dir)

    _fig10_volume_scaling(base, plots_dir)

    print(f"  All plots saved to {plots_dir}")


# ═════════════════════════════════════════════════════════════════════
# fig-01: spatial detection maps
# ═════════════════════════════════════════════════════════════════════
def _fig01_spatial(results_dir: Path, plots_dir: Path) -> None:
    base_npz = results_dir / "baseline" / "spatial.npz"
    if not base_npz.exists():
        return
    d = np.load(base_npz)
    Z, brain, gt = d["Z"], d["brain"], d["gt"]
    det_ptfce = d["det_ptfce"]

    rows = [
        ("ground truth", gt.astype(float), "Reds"),
        ("input Z-score", Z, "hot"),
    ]

    r_tfce_npz = results_dir / "r-tfce-perm" / "spatial.npz"
    r_ptfce_npz = results_dir / "r-ptfce" / "spatial.npz"
    py_tfce_npz = results_dir / "py-tfce-perm" / "spatial.npz"

    if r_tfce_npz.exists():
        rd = np.load(r_tfce_npz, allow_pickle=True)
        if "det_r_tfce" in rd:
            rows.append(("R TFCE (perm)", rd["det_r_tfce"].astype(float), "Blues"))

    if r_ptfce_npz.exists():
        rd = np.load(r_ptfce_npz, allow_pickle=True)
        if "det_r_ptfce" in rd:
            rows.append(("R pTFCE", rd["det_r_ptfce"].astype(float), "BuGn"))

    if py_tfce_npz.exists():
        td = np.load(py_tfce_npz, allow_pickle=True)
        if "det_tfce" in td:
            rows.append(("Py TFCE (200p)", td["det_tfce"].astype(float), "PuBu"))

    rows.append(("Py pTFCE", det_ptfce.astype(float), "RdPu"))

    mid = Z.shape[2] // 2
    slices = [mid - 5, mid, mid + 5]
    n_rows = len(rows)

    fig, axes = plt.subplots(n_rows, 3, figsize=(7, 1.6 * n_rows), dpi=150,
                             gridspec_kw={"hspace": 0.12, "wspace": 0.02})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, (title, data, cmap) in enumerate(rows):
        vmax = max(float(data.max()), 1) if ri == 1 else 1
        for ci, sl in enumerate(slices):
            ax = axes[ri, ci]
            ax.imshow(data[:, :, sl].T, cmap=cmap, vmin=0, vmax=vmax,
                      origin="lower", interpolation="nearest")
            ax.contour(brain[:, :, sl].T, levels=[0.5],
                       colors="white", linewidths=0.3, alpha=0.3)
            if ri >= 2:
                ax.contour(gt[:, :, sl].T, levels=[0.5],
                           colors="lime", linewidths=0.8, linestyles="--")
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(title, fontsize=7, fontweight="bold")
            if ri == 0:
                ax.set_title(f"z={sl}", fontsize=8)

    fig.suptitle("Spatial detection comparison\n(dashed green = ground truth)",
                 fontsize=10, fontweight="bold", y=1.0)
    fig.savefig(plots_dir / "fig-01-spatial-comparison.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    fig-01-spatial-comparison.png")


# ═════════════════════════════════════════════════════════════════════
# fig-02: 4 baselines + enhancements (TPR / Dice / runtime)
# ═════════════════════════════════════════════════════════════════════
def _fig02_method_progression(base, py_tfce, r_ptfce, r_tfce, plots_dir):
    methods, tprs, dices, times, colors = [], [], [], [], []

    if r_tfce and "detection" in r_tfce:
        det = r_tfce["detection"]
        methods.append("R TFCE\n(perm)")
        tprs.append(det.get("tpr", 0)); dices.append(det.get("dice", 0))
        times.append(r_tfce.get("elapsed_s", 0)); colors.append(COLORS["R_TFCE"])

    if r_ptfce and "detection" in r_ptfce:
        det = r_ptfce["detection"]
        methods.append("R pTFCE")
        tprs.append(det.get("tpr", 0)); dices.append(det.get("dice", 0))
        times.append(r_ptfce.get("elapsed_s", 0)); colors.append(COLORS["R_pTFCE"])

    if py_tfce and "TFCE_200p" in py_tfce:
        m = py_tfce["TFCE_200p"]
        methods.append("Py TFCE\n(200p)")
        tprs.append(m.get("tpr", 0)); dices.append(m.get("dice", 0))
        times.append(m.get("time_s", 0)); colors.append(COLORS["Py_TFCE"])

    mc = base.get("main_comparison", {}).get("methods", {})
    if "pTFCE_baseline" in mc:
        m = mc["pTFCE_baseline"]
        methods.append("Py pTFCE")
        tprs.append(m.get("tpr", 0)); dices.append(m.get("dice", 0))
        times.append(m.get("time_s", 0)); colors.append(COLORS["Py_pTFCE"])

    if not methods:
        return

    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=150)
    x = np.arange(len(methods))
    w = 0.35
    ax1.bar(x - w/2, tprs, w, label="TPR", color=colors, alpha=0.85)
    ax1.bar(x + w/2, dices, w, label="Dice", color=colors, alpha=0.55,
            edgecolor=colors, linewidth=1.5, linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8)
    ax1.set_ylabel("TPR / Dice", fontsize=10)
    ax1.set_ylim(0, 1.15)
    _style_ax(ax1)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, times, "ko-", ms=6, lw=1.5, label="time (s)")
    ax2.set_ylabel("runtime (s)", fontsize=10)
    ax2.set_yscale("log")
    _style_ax(ax2)
    ax2.legend(loc="upper right", frameon=False, fontsize=9)

    n_baselines = sum(1 for m in methods if not m.startswith("+"))
    if 0 < n_baselines < len(methods):
        ax1.axvline(n_baselines - 0.5, color="gray", ls=":", lw=1, alpha=0.5)

    fig.suptitle("4 baselines + proposed enhancements (TPR / Dice / runtime)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(plots_dir / "fig-02-method-progression.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    fig-02-method-progression.png")


# ═════════════════════════════════════════════════════════════════════
# fig-03: amplitude ablation
# ═════════════════════════════════════════════════════════════════════
def _fig03_amplitude_ablation(base, plots_dir):
    amps = ["weak", "medium", "strong", "very_strong"]
    amp_labels = ["0.3", "0.5", "0.8", "1.2"]

    series = {}
    if base and "amplitude_ablation" in base:
        series["pTFCE baseline"] = (base["amplitude_ablation"], COLORS["pTFCE_baseline"])

    if not series:
        return

    n_groups = len(amps)
    n_series = len(series)
    w = 0.8 / n_series
    x = np.arange(n_groups)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    for si, (sname, (data, color)) in enumerate(series.items()):
        tprs = [data.get(a, {}).get("tpr", 0) for a in amps]
        dices = [data.get(a, {}).get("dice", 0) for a in amps]
        offset = (si - n_series / 2 + 0.5) * w
        ax1.bar(x + offset, tprs, w, label=sname, color=color, alpha=0.85)
        ax2.bar(x + offset, dices, w, label=sname, color=color, alpha=0.85)

    for ax, metric in [(ax1, "TPR"), (ax2, "Dice")]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"a={l}" for l in amp_labels], fontsize=9)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(frameon=False, fontsize=8)
        _style_ax(ax)
        ax.set_title(f"({'a' if metric == 'TPR' else 'b'}) {metric} vs amplitude",
                     fontweight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(plots_dir / "fig-03-amplitude-ablation.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    fig-03-amplitude-ablation.png")


# ═════════════════════════════════════════════════════════════════════
# fig-04: shape ablation
# ═════════════════════════════════════════════════════════════════════
def _fig04_shape_ablation(base, plots_dir):
    shapes = ["small", "large", "elongated", "multi"]
    data = base.get("shape_ablation", {})
    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = np.arange(len(shapes))
    w = 0.35
    tprs = [data.get(s, {}).get("tpr", 0) for s in shapes]
    dices = [data.get(s, {}).get("dice", 0) for s in shapes]
    ax.bar(x - w/2, tprs, w, label="TPR", color=COLORS["pTFCE_baseline"], alpha=0.85)
    ax.bar(x + w/2, dices, w, label="Dice", color=COLORS["pTFCE_baseline"], alpha=0.5)
    for i, s in enumerate(shapes):
        gv = data.get(s, {}).get("gt_voxels", "?")
        ax.text(i, max(tprs[i], dices[i]) + 0.03, f"{gv}vox",
                ha="center", fontsize=7, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, fontsize=9)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=9)
    _style_ax(ax)
    ax.set_title("pTFCE sensitivity by signal shape", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "fig-04-shape-ablation.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    fig-04-shape-ablation.png")


# ═════════════════════════════════════════════════════════════════════
# fig-05: runtime — 4 baselines + enhancements + TFCE scaling
# ═════════════════════════════════════════════════════════════════════
def _fig05_runtime(base, py_tfce, r_ptfce, r_tfce, plots_dir):
    methods, times, colors = [], [], []

    if r_tfce and "elapsed_s" in r_tfce:
        methods.append("R TFCE\n(perm)")
        times.append(r_tfce["elapsed_s"])
        colors.append(COLORS["R_TFCE"])

    if r_ptfce and "elapsed_s" in r_ptfce:
        methods.append("R pTFCE")
        times.append(r_ptfce["elapsed_s"])
        colors.append(COLORS["R_pTFCE"])

    if py_tfce and "TFCE_200p" in py_tfce:
        methods.append("Py TFCE\n(200p)")
        times.append(py_tfce["TFCE_200p"].get("time_s", 0))
        colors.append(COLORS["Py_TFCE"])

    mc = base.get("main_comparison", {}).get("methods", {})
    if "pTFCE_baseline" in mc:
        methods.append("Py pTFCE")
        times.append(mc["pTFCE_baseline"].get("time_s", 0))
        colors.append(COLORS["Py_pTFCE"])

    # Py TFCE scaling
    if py_tfce:
        for k in ["TFCE_500p", "TFCE_1000p"]:
            if k in py_tfce:
                methods.append(f"Py TFCE\n{k.split('_')[1]}")
                times.append(py_tfce[k].get("time_s", 0))
                colors.append(COLORS["Py_TFCE"])

    if not methods:
        return

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    x = np.arange(len(methods))
    bars = ax.bar(x, times, color=colors, alpha=0.85)
    for b, t in zip(bars, times):
        label = f"{t:.1f}s" if t < 10 else f"{t:.0f}s"
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(times)*0.01,
                label, ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel("wall-clock time (s)")
    _style_ax(ax)

    n_baselines = sum(1 for m in methods if not m.startswith("+")
                      and "500p" not in m and "1000p" not in m)
    if 0 < n_baselines < len(methods):
        ax.axvline(n_baselines - 0.5, color="gray", ls=":", lw=1, alpha=0.5)
        ax.text(n_baselines - 0.6, max(times) * 0.95, "baselines \u2190",
                ha="right", va="top", fontsize=7, color="gray")
        ax.text(n_baselines - 0.4, max(times) * 0.95, "\u2192 enhancements",
                ha="left", va="top", fontsize=7, color="gray")

    ax.set_title("Runtime comparison: 4 baselines + proposed enhancements",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "fig-05-runtime-scaling.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    fig-05-runtime-scaling.png")


# ═════════════════════════════════════════════════════════════════════
# fig-06 – fig-10 (unchanged logic, carried forward)
# ═════════════════════════════════════════════════════════════════════
def _fig06_null_calibration(base, plots_dir):
    nc = base.get("null_calibration", {})
    rej = nc.get("rejection_rate_p005", 0)
    fwer = nc.get("fwer_any_rejection", False)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.bar(["pTFCE baseline"], [rej], color=COLORS["pTFCE_baseline"], alpha=0.85)
    ax.axhline(0.05, color="black", ls="--", lw=1, label="nominal \u03b1=0.05")
    ax.set_ylabel("rejection rate")
    ax.set_ylim(0, 0.15)
    ax.legend(frameon=False)
    _style_ax(ax)
    ax.set_title(f"Null calibration (FWER rejection: {fwer})",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / "fig-06-null-calibration.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    fig-06-null-calibration.png")


def _fig10_volume_scaling(base, plots_dir):
    if not base or "volume_ablation" not in base:
        return

    vdata = base["volume_ablation"]
    vol_names = ["small", "medium", "default"]
    vol_labels = []
    for vn in vol_names:
        if vn in vdata:
            sh = vdata[vn].get("shape", [0, 0, 0])
            vol_labels.append(f"{sh[0]}\u00b3")
        else:
            vol_labels.append(vn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    x = np.arange(len(vol_names))
    ts = [vdata.get(vn, {}).get("time_s", 0) for vn in vol_names]
    dices = [vdata.get(vn, {}).get("dice", 0) for vn in vol_names]

    ax1.bar(x, ts, color=COLORS["pTFCE_baseline"], alpha=0.85)
    ax2.bar(x, dices, color=COLORS["pTFCE_baseline"], alpha=0.85)

    for ax, metric, title in [
        (ax1, "time (s)", "(a) Runtime vs volume"),
        (ax2, "Dice", "(b) Detection vs volume"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(vol_labels, fontsize=9)
        ax.set_ylabel(metric, fontsize=10)
        _style_ax(ax)
        ax.set_title(title, fontweight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(plots_dir / "fig-10-volume-scaling.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("    fig-10-volume-scaling.png")
