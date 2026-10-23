"""
io_curve_fit.py
===============
Constrained 4-parameter Boltzmann fit of a TMS input/output (recruitment) curve,
ported from the MATLAB version (lsqcurvefit, trust-region-reflective).

Model:  MEP(x) = vmin + (vmax - vmin) / (1 + exp((S50 - x) / k))
        vmin = baseline MEP, vmax = saturation MEP, S50 = intensity at the midpoint, k = slope factor

Used by io_curve.py (the NeuroSimo decider) at the end of the session, and can be re-run
offline on a saved trials CSV, e.g. with a different baseline-RMS cutoff:

    python io_curve_fit.py results/7/io_curve/io_curve_trials.csv --subject 7 --cutoff 15

Only needs numpy, scipy and matplotlib.
"""

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

# Default QC / windows. io_curve.py passes its own values (they are also stored in the CSV/JSON).
DEFAULT_BASELINE_RMS_CUTOFF = 15.0          # uV
DEFAULT_MEP_WINDOW = (0.025, 0.050)         # s
DEFAULT_BASELINE_WINDOW = (-0.100, -0.010)  # s

S50_LOWER_BOUND = 10.0
S50_UPPER_BOUND = 100.0   # capped at 100 so S50 can't exceed stimulator limits


def boltzmann(x, vmin, vmax, s50, k):
    """p(1) + (p(2) - p(1)) ./ (1 + exp((p(3) - x) ./ p(4)))"""
    return vmin + (vmax - vmin) / (1.0 + np.exp((s50 - x) / k))


def fit_io_curve(x_clean: np.ndarray, y_clean: np.ndarray) -> Dict[str, Any]:
    """
    Fit on the MEAN MEP at each unique intensity (as in the MATLAB code).
    Returns a dict with success flag, parameters, and the unique-intensity means.
    """
    x_clean = np.asarray(x_clean, dtype=float)
    y_clean = np.asarray(y_clean, dtype=float)

    unique_x, idx = np.unique(x_clean, return_inverse=True)
    mean_y = np.bincount(idx, weights=y_clean) / np.bincount(idx)

    result: Dict[str, Any] = {
        "success": False, "error": None,
        "mep_min": np.nan, "mep_max": np.nan, "s50": np.nan, "slope_k": np.nan,
        "unique_x": unique_x, "mean_y": mean_y,
    }

    if len(unique_x) < 4:
        result["error"] = f"need at least 4 distinct intensities with valid trials, have {len(unique_x)}"
        return result

    obs_max = float(np.max(mean_y))
    obs_min = float(np.min(mean_y))
    if obs_max <= 0:
        result["error"] = "mean MEP is not positive at any intensity"
        return result

    # Initial guess [min, max, S50, k] and bounds, exactly as in the MATLAB code
    init = np.array([obs_min, obs_max, 50.0, 5.0])
    lb = np.array([0.0,          obs_max * 0.8, S50_LOWER_BOUND, 0.1])
    ub = np.array([obs_max / 2,  obs_max * 5.0, S50_UPPER_BOUND, 50.0])
    # lsqcurvefit tolerates a start point outside the bounds; scipy does not -> clip it in.
    init = np.clip(init, lb, ub)

    try:
        beta, _ = curve_fit(boltzmann, unique_x, mean_y, p0=init, bounds=(lb, ub), method="trf", maxfev=10000)
    except (RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
        return result

    result.update(success=True, mep_min=float(beta[0]), mep_max=float(beta[1]),
                  s50=float(beta[2]), slope_k=float(beta[3]))
    return result


# ---------------------------------------------------------------------------
# Trials CSV -> analysis
# ---------------------------------------------------------------------------
def load_trials(path: str) -> Dict[str, np.ndarray]:
    def to_float(v):
        return float(v) if v not in ("", None) else np.nan

    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "trial": np.array([int(r["trial"]) for r in rows]),
        "intensity": np.array([to_float(r["intensity_pct"]) for r in rows]),
        "mep": np.array([to_float(r["mep_ptp_uv"]) for r in rows]),
        "baseline_rms": np.array([to_float(r["baseline_rms_uv"]) for r in rows]),
    }


def analyse(trials_csv: str, out_dir: str, subject_label: str,
            cutoff: float = DEFAULT_BASELINE_RMS_CUTOFF,
            mep_window: Tuple[float, float] = DEFAULT_MEP_WINDOW,
            baseline_window: Tuple[float, float] = DEFAULT_BASELINE_WINDOW) -> Dict[str, Any]:
    """Apply the baseline-RMS QC, fit the curve, save figure (png + pdf) and a result JSON."""
    t = load_trials(trials_csv)
    x_raw, y_raw, rms = t["intensity"], t["mep"], t["baseline_rms"]

    # Rejected: baseline too noisy, or the MEP/baseline could not be measured at all
    unmeasurable = np.isnan(y_raw) | np.isnan(rms)
    is_clean = (~unmeasurable) & (rms <= cutoff)
    x_clean, y_clean = x_raw[is_clean], y_raw[is_clean]

    fit = fit_io_curve(x_clean, y_clean) if len(x_clean) else {
        "success": False, "error": "no valid trials after QC", "mep_min": np.nan, "mep_max": np.nan,
        "s50": np.nan, "slope_k": np.nan, "unique_x": np.array([]), "mean_y": np.array([])}

    print("\n--- TMS IO Curve Parameters (Constrained) ---")
    print(f"Trials: {len(x_raw)} total, {int((~is_clean).sum())} rejected (baseline RMS > {cutoff:g} uV or unmeasurable)")
    if fit["success"]:
        print(f"MEP_min: {fit['mep_min']:.2f} uV")
        print(f"MEP_max: {fit['mep_max']:.2f} uV (Estimated Saturation)")
        print(f"S50:     {fit['s50']:.2f} % MSO")
        print(f"Slope k: {fit['slope_k']:.2f}")
    else:
        print(f"Fitting error: {fit['error']}")
    print("----------------------------------------------")

    at_bound = bool(fit["success"] and (
        abs(fit["s50"] - S50_LOWER_BOUND) < 0.5 or abs(fit["s50"] - S50_UPPER_BOUND) < 0.5))
    if at_bound:
        print(f"WARNING: S50 sits on its bound ({S50_LOWER_BOUND:g} or {S50_UPPER_BOUND:g}). "
              f"The fit is not trustworthy; inspect the figure before using it.")

    fig_stem = _plot(out_dir, subject_label, x_raw, y_raw, rms, is_clean, cutoff, fit, mep_window, baseline_window)

    summary = {
        "subject": subject_label,
        "fit_success": bool(fit["success"]),
        "error": fit["error"],
        "s50_pct_mso": None if not fit["success"] else fit["s50"],
        "s50_rounded_pct_mso": None if not fit["success"] else int(round(fit["s50"])),
        "mep_min_uv": None if not fit["success"] else fit["mep_min"],
        "mep_max_uv": None if not fit["success"] else fit["mep_max"],
        "slope_k": None if not fit["success"] else fit["slope_k"],
        "s50_at_bound": at_bound,
        "n_trials": int(len(x_raw)),
        "n_rejected": int((~is_clean).sum()),
        "baseline_rms_cutoff_uv": cutoff,
        "mep_window_s": list(mep_window),
        "baseline_window_s": list(baseline_window),
        "trials_csv": os.path.abspath(trials_csv),
        "figure": os.path.abspath(fig_stem + ".png") if fig_stem else None,
        "analysed_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{subject_label}_IO_result.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Result saved to {os.path.abspath(json_path)}")
    return summary


def _plot(out_dir, subject_label, x_raw, y_raw, rms, is_clean, cutoff, fit, mep_window, baseline_window) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), facecolor="w")

    # Subplot 1: QC histogram of baseline RMS
    finite_rms = rms[~np.isnan(rms)]
    if len(finite_rms):
        ax1.hist(finite_rms, bins=20, color=(0.7, 0.7, 0.7), edgecolor="k")
    ax1.axvline(cutoff, color="r", linestyle="--", linewidth=2)
    ax1.text(cutoff, ax1.get_ylim()[1], " Threshold", color="r", va="top")
    ax1.set_xlabel("Baseline RMS (\u00b5V)")
    ax1.set_ylabel("Trial Count")
    ax1.set_title(f"Quality Control: Baseline [{baseline_window[0] * 1000:g},{baseline_window[1] * 1000:g}] ms")
    ax1.grid(True)
    ax1.spines[["top", "right"]].set_visible(False)

    # Subplot 2: IO curve + fit
    ax2.scatter(x_raw[is_clean], y_raw[is_clean], s=20, color=(0.6, 0.6, 0.6), alpha=0.4, label="Trials")
    rejected = ~is_clean & ~np.isnan(y_raw)
    if rejected.any():
        ax2.plot(x_raw[rejected], y_raw[rejected], "rx", markersize=5, label="Rejected")
    if len(fit["unique_x"]):
        ax2.plot(fit["unique_x"], fit["mean_y"], "ko", markersize=8, markerfacecolor="w",
                 markeredgewidth=1.5, label="Mean")
    if fit["success"]:
        x_fit = np.linspace(0, 100, 150)
        ax2.plot(x_fit, boltzmann(x_fit, fit["mep_min"], fit["mep_max"], fit["s50"], fit["slope_k"]),
                 color=(0.85, 0.33, 0.1), linewidth=3, label="Fit")
        y_mid = (fit["mep_min"] + fit["mep_max"]) / 2
        ax2.plot(fit["s50"], y_mid, "bs", markersize=10,
                 label=f"S50: {fit['s50']:.1f}%, Max: {fit['mep_max']:.0f}\u00b5V")
        ax2.plot([fit["s50"], fit["s50"]], [0, y_mid], "b--")
        ax2.plot([0, fit["s50"]], [y_mid, y_mid], "b:")
    else:
        ax2.set_title("FIT FAILED: " + str(fit["error"]), color="r")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Intensity (% MSO)")
    ax2.set_ylabel("MEP Amplitude (\u00b5V)")
    if fit["success"]:
        ax2.set_title(f"M1 IO Curve: MEP window [{mep_window[0] * 1000:g},{mep_window[1] * 1000:g}] ms")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    s50_tag = "FAILED" if not fit["success"] else f"{fit['s50']:.1f}".replace(".", "_")
    stem = os.path.join(out_dir, f"{subject_label}_IO_S50_{s50_tag}")
    fig.savefig(stem + ".png", dpi=150)
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    print(f"Figure saved to {os.path.abspath(stem)}.png / .pdf")
    return stem


def main():
    ap = argparse.ArgumentParser(description="Fit a Boltzmann IO curve to a saved trials CSV.")
    ap.add_argument("trials_csv")
    ap.add_argument("--subject", default="subject")
    ap.add_argument("--outdir", default=None, help="default: same folder as the CSV")
    ap.add_argument("--cutoff", type=float, default=DEFAULT_BASELINE_RMS_CUTOFF,
                    help="baseline RMS rejection threshold in uV")
    args = ap.parse_args()
    analyse(args.trials_csv, args.outdir or os.path.dirname(os.path.abspath(args.trials_csv)),
            args.subject, cutoff=args.cutoff)


if __name__ == "__main__":
    main()
