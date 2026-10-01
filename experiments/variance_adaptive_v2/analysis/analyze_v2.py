#!/usr/bin/env python3
"""
analyze_v2.py — Statistical analysis and visualization for variance_adaptive_v2.

Purpose:
  1. Cross-validate the C++ Wilcoxon test using scipy.stats.wilcoxon.
  2. Produce all visualization plots specified in Section 13 of the spec.

Usage:
  python analysis/analyze_v2.py --experiment C [--results-dir results]

Outputs (written to results/<experiment>/plots/):
  01_task_cost_trace.png         Task index vs task cost, colored by phase
  02_window_variance_trace.png   Task index vs sliding-window variance
  03_chunk_size_trace.png        Decision index vs chunk size (all 3 arms)
  04_chunk_timeline.png          Chunk execution timeline (Gantt-style)
  05_total_runtime_bars.png      T_total mean ± std for all 3 arms
  06_scheduling_decisions.png    Regime boundaries + chunk-size decisions overlay
  07_detection_lag.png           Observation lag vs action lag per transition
  python_statistical_results.json  Scipy Wilcoxon cross-check results

Requirements:
  pip install pandas matplotlib scipy numpy
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Analyze variance_adaptive_v2 results")
    p.add_argument("--experiment", default="C",
                   help="Experiment preset label (A, B, C, D) — used to locate results directory")
    p.add_argument("--results-dir", default="results",
                   help="Base results directory (default: results)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance threshold (default: 0.05)")
    p.add_argument("--min-effect", type=float, default=0.05,
                   help="Minimum practical effect fraction (default: 0.05 = 5%%)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(exp_dir: str):
    def req(name):
        path = os.path.join(exp_dir, name)
        if not os.path.exists(path):
            sys.exit(f"ERROR: Required file not found: {path}")
        return path

    raw_df      = pd.read_csv(req("raw_results.csv"))
    trace_df    = pd.read_csv(req("scheduler_trace.csv"))

    workload_path = os.path.join(exp_dir, "workload.csv")
    workload_df   = pd.read_csv(workload_path) if os.path.exists(workload_path) else None

    with open(req("config.json")) as f:
        config = json.load(f)

    stat_path = os.path.join(exp_dir, "statistical_results.json")
    cpp_stats = json.load(open(stat_path)) if os.path.exists(stat_path) else {}

    lag_path = os.path.join(exp_dir, "detection_lag_results.json")
    lag_data = json.load(open(lag_path)) if os.path.exists(lag_path) else {}

    return raw_df, trace_df, workload_df, config, cpp_stats, lag_data


# ---------------------------------------------------------------------------
# Scipy Wilcoxon cross-check
# ---------------------------------------------------------------------------
def wilcoxon_crosscheck(raw_df: pd.DataFrame, alpha: float, min_effect: float,
                         config: dict) -> dict:
    """Independently validate the C++ Wilcoxon results using scipy."""
    results = {}

    comparisons = [
        ("primary",   "t_total_adaptive_ms", "t_total_static_matched_ms",
         "adaptive vs static_matched"),
        ("secondary", "t_total_adaptive_ms", "t_total_static_large_ms",
         "adaptive vs static_large"),
        ("tertiary",  "t_total_adaptive_ms", "t_total_inverted_adaptive_ms",
         "adaptive vs inverted_adaptive"),
    ]

    for key, col_a, col_b, label in comparisons:
        a = raw_df[col_a].values
        b = raw_df[col_b].values
        diffs = a - b

        try:
            stat, p = stats.wilcoxon(a, b, alternative="two-sided", correction=True)
        except Exception as e:
            warnings.warn(f"scipy.stats.wilcoxon failed for {label}: {e}")
            stat, p = np.nan, np.nan

        mean_b = np.mean(b)
        rel_pct = 100.0 * (-np.mean(diffs)) / mean_b if mean_b != 0 else 0.0
        n = len(a)
        # Effect size r = Z / sqrt(n) — approximate from W statistic
        # scipy returns the test statistic, not Z; compute r from p
        Z = stats.norm.ppf(1 - p / 2) if p < 1.0 else 0.0
        r = Z / np.sqrt(n) if n > 0 else 0.0

        is_sig    = p < alpha
        meets_eff = rel_pct >= min_effect * 100.0
        is_impr   = is_sig and meets_eff

        results[key] = {
            "label": label,
            "n": int(n),
            "W_stat": float(stat) if not np.isnan(stat) else None,
            "p_value": float(p),
            "effect_size_r": float(r),
            "mean_diff_ms": float(np.mean(diffs)),
            "relative_pct": float(rel_pct),
            "is_significant": bool(is_sig),
            "meets_effect_threshold": bool(meets_eff),
            "is_improvement": bool(is_impr),
            "note": ("IMPROVEMENT" if is_impr
                     else "significant but small effect" if is_sig
                     else "large effect but not significant" if meets_eff
                     else "no meaningful difference"),
            "validator": "scipy.stats.wilcoxon (two-sided, continuity correction)"
        }

    return results


# ---------------------------------------------------------------------------
# Phase color map
# ---------------------------------------------------------------------------
PHASE_COLORS = ["#4CAF50", "#F44336", "#2196F3", "#FF9800"]
PHASE_LABELS = ["Phase 0", "Phase 1", "Phase 2", "Phase 3"]


def phase_color(pid):
    if pid < 0 or pid >= len(PHASE_COLORS):
        return "#888888"
    return PHASE_COLORS[pid]


# ---------------------------------------------------------------------------
# Plot 1: Task cost trace
# ---------------------------------------------------------------------------
def plot_task_cost_trace(workload_df, config, out_dir):
    if workload_df is None:
        print("  Skipping task cost trace (workload.csv not found)")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    phases = config.get("phases", [])

    # Plot as scatter with phase coloring (sample for performance)
    n = len(workload_df)
    step = max(1, n // 5000)
    sub = workload_df.iloc[::step]

    for pid in sub["phase_id"].unique():
        mask = sub["phase_id"] == pid
        ax.scatter(sub.loc[mask, "task_index"], sub.loc[mask, "task_cost"],
                   s=0.5, c=phase_color(pid), label=f"Phase {pid}", rasterized=True)

    # Phase boundary vertical lines
    for p in phases:
        ax.axvline(p["start_task"], color="black", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.text(p["start_task"] + n * 0.002, ax.get_ylim()[1] * 0.9,
                p["name"], fontsize=7, rotation=90, va="top")

    ax.set_xlabel("Task Index")
    ax.set_ylabel("Task Cost (raw iterations)")
    ax.set_title(f"Workload Task Costs — {config.get('experiment_name', '')}")
    ax.legend(markerscale=8, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_task_cost_trace.png"), dpi=150)
    plt.close(fig)
    print("  Saved 01_task_cost_trace.png")


# ---------------------------------------------------------------------------
# Plot 2: Sliding-window variance trace (adaptive arm only)
# ---------------------------------------------------------------------------
def plot_window_variance(trace_df, config, out_dir):
    adap = trace_df[(trace_df["arm"] == "adaptive") & (trace_df["rep"] == 0)].copy()
    if adap.empty:
        print("  Skipping window variance (no adaptive trace for rep 0)")
        return

    phases = config.get("phases", [])
    total  = config.get("total_tasks", 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(adap["task_start"], adap["window_variance"], linewidth=0.8, color="#1565C0")
    ax1.axhline(config.get("var_high_thresh", 0), color="red",
                linestyle="--", linewidth=1, label="var_high_thresh")
    ax1.axhline(config.get("var_low_thresh", 0),  color="green",
                linestyle="--", linewidth=1, label="var_low_thresh")
    for p in phases:
        ax1.axvline(p["start_task"], color="gray", linestyle=":", linewidth=0.7)
    ax1.set_ylabel("Window Variance (σ²)")
    ax1.set_title("Sliding-Window Task-Cost Variance (Adaptive Arm, Rep 0)")
    ax1.legend(fontsize=8)

    ax2.step(adap["task_start"], adap["chunk_size"],
             where="post", linewidth=1, color="#6A1B9A")
    ax2.axhline(config.get("max_chunk", 10000), color="orange",
                linestyle="--", linewidth=0.8, label="MAX_CHUNK")
    ax2.axhline(config.get("min_chunk", 1000),  color="brown",
                linestyle="--", linewidth=0.8, label="MIN_CHUNK")
    for p in phases:
        ax2.axvline(p["start_task"], color="gray", linestyle=":", linewidth=0.7)
    ax2.set_xlabel("Task Index (at scheduling decision)")
    ax2.set_ylabel("Chunk Size")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_window_variance_trace.png"), dpi=150)
    plt.close(fig)
    print("  Saved 02_window_variance_trace.png")


# ---------------------------------------------------------------------------
# Plot 3: Chunk size trace — all three arms, rep 0
# ---------------------------------------------------------------------------
def plot_chunk_size_trace(trace_df, config, out_dir):
    fig, ax = plt.subplots(figsize=(14, 4))
    arm_styles = {
        "adaptive":          ("#1565C0", "-",  0.9),
        "inverted_adaptive": ("#9C27B0", "-",  0.9),
        "static_large":      ("#F44336", "--", 0.8),
        "static_matched":    ("#4CAF50", "-.", 0.8),
    }
    rep0 = trace_df[trace_df["rep"] == 0]
    for arm, (color, ls, lw) in arm_styles.items():
        sub = rep0[rep0["arm"] == arm]
        if sub.empty:
            continue
        ax.step(sub["decision_idx"], sub["chunk_size"],
                where="post", color=color, linestyle=ls,
                linewidth=lw, label=arm)

    phases = config.get("phases", [])
    total  = config.get("total_tasks", 1)
    for p in phases:
        ax.axvline(p["start_task"] / max(total / max(len(rep0), 1), 1),
                   color="gray", linestyle=":", linewidth=0.6)

    ax.set_xlabel("Scheduling Decision Index")
    ax.set_ylabel("Chunk Size")
    ax.set_title(f"Chunk Size per Scheduling Decision — {config.get('experiment_name', '')} (Rep 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_chunk_size_trace.png"), dpi=150)
    plt.close(fig)
    print("  Saved 03_chunk_size_trace.png")


# ---------------------------------------------------------------------------
# Plot 4: Chunk execution timeline (Gantt-style), rep 0, adaptive arm
# ---------------------------------------------------------------------------
def plot_chunk_timeline(trace_df, config, out_dir):
    adap = trace_df[(trace_df["arm"] == "adaptive") & (trace_df["rep"] == 0)].copy()
    if adap.empty:
        return

    # Build a time-ordered Gantt using t_total_chunk_ms (cumulative start)
    adap = adap.sort_values("task_start").reset_index(drop=True)
    adap["t_start_cum"] = adap["t_total_chunk_ms"].cumsum().shift(1, fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 3))
    for _, row in adap.iterrows():
        ax.barh(0, row["t_total_chunk_ms"], left=row["t_start_cum"],
                height=0.5, color=phase_color(int(row["phase_id_at_decision"])),
                edgecolor="none", alpha=0.8)

    handles = [mpatches.Patch(color=phase_color(i), label=f"Phase {i}")
               for i in range(4)]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlabel("Cumulative Wall-clock Time (ms)")
    ax.set_title("Chunk Execution Timeline — Adaptive Arm (Rep 0)")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_chunk_timeline.png"), dpi=150)
    plt.close(fig)
    print("  Saved 04_chunk_timeline.png")


# ---------------------------------------------------------------------------
# Plot 5: Total runtime bar chart — all 3 arms, mean ± std
# ---------------------------------------------------------------------------
def plot_total_runtime_bars(raw_df, config, out_dir):
    arms = ["adaptive", "static_matched", "static_large", "inverted_adaptive"]
    cols = ["t_total_adaptive_ms", "t_total_static_matched_ms", "t_total_static_large_ms", "t_total_inverted_adaptive_ms"]
    labels = ["ADAPTIVE", "STATIC-MATCHED\n(primary comparison)", "STATIC-LARGE\n(secondary)", "INVERTED-ADAPTIVE\n(mechanism test)"]
    colors = ["#1565C0", "#4CAF50", "#F44336", "#9C27B0"]

    means = [raw_df[c].mean() for c in cols]
    stds  = [raw_df[c].std()  for c in cols]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(arms))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + max(means) * 0.01,
                f"{m:.1f}±{s:.1f}ms", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total End-to-End Execution Time (ms)")
    ax.set_title(f"Primary Metric: T_total — {config.get('experiment_name', '')}\n"
                 f"n={len(raw_df)} repetitions | PRIMARY: MATCHED vs ADAPTIVE")
    ax.annotate("← Primary comparison →", xy=(0.38, -0.18),
                xycoords="axes fraction", ha="center", fontsize=8,
                arrowprops=None, color="green")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_total_runtime_bars.png"), dpi=150)
    plt.close(fig)
    print("  Saved 05_total_runtime_bars.png")


# ---------------------------------------------------------------------------
# Plot 6: Scheduling decisions overlay with regime boundaries
# ---------------------------------------------------------------------------
def plot_scheduling_decisions(trace_df, config, out_dir):
    phases = config.get("phases", [])
    adap   = trace_df[(trace_df["arm"] == "adaptive") & (trace_df["rep"] == 0)].copy()
    if adap.empty:
        return

    inv = trace_df[(trace_df["arm"] == "inverted_adaptive") & (trace_df["rep"] == 0)].copy()

    fig, ax = plt.subplots(figsize=(14, 4))

    # Background phase bands
    total = config.get("total_tasks", 1)
    for i, p in enumerate(phases):
        ax.axvspan(p["start_task"], p["end_task"], alpha=0.08,
                   color=PHASE_COLORS[i % len(PHASE_COLORS)])
        ax.text((p["start_task"] + p["end_task"]) / 2,
                config.get("max_chunk", 10000) * 1.02,
                p["name"], ha="center", fontsize=7, color=PHASE_COLORS[i % len(PHASE_COLORS)])

    # Chunk-size decisions
    ax.step(adap["task_start"], adap["chunk_size"],
            where="post", color="#1565C0", linewidth=1.2, label="Adaptive chunk size")
    
    if not inv.empty:
        ax.step(inv["task_start"], inv["chunk_size"],
                where="post", color="#9C27B0", linewidth=1.2, linestyle="--", label="Inverted-Adaptive chunk size")

    # Phase boundaries
    for p in phases[1:]:
        ax.axvline(p["start_task"], color="black", linestyle="--",
                   linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Task Index")
    ax.set_ylabel("Chunk Size")
    ax.set_title(f"Scheduling Decisions with Regime Boundaries — {config.get('experiment_name', '')} (Rep 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_scheduling_decisions.png"), dpi=150)
    plt.close(fig)
    print("  Saved 06_scheduling_decisions.png")


# ---------------------------------------------------------------------------
# Plot 7: Detection lag per transition
# ---------------------------------------------------------------------------
def plot_detection_lag(lag_data, config, out_dir):
    transitions = lag_data.get("transitions", [])
    if not transitions:
        print("  Skipping detection lag plot (no data)")
        return

    labels = [f"P{t['from_phase']}→P{t['to_phase']}" for t in transitions]
    obs    = [t.get("mean_observation_lag_tasks", 0) or 0 for t in transitions]
    act    = [t.get("mean_action_lag_tasks", 0)      or 0 for t in transitions]
    window_size = config.get("window_size", 20)

    x = np.arange(len(transitions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, obs, width, label="observation_lag_tasks",
           color="#1565C0", alpha=0.8)
    ax.bar(x + width/2, act, width, label="action_lag_tasks",
           color="#F44336", alpha=0.8)
    ax.axhline(window_size, color="green", linestyle="--", linewidth=1,
               label=f"WINDOW_SIZE={window_size} (obs_lag bound)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tasks from regime boundary")
    ax.set_title("Causal Detection Lag per Regime Transition\n"
                 "(observation_lag ≤ WINDOW_SIZE; action_lag ≥ observation_lag)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_detection_lag.png"), dpi=150)
    plt.close(fig)
    print("  Saved 07_detection_lag.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Locate the experiment directory — search for a folder whose name
    # contains the experiment letter.
    exp_dir = None
    base = args.results_dir
    if os.path.isdir(base):
        for name in os.listdir(base):
            if name.startswith(args.experiment + "_") or name == args.experiment:
                exp_dir = os.path.join(base, name)
                break
    # Fallback: try direct path
    if exp_dir is None:
        exp_dir = os.path.join(base, args.experiment)
    if not os.path.isdir(exp_dir):
        sys.exit(f"ERROR: Cannot find experiment directory for '{args.experiment}' under '{base}'")

    print(f"Analyzing: {exp_dir}")

    raw_df, trace_df, workload_df, config, cpp_stats, lag_data = load_data(exp_dir)

    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ----- Scipy cross-check -----
    print("\n[1] Running scipy.stats.wilcoxon cross-check...")
    crosscheck = wilcoxon_crosscheck(raw_df, args.alpha, args.min_effect, config)

    for key, res in crosscheck.items():
        print(f"  {key}: p={res['p_value']:.4f}  effect={res['relative_pct']:.2f}%  "
              f"-> {res['note']}")

    # Compare against C++ values
    if cpp_stats:
        for key, c_key in [("primary",   "primary_test_matched_vs_adaptive"),
                           ("secondary", "secondary_test_large_vs_adaptive"),
                           ("tertiary",  "tertiary_test_inverted_vs_adaptive")]:
            if c_key in cpp_stats:
                cpp_p   = cpp_stats[c_key]["p_value"]
                scipy_p = crosscheck[key]["p_value"]
                delta   = abs(cpp_p - scipy_p)
                status  = "OK" if delta < 0.02 else "MISMATCH"
                print(f"  [{status}] {key}: C++ p={cpp_p:.4f}  scipy p={scipy_p:.4f}  "
                      f"delta={delta:.4f}")

    # Save cross-check results
    out_json = os.path.join(exp_dir, "python_statistical_results.json")
    with open(out_json, "w") as f:
        json.dump(crosscheck, f, indent=2)
    print(f"  Saved python_statistical_results.json")

    # ----- Plots -----
    print("\n[2] Generating plots...")
    plot_task_cost_trace(workload_df, config, plots_dir)
    plot_window_variance(trace_df, config, plots_dir)
    plot_chunk_size_trace(trace_df, config, plots_dir)
    plot_chunk_timeline(trace_df, config, plots_dir)
    plot_total_runtime_bars(raw_df, config, plots_dir)
    plot_scheduling_decisions(trace_df, config, plots_dir)
    plot_detection_lag(lag_data, config, plots_dir)

    print(f"\nAll outputs written to: {exp_dir}/")


if __name__ == "__main__":
    main()
