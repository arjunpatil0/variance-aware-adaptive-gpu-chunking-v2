#!/usr/bin/env python3
"""
build_dashboard.py — Generates a self-contained HTML summary page.
"""

import os
import json
import base64
import argparse
import sys

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_dashboard(results_dir):
    experiments = ["A", "B", "C", "D"]
    html = [
        "<html>",
        "<head>",
        "<title>Variance Adaptive v2 - Results Dashboard</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 20px; color: #333; }",
        "h1 { border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; }",
        "h2 { margin-top: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }",
        ".exp-section { background: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        ".grid { display: flex; flex-wrap: wrap; gap: 20px; }",
        ".card { background: white; padding: 15px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); flex: 1; min-width: 400px; }",
        "img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }",
        "table { border-collapse: collapse; width: 100%; margin-top: 15px; }",
        "th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }",
        "th { background-color: #f2f2f2; }",
        ".sig { color: #2e7d32; font-weight: bold; }",
        ".not-sig { color: #c62828; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Variance Adaptive v2: Occupancy Hypothesis Test</h1>",
        "<p>This dashboard summarizes the results across all four synthetic workloads. It focuses on the primary T_total metrics, the adaptive vs inverted-adaptive mechanism test, and occupancy profiles.</p>"
    ]

    for exp in experiments:
        # Find directory
        exp_dir = None
        for name in os.listdir(results_dir):
            if name.startswith(exp + "_") or name == exp:
                exp_dir = os.path.join(results_dir, name)
                break
        
        if not exp_dir or not os.path.exists(exp_dir):
            continue

        html.append(f"<div class='exp-section' id='exp-{exp}'>")
        html.append(f"<h2>Experiment {exp}</h2>")
        
        # Load stats
        stats_path = os.path.join(exp_dir, "statistical_results.json")
        stats = {}
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)

        html.append("<div class='grid'>")

        # Visuals
        plots_dir = os.path.join(exp_dir, "plots")
        
        # 1. Total runtime bars
        b64_bars = get_base64_image(os.path.join(plots_dir, "05_total_runtime_bars.png"))
        if b64_bars:
            html.append("<div class='card'>")
            html.append("<h3>Mean T_total (n=20)</h3>")
            html.append(f"<img src='data:image/png;base64,{b64_bars}' />")
            html.append("</div>")

        # 2. Chunk size trace (shows inverted)
        b64_trace = get_base64_image(os.path.join(plots_dir, "03_chunk_size_trace.png"))
        if b64_trace:
            html.append("<div class='card'>")
            html.append("<h3>Chunk Size Policies (Rep 0)</h3>")
            html.append(f"<img src='data:image/png;base64,{b64_trace}' />")
            html.append("</div>")

        html.append("</div>") # End visual grid

        # Statistical comparisons
        html.append("<div class='card' style='margin-top: 20px'>")
        html.append("<h3>Statistical Pairwise Comparisons (Wilcoxon Signed-Rank)</h3>")
        html.append("<table>")
        html.append("<tr><th>Comparison</th><th>Mean Diff</th><th>Effect Size</th><th>p-value</th><th>Significance</th></tr>")
        
        for key, name in [
            ("primary_test_matched_vs_adaptive", "STATIC-MATCHED vs ADAPTIVE (Primary)"),
            ("secondary_test_large_vs_adaptive", "STATIC-LARGE vs ADAPTIVE (Secondary)"),
            ("tertiary_test_inverted_vs_adaptive", "INVERTED vs ADAPTIVE (Mechanism Test)")
        ]:
            if key in stats:
                t = stats[key]
                sig_class = "sig" if t.get("is_significant") else "not-sig"
                p_val = t.get("p_value", 1.0)
                html.append("<tr>")
                html.append(f"<td>{name}</td>")
                html.append(f"<td>{t.get('mean_diff_ms', 0):.2f} ms</td>")
                html.append(f"<td>{t.get('relative_pct', 0):.2f}%</td>")
                html.append(f"<td>{p_val:.4f}</td>")
                html.append(f"<td class='{sig_class}'>{'Significant' if t.get('is_significant') else 'Not Significant'}</td>")
                html.append("</tr>")
        
        html.append("</table>")
        html.append("</div>")
        
        # Occupancy Table if exists
        occ_csv = os.path.join(exp_dir, "occupancy_profile.csv")
        if os.path.exists(occ_csv):
            import csv
            html.append("<div class='card' style='margin-top: 20px'>")
            html.append("<h3>Occupancy Profile (Nsight Compute)</h3>")
            html.append("<table>")
            html.append("<tr><th>Chunk Size</th><th>Achieved Occupancy (%)</th><th>Samples</th></tr>")
            with open(occ_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    html.append(f"<tr><td>{row['chunk_size']}</td><td>{row['achieved_occupancy_pct']}%</td><td>{row['num_samples']}</td></tr>")
            html.append("</table>")
            html.append("</div>")

        html.append("</div>") # End exp section

    html.append("</body></html>")

    out_path = os.path.join(results_dir, "dashboard.html")
    with open(out_path, "w") as f:
        f.write("\n".join(html))
    print(f"Dashboard generated at {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    build_dashboard(args.results_dir)

if __name__ == "__main__":
    main()
