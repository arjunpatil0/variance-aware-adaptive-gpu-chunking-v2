#!/usr/bin/env python3
"""
profile_occupancy.py — Runs Nsight Compute to profile warp occupancy.

Checks if `ncu` is available. If so, runs the profiling executable across
a spread of chunk sizes, extracts achieved occupancy, and saves to CSV.
"""

import argparse
import csv
import os
import subprocess
import sys

def check_ncu():
    try:
        res = subprocess.run(["ncu", "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    return False

def parse_ncu_csv(csv_path):
    # ncu --csv output parsing
    # The metric we want is sm__warps_active.avg.pct_of_peak_sustained_active
    # It might be in the 'Metric Value' column when 'Metric Name' matches.
    results = []
    if not os.path.exists(csv_path):
        return results

    with open(csv_path, 'r', newline='') as f:
        # Skip the initial lines until we find the header starting with '"ID","Process ID"'
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row: continue
            if row[0] == "ID":
                header = row
                break
            elif row[0] == "Index": # Sometimes it's Index
                header = row
                break
        
        if not header:
            return results

        # Find columns
        try:
            val_idx = header.index("Metric Value")
            name_idx = header.index("Metric Name")
        except ValueError:
            return results

        for row in reader:
            if len(row) <= max(val_idx, name_idx): continue
            if row[name_idx] == "sm__warps_active.avg.pct_of_peak_sustained_active":
                try:
                    results.append(float(row[val_idx].replace('%', '')))
                except ValueError:
                    pass
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment label (A,B,C,D)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    if not check_ncu():
        print("Nsight Compute (ncu) is not available on the system.")
        print("Skipping occupancy profiling. (Do not simulate this metric).")
        sys.exit(0)

    # Locate the experiment directory
    exp_dir = None
    base = args.results_dir
    if os.path.isdir(base):
        for name in os.listdir(base):
            if name.startswith(args.experiment + "_") or name == args.experiment:
                exp_dir = os.path.join(base, name)
                break
    if not exp_dir:
        exp_dir = os.path.join(base, args.experiment)
    
    os.makedirs(exp_dir, exist_ok=True)
    out_csv = os.path.join(exp_dir, "occupancy_profile.csv")

    print(f"Running occupancy profiling for experiment {args.experiment}...")

    # Run ncu with CSV output
    temp_ncu = os.path.join(exp_dir, "temp_ncu.csv")
    
    cmd = [
        "ncu", "--csv", "--page", "raw",
        "--metrics", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "-o", temp_ncu.replace(".csv", ""), # ncu appends .csv
        "-f", # force overwrite
        "variance_adaptive_v2_prof.exe",
        "--experiment", args.experiment,
        "--profile-occupancy"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        print(f"ncu failed with code {res.returncode}")
        print(res.stderr)
        sys.exit(1)

    # Parse output
    # ncu outputs to temp_ncu.csv
    ncu_file = temp_ncu
    if not os.path.exists(ncu_file):
        print("Failed to find ncu output file.")
        sys.exit(1)

    vals = parse_ncu_csv(ncu_file)
    if not vals:
        print("Warning: Could not extract metric from ncu output. The metric name might be different on this architecture.")
        # Try a fallback metric
        cmd_fallback = cmd.copy()
        cmd_fallback[5] = "sm__warps_active.avg.per_cycle_active"
        subprocess.run(cmd_fallback, capture_output=True)
        vals = parse_ncu_csv(ncu_file)

    # The executable runs chunks: 1000, 2000, 3000, 5000, 7500, 10000. 5 samples each.
    # Total 30 samples.
    spread = [1000, 2000, 3000, 5000, 7500, 10000]
    expected_samples = len(spread) * 5

    # Group results
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_size", "achieved_occupancy_pct", "num_samples"])
        
        idx = 0
        for chunk in spread:
            chunk_vals = vals[idx:idx+5]
            if chunk_vals:
                avg_occ = sum(chunk_vals) / len(chunk_vals)
                writer.writerow([chunk, f"{avg_occ:.2f}", len(chunk_vals)])
            idx += 5
            
    print(f"Occupancy profile saved to {out_csv}")
    
    if os.path.exists(temp_ncu):
        os.remove(temp_ncu)

if __name__ == "__main__":
    main()
