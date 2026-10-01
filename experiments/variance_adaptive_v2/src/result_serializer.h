#pragma once
// result_serializer.h — Writes all experiment outputs to JSON and CSV.
//
// WHY a dedicated serialiser: keeping I/O logic out of the experiment runner
// keeps the runner focused on orchestration and makes output formats easy to
// change without touching experiment logic.
//
// Output files written per experiment:
//   config.json              — full parameter snapshot
//   workload.csv             — task_index, task_cost, phase_id
//   scheduler_trace.csv      — per-chunk decisions for all arms / reps
//   raw_results.csv          — per-rep T_total for all 3 arms + metadata
//   statistical_results.json — means, stds, Wilcoxon, CI, effect sizes
//   detection_lag_results.json — per-transition observation+action lag
//
// Per-repetition data is APPENDED to the CSV files rather than held in
// memory, so large experiments do not exhaust RAM.

#include "config.h"
#include "workload_generator.h"
#include "scheduler.h"
#include "statistical_analysis.h"
#include <string>
#include <vector>
#include <fstream>

// ---------------------------------------------------------------------------
// ResultSerializer
// ---------------------------------------------------------------------------
class ResultSerializer {
public:
    // Creates the output directory and opens/initialises the CSV files.
    // out_dir is created if it does not exist.
    explicit ResultSerializer(const std::string& out_dir);
    ~ResultSerializer();

    // Write config.json (call once before the experiment runs).
    void write_config(const ExperimentConfig& cfg,
                      const std::vector<PhaseInfo>& phases);

    // Write workload.csv (call once per repetition 0; workload is the same
    // across all reps for a given experiment, so we only write rep 0).
    void write_workload(const std::vector<int>& task_costs,
                        const std::vector<PhaseInfo>& phases);

    // Append one chunk's row to scheduler_trace.csv.
    void append_trace_row(int rep,
                          const SchedulerDecision& d,
                          const std::string& arm_name);

    // Append one repetition's summary to raw_results.csv.
    void append_rep_result(int rep,
                           unsigned int workload_seed,
                           double t_total_adaptive_ms,
                           double t_total_static_large_ms,
                           double t_total_static_matched_ms,
                           double t_total_inverted_ms,
                           int mean_adaptive_chunk,
                           int n_chunks_adaptive,
                           int n_chunks_static_large,
                           int n_chunks_static_matched,
                           int n_chunks_inverted,
                           const std::string& arm_order);

    // Write statistical_results.json (call once after all reps complete).
    void write_statistical_results(
        const ArmStats& stats_adaptive,
        const ArmStats& stats_static_large,
        const ArmStats& stats_static_matched,
        const ArmStats& stats_inverted,
        const WilcoxonResult& primary_test,    // STATIC-MATCHED vs ADAPTIVE
        const WilcoxonResult& secondary_test,  // STATIC-LARGE   vs ADAPTIVE
        const WilcoxonResult& tertiary_test);  // INVERTED       vs ADAPTIVE

    // Write detection_lag_results.json (call once after all reps complete).
    void write_detection_lag_results(
        const std::vector<std::vector<DetectionLagResult>>& all_rep_lags,
        const std::vector<PhaseInfo>& phases);

    // Write calibration.json (separate from per-experiment output).
    static void write_calibration(const std::string& out_dir,
                                  int chosen_multiplier,
                                  double overhead_ms,
                                  double var_low_thresh,
                                  double var_high_thresh,
                                  const std::vector<std::pair<int,double>>& sweep);

    const std::string& out_dir() const { return out_dir_; }

private:
    std::string out_dir_;
    std::ofstream trace_csv_;
    std::ofstream raw_csv_;

    // Escapes a string for CSV (wraps in quotes if it contains commas/newlines).
    static std::string csv_str(const std::string& s);
    // Creates directory recursively (Windows-compatible).
    static void ensure_dir(const std::string& path);
};
