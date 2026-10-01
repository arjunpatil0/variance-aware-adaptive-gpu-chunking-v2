#pragma once
// statistical_analysis.h — Statistical computations for the experiment.
//
// PRIMARY comparison: STATIC-MATCHED vs ADAPTIVE (paired Wilcoxon).
// SECONDARY comparison: STATIC-LARGE vs ADAPTIVE (paired Wilcoxon, weaker).
//
// Statistical test chosen BEFORE experiments: Wilcoxon signed-rank test.
// WHY Wilcoxon rather than paired t-test: paired runtime differences may not
// satisfy normality assumptions for n=20 across repeated GPU experiments;
// Wilcoxon is more robust.  scipy.stats.wilcoxon in analyze_v2.py
// independently validates the C++ implementation.
//
// α = 0.05; minimum practical effect = 5% lower mean T_total for ADAPTIVE.
// Both conditions must hold to call a result an "improvement."
//
// Detection lag:
//   observation_lag_tasks: tasks from regime boundary until window variance
//     enters the new regime's territory (bounded by WINDOW_SIZE).
//   action_lag_tasks: tasks from regime boundary until chunk size moves
//     by at least response_fraction toward the new regime target.
//   action_lag >= observation_lag in general (in-flight chunk adds delay).

#pragma once
#include "config.h"
#include "scheduler.h"
#include "workload_generator.h"
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Per-arm runtime summary
// ---------------------------------------------------------------------------
struct ArmStats {
    std::string arm_name;
    int    n;
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
};

// ---------------------------------------------------------------------------
// Wilcoxon signed-rank test result (two-tailed, paired)
// ---------------------------------------------------------------------------
struct WilcoxonResult {
    std::string arm_a;          // e.g. "adaptive"
    std::string arm_b;          // e.g. "static_matched"
    int    n;                   // pairs used
    double W_stat;              // test statistic
    double Z_score;
    double p_value;
    double effect_size_r;       // r = Z / sqrt(n), ranges [-1,1]
    double mean_diff_ms;        // mean(a - b)  negative = a is faster
    double std_diff_ms;
    double relative_pct;        // 100*(mean_b - mean_a)/mean_b  positive = a faster
    double ci_lower_95;         // 95% CI lower bound on mean(a-b)
    double ci_upper_95;
    bool   is_significant;      // p < alpha
    bool   meets_effect_threshold; // |relative_pct| >= 5%
    bool   is_improvement;      // both conditions AND a is faster
    std::string interpretation;  // human-readable summary
    std::string caution;         // set when is_improvement is false despite appearing better
};

// ---------------------------------------------------------------------------
// Detection lag per regime transition
// ---------------------------------------------------------------------------
struct DetectionLagResult {
    int    transition_index;        // 0-based transition number
    int    from_phase;
    int    to_phase;
    int    boundary_task;           // ground-truth regime boundary
    bool   new_regime_high_variance; // whether the new phase is high-variance
    int    observation_lag_tasks;   // tasks until window σ² crosses into new regime territory
                                    // -1 if not detected within the phase
    int    action_lag_tasks;        // tasks until chunk size moves by response_fraction
                                    // -1 if no response detected
    int    decision_index_responded; // -1 if no response
    bool   responded_within_window; // observation_lag_tasks <= WINDOW_SIZE
};

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

// Compute per-arm summary statistics from a vector of T_total observations.
ArmStats compute_arm_stats(const std::string& name,
                            const std::vector<double>& totals_ms);

// Wilcoxon signed-rank test on paired observations.
// a[i] and b[i] must correspond to the same workload repetition.
WilcoxonResult wilcoxon_signed_rank(
    const std::string& arm_a_name,
    const std::string& arm_b_name,
    const std::vector<double>& a_ms,
    const std::vector<double>& b_ms,
    double alpha,
    double min_effect_fraction);

// Compute detection lags for all phase transitions from the adaptive trace.
// Called once per adaptive run, after the run completes.
// Annotates the relevant SchedulerDecision entries in-place.
std::vector<DetectionLagResult> compute_detection_lags(
    std::vector<SchedulerDecision>& trace,  // modified in-place to set lag fields
    const std::vector<PhaseInfo>&   phases,
    const ExperimentConfig&         cfg);

// Determine which phase a given task_index belongs to.
int phase_of_task(int task_index, const std::vector<PhaseInfo>& phases);

// Returns true if phase p is considered "high variance" for the purpose of
// detection-lag direction.  Uses the actual_variance from PhaseInfo.
// High-variance phases are expected to drive the scheduler toward MIN_CHUNK.
bool is_high_variance_phase(int phase_id, const std::vector<PhaseInfo>& phases,
                             double var_high_thresh);
