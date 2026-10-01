#pragma once
// config.h — All tunable parameters for the variance_adaptive_v2 experiment.
//
// WHY: Centralising every parameter here prevents "magic numbers" scattered
// across multiple translation units and makes the configuration snapshot
// written to config.json exactly reflect what the code used.
//
// No CUDA headers here — config.h is included by pure host-side code too.

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Per-phase workload specification
// ---------------------------------------------------------------------------
enum class DistributionKind {
    Uniform,      // param1=lo, param2=hi
    Bimodal,      // 50% Uniform[param1,param2], 50% Uniform[param3,param4]
    LogNormal,    // param1=mu (log-space), param2=sigma (log-space)
    //  clamped to [clamp_min, clamp_max]
};

struct PhaseConfig {
    std::string   name;
    int           start_task;   // inclusive
    int           end_task;     // exclusive
    DistributionKind distribution;
    double        param1;       // meaning depends on distribution
    double        param2;
    double        param3;       // only used for Bimodal
    double        param4;       // only used for Bimodal
    int           clamp_min;
    int           clamp_max;
};

// ---------------------------------------------------------------------------
// Top-level experiment configuration
// ---------------------------------------------------------------------------
struct ExperimentConfig {

    // -- Workload -------------------------------------------------------
    int          total_tasks       = 800'000;
    // cost_multiplier: applied inside the CUDA kernel (W *= cost_multiplier).
    // Set by the --calibrate pass; not chosen by guessing.
    int          cost_multiplier   = 1;
    unsigned int workload_seed     = 42;  // default; overridden per repetition
    std::vector<PhaseConfig> phases;      // populated by experiment preset

    // -- Chunk-size bounds ----------------------------------------------
    // MIN_CHUNK/MAX_CHUNK bound the adaptive scheduler's operating range.
    // Derived from the constraint: >= 20 chunks per phase at MAX_CHUNK:
    //   phase_tasks = total_tasks / num_phases = 800000 / 4 = 200000
    //   200000 / 10000 = 20 chunks at MAX_CHUNK  ✓
    int min_chunk          = 1'000;
    int max_chunk          = 10'000;
    int chunk_step         = 1'000;   // one step per adaptive decision
    int static_large_chunk = 10'000; // = max_chunk; the coarse-grained baseline

    // -- CUDA kernel ----------------------------------------------------
    int threads_per_block  = 256;

    // -- Sliding-window adaptive scheduler ------------------------------
    // WINDOW_SIZE = 20: large enough for stable variance; small enough to
    // react within a phase.  Bounds observation_lag_tasks <= WINDOW_SIZE.
    int    window_size      = 20;
    // Thresholds set during --calibrate based on observed per-phase variance.
    // Fixed before any final timing measurements are collected.
    double var_low_thresh   = 0.0;   // sigma^2 below this  → increase chunk
    double var_high_thresh  = 0.0;   // sigma^2 above this  → decrease chunk

    // response_fraction: for detection-lag computation.
    // A decision is a "response" if |new_chunk - prev_chunk| >=
    //   response_fraction * |target_chunk - prev_chunk|
    // Fixed at 0.25 before any experiments; do not change post-hoc.
    double response_fraction = 0.25;

    // -- Experiment execution -------------------------------------------
    int num_warmup_reps     = 3;    // excluded from all statistics
    int num_final_reps      = 20;   // minimum required; do not reduce
    int inter_arm_sleep_ms  = 200;  // between every arm change (thermal normalise)

    // -- Statistical methodology ----------------------------------------
    // All thresholds fixed before the first timing run.
    double alpha                = 0.05;  // significance level
    double min_effect_fraction  = 0.05;  // 5% minimum practical effect threshold

    // -- Output ---------------------------------------------------------
    std::string experiment_name;  // e.g. "C_lowHighLow"
    std::string results_base_dir; // e.g. "results/"
};
