#pragma once
// experiment_runner.h — Orchestrates the three-arm, two-pass experiment.
//
// PROTOCOL PER REPETITION (ref: implementation plan §7):
//
//   PASS 1 (always first — required to derive STATIC-MATCHED chunk size):
//     Run ADAPTIVE scheduler on the generated workload.
//     Record mean adaptive chunk size.
//
//   PASS 2 (internal order RANDOMISED to avoid systematic arm-ordering bias):
//     Run STATIC-LARGE  with chunk = cfg.static_large_chunk
//     Run STATIC-MATCHED with chunk = round(mean_adaptive_chunk)
//     The two arms are reordered each repetition by a coin flip seeded on
//     the repetition index.
//
//   Between every arm: sleep(cfg.inter_arm_sleep_ms) + cudaDeviceSynchronize()
//   so GPU clock state has a chance to normalize before the next arm.
//
// CAUSAL GUARANTEE:
//   The CausalObserver passed to AdaptiveScheduler::select_chunk() contains
//   only task costs for [0, processed) — it is rebuilt before each decision
//   with the current processed count.  The full task_costs array is held by
//   ExperimentRunner; it is NEVER passed directly to the scheduler.
//
// WORKLOAD SEEDING:
//   Each repetition gets seed = cfg.workload_seed + rep * SEED_STRIDE.
//   SEED_STRIDE is large enough to avoid seed correlation between reps.
//   All three arms within one repetition use the SAME seed (same workload).

#include "config.h"
#include "workload_generator.h"
#include "adaptive_scheduler.h"
#include "inverted_adaptive_scheduler.h"
#include "static_large_scheduler.h"
#include "cuda_executor.h"
#include "statistical_analysis.h"
#include "result_serializer.h"
#include <vector>
#include <string>
#include <functional>

// Spacing between per-rep seeds — large prime to avoid short-period overlap.
static constexpr unsigned int SEED_STRIDE = 1000003u;

struct RepetitionResult {
    int          rep;
    unsigned int workload_seed;
    double       t_total_adaptive_ms;
    double       t_total_static_large_ms;
    double       t_total_static_matched_ms;
    double       t_total_inverted_ms;       // NEW: INVERTED-ADAPTIVE arm
    int          mean_adaptive_chunk_rounded;
    int          n_chunks_adaptive;
    int          n_chunks_static_large;
    int          n_chunks_static_matched;
    int          n_chunks_inverted;         // NEW
    std::string  arm_order;
    std::vector<DetectionLagResult> detection_lags;
};

class ExperimentRunner {
public:
    ExperimentRunner(ExperimentConfig cfg,
                     const std::string& out_dir);

    // Run all warmup + final repetitions and write results.
    void run();

    // Run calibration mode: sweep cost_multiplier, pick the smallest that
    // puts T_kernel >= 10x launch overhead, set var thresholds accordingly.
    // Writes results/calibration.json and updates cfg_.
    void run_calibration();

    const ExperimentConfig& config() const { return cfg_; }

private:
    ExperimentConfig  cfg_;
    ResultSerializer  serializer_;

    // Run one repetition (warmup=true → results not persisted).
    RepetitionResult run_one_repetition(int rep, bool warmup);

    // Run one scheduler arm on the given workload.
    // Returns T_total (wall-clock for the full scheduler loop).
    double run_arm(Scheduler& sched,
                   CudaExecutor& executor,
                   const std::vector<int>& task_costs,
                   const std::vector<PhaseInfo>& phases,
                   ResultSerializer* ser,   // null during warmup
                   int rep);

    // Sleep + sync between arms
    void inter_arm_pause() const;

    // Annotate phase_id on each SchedulerDecision in the trace.
    static void annotate_phases(std::vector<SchedulerDecision>& trace,
                                 const std::vector<PhaseInfo>& phases);
};
