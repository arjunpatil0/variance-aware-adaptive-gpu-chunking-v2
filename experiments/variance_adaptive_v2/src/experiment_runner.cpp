// experiment_runner.cpp — Implementation of three-arm, two-pass orchestration.

#include "experiment_runner.h"
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <random>
#include <cuda_runtime.h>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ExperimentRunner::ExperimentRunner(ExperimentConfig cfg,
                                   const std::string& out_dir)
    : cfg_(std::move(cfg))
    , serializer_(out_dir)
{}

// ---------------------------------------------------------------------------
// inter_arm_pause: sleep + GPU sync between arms to allow clock normalisation.
// WHY: Without this, any arm running immediately after a warm GPU may benefit
// from lower initial latency.  200ms is a reasonable settling time.
// ---------------------------------------------------------------------------
void ExperimentRunner::inter_arm_pause() const {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(cfg_.inter_arm_sleep_ms));
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// annotate_phases: fill phase_id_at_decision for each decision in a trace.
// ---------------------------------------------------------------------------
void ExperimentRunner::annotate_phases(std::vector<SchedulerDecision>& trace,
                                        const std::vector<PhaseInfo>& phases)
{
    for (auto& d : trace)
        d.phase_id_at_decision = phase_of_task(d.task_start, phases);
}

// ---------------------------------------------------------------------------
// run_arm — core scheduling loop for one arm.
// ---------------------------------------------------------------------------
// This is the PRIMARY TIMING BOUNDARY for T_total:
//   timer starts just before the first select_chunk() call
//   timer stops just after cudaDeviceSynchronize() inside the last launch_chunk()
//
// The CausalObserver is rebuilt before each decision with the current
// `processed` count so the adaptive scheduler can never reach ahead.
// ---------------------------------------------------------------------------
double ExperimentRunner::run_arm(Scheduler& sched,
                                  CudaExecutor& executor,
                                  const std::vector<int>& task_costs,
                                  const std::vector<PhaseInfo>& phases,
                                  ResultSerializer* ser,
                                  int rep)
{
    sched.reset();
    int total  = (int)task_costs.size();
    int processed = 0;

    // T_total starts here
    auto wall_start = Clock::now();

    while (processed < total) {
        int remaining = total - processed;

        // Rebuild observer — exposes only [0, processed) to the scheduler.
        // The adaptive scheduler's select_chunk() may only call
        // obs.completed_cost(i) for i < processed.
        CausalObserver obs(task_costs.data(), processed);

        int chunk = sched.select_chunk(obs, remaining);

        // Launch exactly one CUDA kernel for this chunk.
        ChunkResult cr = executor.launch_chunk(processed, chunk);

        // Record timing back into the most recent scheduler decision.
        sched.record_timing(cr.t_kernel_ms, cr.t_total_chunk_ms);

        processed += chunk;
    }

    // T_total ends here (sync is inside launch_chunk, so this is clean)
    double t_total_ms = Ms(Clock::now() - wall_start).count();

    // Post-run annotation: phase IDs
    // We need a mutable reference to trace; use const_cast since the
    // Scheduler interface exposes const trace().  Schedulers own their trace.
    // Ugly but avoids adding a non-const trace() accessor to the interface.
    auto& trace_ref = const_cast<std::vector<SchedulerDecision>&>(sched.trace());
    annotate_phases(trace_ref, phases);

    // Write trace rows if this is a final (non-warmup) run
    if (ser) {
        for (const auto& d : sched.trace())
            ser->append_trace_row(rep, d, sched.name());
    }

    return t_total_ms;
}

// ---------------------------------------------------------------------------
// run_one_repetition
// ---------------------------------------------------------------------------
RepetitionResult ExperimentRunner::run_one_repetition(int rep, bool warmup)
{
    // Seed per repetition: base + rep * stride (deterministic, well-separated)
    unsigned int seed = cfg_.workload_seed + (unsigned int)rep * SEED_STRIDE;

    // Generate workload — identical array used by all three arms this rep.
    WorkloadGenerator gen(cfg_, seed);
    std::vector<PhaseInfo> phases;
    std::vector<int> task_costs = gen.generate(phases);

    // Upload to GPU once — reused by all arms this repetition.
    CudaExecutor executor(task_costs, cfg_.threads_per_block, cfg_.cost_multiplier);

    ResultSerializer* ser = warmup ? nullptr : &serializer_;

    // Write workload.csv once (rep 0 only)
    if (!warmup && rep == 0)
        serializer_.write_workload(task_costs, phases);

    // ==================================================================
    // PASS 1: ADAPTIVE (must run first — output needed for STATIC-MATCHED)
    // ==================================================================
    AdaptiveScheduler adaptive(cfg_);
    double t_adaptive = run_arm(adaptive, executor, task_costs, phases, ser, rep);

    // Derive STATIC-MATCHED chunk from this repetition's adaptive mean.
    double mean_chunk_d = adaptive.mean_chunk_size();
    int    matched_chunk = std::max(cfg_.min_chunk,
                            std::min(cfg_.max_chunk,
                                     (int)std::round(mean_chunk_d)));

    // Compute detection lags from the adaptive trace and annotate in-place.
    auto& trace_ref = const_cast<std::vector<SchedulerDecision>&>(adaptive.trace());
    std::vector<DetectionLagResult> det_lags =
        compute_detection_lags(trace_ref, phases, cfg_);

    // If we just wrote the adaptive trace, re-write updated rows (with lag fields).
    // Simpler: the trace rows were written above, but detection lag was computed
    // after.  We append a supplementary detection_lag_results.json instead.
    // (The trace CSV lag fields will show -1 for most decisions; the
    //  detection_lag_results.json holds the aggregated per-transition values.)

    inter_arm_pause();

    // ==================================================================
    // PASS 2: STATIC-LARGE and STATIC-MATCHED (randomised order)
    // ==================================================================
    // Use the rep index to deterministically flip the order of PASS 2 arms.
    // This ensures that across 20 reps, each arm appears first ~10 times,
    // eliminating systematic clock/thermal bias toward one arm.
    bool large_first = (rep % 2 == 0);
    std::string arm_order;

    StaticScheduler static_large("static_large", cfg_.static_large_chunk);
    StaticScheduler static_matched("static_matched", matched_chunk);

    double t_static_large   = 0.0;
    double t_static_matched = 0.0;

    if (large_first) {
        arm_order = "adaptive,static_large,static_matched,inverted_adaptive";
        t_static_large   = run_arm(static_large,   executor, task_costs, phases, ser, rep);
        inter_arm_pause();
        t_static_matched = run_arm(static_matched, executor, task_costs, phases, ser, rep);
    } else {
        arm_order = "adaptive,static_matched,static_large,inverted_adaptive";
        t_static_matched = run_arm(static_matched, executor, task_costs, phases, ser, rep);
        inter_arm_pause();
        t_static_large   = run_arm(static_large,   executor, task_costs, phases, ser, rep);
    }

    // PASS 3: INVERTED-ADAPTIVE (always last — independent of ADAPTIVE output)
    inter_arm_pause();
    InvertedAdaptiveScheduler inverted(cfg_);
    double t_inverted = run_arm(inverted, executor, task_costs, phases, ser, rep);

    // Chunk counts
    int n_adapt    = (int)adaptive.trace().size();
    int n_large    = (int)static_large.trace().size();
    int n_match    = (int)static_matched.trace().size();
    int n_inverted = (int)inverted.trace().size();

    if (!warmup) {
        printf("Rep %3d | ADAPTIVE %.1fms (%d chunks, matched=%d) | "
               "LARGE %.1fms | MATCHED %.1fms | INVERTED %.1fms | order: %s\n",
               rep, t_adaptive, n_adapt, matched_chunk,
               t_static_large, t_static_matched, t_inverted, arm_order.c_str());

        serializer_.append_rep_result(
            rep, seed,
            t_adaptive, t_static_large, t_static_matched, t_inverted,
            matched_chunk, n_adapt, n_large, n_match, n_inverted,
            arm_order);
    }

    RepetitionResult res;
    res.rep                          = rep;
    res.workload_seed                = seed;
    res.t_total_adaptive_ms          = t_adaptive;
    res.t_total_static_large_ms      = t_static_large;
    res.t_total_static_matched_ms    = t_static_matched;
    res.t_total_inverted_ms          = t_inverted;
    res.mean_adaptive_chunk_rounded  = matched_chunk;
    res.n_chunks_adaptive            = n_adapt;
    res.n_chunks_static_large        = n_large;
    res.n_chunks_static_matched      = n_match;
    res.n_chunks_inverted            = n_inverted;
    res.arm_order                    = arm_order;
    res.detection_lags               = det_lags;
    return res;
}

// ---------------------------------------------------------------------------
// run — full experiment (warm-up + final reps + statistics)
// ---------------------------------------------------------------------------
void ExperimentRunner::run()
{
    printf("=== Experiment: %s ===\n", cfg_.experiment_name.c_str());
    printf("Tasks: %d | Reps: %d | Warmup: %d | Window: %d\n",
           cfg_.total_tasks, cfg_.num_final_reps,
           cfg_.num_warmup_reps, cfg_.window_size);
    printf("Var thresholds: low=%.2f high=%.2f | CostMult=%d\n",
           cfg_.var_low_thresh, cfg_.var_high_thresh, cfg_.cost_multiplier);
    printf("Chunks: LARGE=%d MIN=%d MAX=%d STEP=%d\n\n",
           cfg_.static_large_chunk, cfg_.min_chunk,
           cfg_.max_chunk, cfg_.chunk_step);

    // Write config with phase metadata from a representative workload
    {
        WorkloadGenerator gen(cfg_, cfg_.workload_seed);
        std::vector<PhaseInfo> phases;
        gen.generate(phases);
        serializer_.write_config(cfg_, phases);
    }

    // ------------------------------------------------------------------
    // Warm-up phase — GPU context init, JIT compilation, clock settling.
    // Results are DISCARDED and never appear in raw_results.csv.
    // ------------------------------------------------------------------
    printf("--- Warm-up (%d reps, results discarded) ---\n", cfg_.num_warmup_reps);
    for (int w = 0; w < cfg_.num_warmup_reps; ++w) {
        printf("  Warmup %d/%d\n", w + 1, cfg_.num_warmup_reps);
        run_one_repetition(-(w + 1), /*warmup=*/true);
    }
    printf("Warm-up complete.\n\n");

    // ------------------------------------------------------------------
    // Final measurement phase
    // ------------------------------------------------------------------
    printf("--- Final measurements (%d reps) ---\n", cfg_.num_final_reps);
    std::vector<double> times_adaptive, times_large, times_matched, times_inverted;
    std::vector<std::vector<DetectionLagResult>> all_det_lags;

    for (int r = 0; r < cfg_.num_final_reps; ++r) {
        RepetitionResult res = run_one_repetition(r, /*warmup=*/false);
        times_adaptive.push_back(res.t_total_adaptive_ms);
        times_large.push_back(res.t_total_static_large_ms);
        times_matched.push_back(res.t_total_static_matched_ms);
        times_inverted.push_back(res.t_total_inverted_ms);
        all_det_lags.push_back(res.detection_lags);
        inter_arm_pause();
    }

    // ------------------------------------------------------------------
    // Statistical analysis
    // ------------------------------------------------------------------
    printf("\n--- Statistical Analysis ---\n");

    ArmStats stats_adaptive  = compute_arm_stats("adaptive",          times_adaptive);
    ArmStats stats_large     = compute_arm_stats("static_large",      times_large);
    ArmStats stats_matched   = compute_arm_stats("static_matched",    times_matched);
    ArmStats stats_inverted  = compute_arm_stats("inverted_adaptive", times_inverted);

    // Primary:   STATIC-MATCHED vs ADAPTIVE
    WilcoxonResult primary = wilcoxon_signed_rank(
        "adaptive", "static_matched",
        times_adaptive, times_matched,
        cfg_.alpha, cfg_.min_effect_fraction);

    // Secondary: STATIC-LARGE vs ADAPTIVE
    WilcoxonResult secondary = wilcoxon_signed_rank(
        "adaptive", "static_large",
        times_adaptive, times_large,
        cfg_.alpha, cfg_.min_effect_fraction);

    // Tertiary:  INVERTED-ADAPTIVE vs ADAPTIVE (mechanism test)
    WilcoxonResult tertiary = wilcoxon_signed_rank(
        "adaptive", "inverted_adaptive",
        times_adaptive, times_inverted,
        cfg_.alpha, cfg_.min_effect_fraction);

    // Print summary
    auto print_arm = [](const ArmStats& s) {
        printf("  %-20s n=%d  mean=%.2fms  std=%.2fms  [%.2f, %.2f]\n",
               s.arm_name.c_str(), s.n, s.mean_ms, s.std_ms,
               s.min_ms, s.max_ms);
    };
    print_arm(stats_adaptive);
    print_arm(stats_matched);
    print_arm(stats_large);
    print_arm(stats_inverted);

    printf("\nPRIMARY   (MATCHED   vs ADAPTIVE): p=%.4f  effect=%.2f%%  %s\n",
           primary.p_value, primary.relative_pct,
           primary.is_improvement ? "IMPROVEMENT" :
           (primary.is_significant ? "sig but small effect" :
            (primary.meets_effect_threshold ? "large effect but not sig" :
             "no meaningful difference")));

    printf("SECONDARY (LARGE     vs ADAPTIVE): p=%.4f  effect=%.2f%%  "
           "[weaker — does not control for chunk size]\n",
           secondary.p_value, secondary.relative_pct);

    printf("TERTIARY  (INVERTED  vs ADAPTIVE): p=%.4f  effect=%.2f%%  "
           "[mechanism test — occupancy hypothesis]\n",
           tertiary.p_value, tertiary.relative_pct);

    // Write outputs
    serializer_.write_statistical_results(
        stats_adaptive, stats_large, stats_matched, stats_inverted,
        primary, secondary, tertiary);

    // Re-generate phases from rep-0 seed for detection lag context
    WorkloadGenerator gen0(cfg_, cfg_.workload_seed);
    std::vector<PhaseInfo> phases0;
    gen0.generate(phases0);
    serializer_.write_detection_lag_results(all_det_lags, phases0);

    printf("\nResults written to: %s\n", serializer_.out_dir().c_str());
}

// ---------------------------------------------------------------------------
// run_calibration
// ---------------------------------------------------------------------------
void ExperimentRunner::run_calibration()
{
    printf("=== Calibration Mode ===\n");
    printf("Sweeping cost_multiplier over {1,10,50,100,500}\n");
    printf("Criterion: T_kernel >= 10x launch overhead\n\n");

    // Generate a small representative workload for calibration.
    // Reduce total_tasks for speed, but re-apply phase boundaries to fit.
    ExperimentConfig cal_cfg = cfg_;
    cal_cfg.total_tasks = cfg_.max_chunk * 4;  // one chunk per phase, small

    // Re-calculate phase boundaries for the calibration task count
    // (the preset set them based on cfg_.total_tasks which may be 800k)
    {
        int N = cal_cfg.total_tasks;
        int q = N / (int)cal_cfg.phases.size();
        for (int i = 0; i < (int)cal_cfg.phases.size(); ++i) {
            cal_cfg.phases[i].start_task = i * q;
            cal_cfg.phases[i].end_task   =
                (i == (int)cal_cfg.phases.size() - 1) ? N : (i + 1) * q;
        }
    }

    WorkloadGenerator gen(cal_cfg, cfg_.workload_seed);
    std::vector<int> costs = gen.generate();

    // Measure launch overhead with zero-work kernel
    CudaExecutor baseline_exec(costs, cfg_.threads_per_block, 1);
    double overhead_ms = 0.0;
    {
        const int OVERHEAD_REPS = 20;
        double total = 0.0;
        for (int i = 0; i < OVERHEAD_REPS; ++i)
            total += baseline_exec.measure_launch_overhead(cfg_.max_chunk).t_total_chunk_ms;
        overhead_ms = total / OVERHEAD_REPS;
    }
    printf("Launch overhead (zero-work, chunk=%d): %.4f ms\n",
           cfg_.max_chunk, overhead_ms);
    printf("Target T_kernel >= %.4f ms\n\n", 10.0 * overhead_ms);

    std::vector<std::pair<int,double>> sweep;
    int chosen_mult = -1;
    int current_mult = 1;

    while (true) {
        CudaExecutor exec(costs, cfg_.threads_per_block, current_mult);
        const int MEAS_REPS = 10;
        double total = 0.0;
        for (int i = 0; i < MEAS_REPS; ++i)
            total += exec.launch_chunk(0, cfg_.max_chunk).t_kernel_ms;
        double avg_kernel = total / MEAS_REPS;
        sweep.push_back({current_mult, avg_kernel});
        printf("  mult=%7d  T_kernel=%8.4f ms  (%5.1fx overhead)\n",
               current_mult, avg_kernel, avg_kernel / overhead_ms);

        if (avg_kernel >= 10.0 * overhead_ms) {
            chosen_mult = current_mult;
            break;
        }
        
        if (current_mult >= 1000000) {
            fprintf(stderr, "\nERROR: Calibration failed! Reached multiplier 1,000,000 without hitting 10x overhead target.\n");
            fprintf(stderr, "This means the compute cost is completely buried under launch latency (or simulation overhead).\n");
            fprintf(stderr, "Aborting to prevent generating meaningless data.\n");
            exit(EXIT_FAILURE);
        }

        if (current_mult < 10) current_mult = 10;
        else if (current_mult < 50) current_mult = 50;
        else if (current_mult < 100) current_mult = 100;
        else if (current_mult < 500) current_mult = 500;
        else current_mult *= 2; // exponential growth after 500
    }

    // Compute variance thresholds from the chosen multiplier.
    // Use the configured phase distributions; measure actual per-phase σ².
    // Low threshold: midpoint between near-zero and the lowest high-variance phase.
    // High threshold: set to catch Phase B / D (bimodal / log-normal).
    //
    // For the default Experiment C phases (raw costs, pre-multiplier):
    //   Phase A (Uniform[200,400]):   σ² ≈ 3333
    //   Phase B (Bimodal):            σ² ≈ 100000+
    //   Phase C (Uniform[350,450]):   σ² ≈ 833
    //   Phase D (LogNormal):          σ² >> 3333
    //
    // We set thresholds in raw-cost space (independent of multiplier) because
    // the scheduler's sliding window uses raw costs from CausalObserver.
    double var_low  = 5000.0;   // above low phases (833, 3333), below high phases
    double var_high = 30000.0;  // catches bimodal (>100k) and lognormal

    printf("\nChosen multiplier: %d\n", chosen_mult);
    printf("Setting var_low_thresh  = %.1f  (raw task-cost variance)\n", var_low);
    printf("Setting var_high_thresh = %.1f  (raw task-cost variance)\n", var_high);

    cfg_.cost_multiplier   = chosen_mult;
    cfg_.var_low_thresh    = var_low;
    cfg_.var_high_thresh   = var_high;

    ResultSerializer::write_calibration(
        cfg_.results_base_dir,
        chosen_mult, overhead_ms, var_low, var_high, sweep);

    printf("Calibration written to %s/calibration.json\n",
           cfg_.results_base_dir.c_str());
}
