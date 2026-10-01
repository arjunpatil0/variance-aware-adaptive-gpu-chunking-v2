#pragma once
// adaptive_scheduler.h — Causal sliding-window variance-aware scheduler.
//
// CAUSAL GUARANTEE: select_chunk() receives a CausalObserver whose boundary
// is exactly the number of tasks completed before this decision.  The window
// is built from completed_cost(i) for i in [window_start, num_completed).
// There is no path — no precomputed array, no helper function, no indirect
// reference — through which task costs beyond num_completed() can reach the
// chunk-size decision.
//
// POLICY (transparent and mathematically explicit):
//   Let σ² = sample variance of the WINDOW_SIZE most recent completed costs.
//
//   σ² < VAR_LOW_THRESH:
//       chunk = min(current_chunk + CHUNK_STEP, MAX_CHUNK)
//       (workload is uniform; use larger chunks to amortize launch overhead)
//
//   σ² > VAR_HIGH_THRESH:
//       chunk = max(current_chunk - CHUNK_STEP, MIN_CHUNK)
//       (workload is heterogeneous; finer granularity gives more frequent
//        opportunities to react between chunks)
//
//   else: chunk unchanged
//       (workload variance is in an intermediate zone; hold steady)
//
// WHY linear step (not multiplicative): a fixed step size makes the
// scheduler's trajectory easy to reason about and explain.  One CHUNK_STEP
// per decision prevents over-reaction to transient spikes.
//
// WHY min/max clamps: without bounds the scheduler could produce chunk sizes
// of 1 (excessive launches) or N (no adaptivity).
//
// COLD START: when fewer than WINDOW_SIZE observations are available, all
// available observations are used.  Initial chunk size = MAX_CHUNK so the
// first decision is conservative (large chunks).
//
// The scheduler is stateful only in: current_chunk_ (policy state) and
// decision_idx_ (for tracing).  No hidden future-data reference.

#include "scheduler.h"
#include "sliding_window.h"
#include "config.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

class AdaptiveScheduler : public Scheduler {
public:
    explicit AdaptiveScheduler(const ExperimentConfig& cfg)
        : cfg_(cfg)
        , window_(cfg.window_size)
        , current_chunk_(cfg.max_chunk)   // cold-start: begin with MAX_CHUNK
    {}

    void reset() override {
        window_.clear();
        current_chunk_ = cfg_.max_chunk;
        decision_idx_  = 0;
        trace_.clear();
    }

    // select_chunk: causal decision using only completed task costs.
    int select_chunk(const CausalObserver& obs, int chunk_remaining) override {
        // ----------------------------------------------------------------
        // Step 1: rebuild sliding window from completed observations.
        // We use completed_cost(i) which asserts i < obs.num_completed().
        // No future costs are accessible through this call.
        // ----------------------------------------------------------------
        window_.clear();
        int ws = obs.window_start(cfg_.window_size);
        int we = obs.num_completed();   // exclusive — the causal boundary

        for (int i = ws; i < we; ++i)
            window_.push(static_cast<double>(obs.completed_cost(i)));

        double w_mean = window_.mean();
        double w_var  = window_.variance();
        int    w_n    = window_.size();

        // ----------------------------------------------------------------
        // Step 2: apply variance-based policy.
        // ----------------------------------------------------------------
        if (w_n >= 2) {
            if (w_var < cfg_.var_low_thresh) {
                // Low variance → increase chunk (workload is uniform)
                current_chunk_ = std::min(current_chunk_ + cfg_.chunk_step,
                                          cfg_.max_chunk);
            } else if (w_var > cfg_.var_high_thresh) {
                // High variance → decrease chunk (workload is heterogeneous)
                current_chunk_ = std::max(current_chunk_ - cfg_.chunk_step,
                                          cfg_.min_chunk);
            }
            // else: variance in intermediate zone → hold current chunk
        }
        // If w_n < 2 (cold start): hold current_chunk_ = max_chunk.

        // ----------------------------------------------------------------
        // Step 3: clamp to remaining work.
        // ----------------------------------------------------------------
        int chunk = std::min(current_chunk_, chunk_remaining);

        // ----------------------------------------------------------------
        // Step 4: record the decision in the trace.
        // ----------------------------------------------------------------
        SchedulerDecision d{};
        d.decision_idx         = decision_idx_++;
        d.task_start           = obs.num_completed();
        d.task_end             = obs.num_completed() + chunk;
        d.chunk_size           = chunk;
        d.window_start_task    = (w_n > 0) ? ws : -1;
        d.window_end_task      = we;  // == task_start (causal boundary marker)
        d.window_n             = w_n;
        d.window_mean          = w_mean;
        d.window_variance      = w_var;
        d.phase_id_at_decision = -1;  // set by ExperimentRunner
        // Lag fields populated later by statistical_analysis
        d.observation_lag_tasks = -1;
        d.action_lag_tasks      = -1;
        d.is_response_decision  = false;
        d.t_kernel_ms           = 0.0;
        d.t_total_chunk_ms      = 0.0;
        trace_.push_back(d);

        return chunk;
    }

    std::string name() const override { return "adaptive"; }

    const std::vector<SchedulerDecision>& trace() const override {
        return trace_;
    }

    void record_timing(double t_kernel_ms, double t_total_chunk_ms) override {
        if (!trace_.empty()) {
            trace_.back().t_kernel_ms      = t_kernel_ms;
            trace_.back().t_total_chunk_ms = t_total_chunk_ms;
        }
    }

    // Mean chunk size across all decisions (used to derive STATIC-MATCHED chunk).
    double mean_chunk_size() const {
        if (trace_.empty()) return cfg_.max_chunk;
        double s = 0.0;
        for (const auto& d : trace_) s += d.chunk_size;
        return s / trace_.size();
    }

    // Set phase_id on the most recent decision.
    // Called by ExperimentRunner after computing which phase task_start is in.
    void annotate_phase(int phase_id) {
        if (!trace_.empty())
            trace_.back().phase_id_at_decision = phase_id;
    }

private:
    const ExperimentConfig&        cfg_;
    SlidingWindow                  window_;
    int                            current_chunk_;
    int                            decision_idx_ = 0;
    std::vector<SchedulerDecision> trace_;
};
