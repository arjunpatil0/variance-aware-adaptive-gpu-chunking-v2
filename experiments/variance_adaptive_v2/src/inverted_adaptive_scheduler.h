#pragma once
// inverted_adaptive_scheduler.h — Inverted causal sliding-window scheduler.
//
// PURPOSE (mechanism test):
//   This arm implements the exact OPPOSITE policy of AdaptiveScheduler.
//   When variance is HIGH -> increase chunk size toward MAX_CHUNK.
//   When variance is LOW  -> decrease chunk size toward MIN_CHUNK.
//
// The occupancy hypothesis predicts that ADAPTIVE loses because shrinking
// chunks during high-variance phases starves the hardware warp scheduler.
// If correct, INVERTED-ADAPTIVE should IMPROVE over ADAPTIVE during those
// phases, because it grows chunks during high variance, giving the warp
// scheduler a larger pool of work to overlap with stragglers.
//
// CAUSAL GUARANTEE: identical to AdaptiveScheduler — CausalObserver
// enforces the past-only boundary.  No future task data is accessible.
//
// POLICY (inverted):
//   σ² > VAR_HIGH_THRESH:
//       chunk = min(current_chunk + CHUNK_STEP, MAX_CHUNK)   [INVERTED]
//       (high variance -> GROW chunk, keep HW queues saturated)
//
//   σ² < VAR_LOW_THRESH:
//       chunk = max(current_chunk - CHUNK_STEP, MIN_CHUNK)   [INVERTED]
//       (low variance -> SHRINK chunk, amortize overhead less aggressively)
//
//   else: chunk unchanged

#include "scheduler.h"
#include "sliding_window.h"
#include "config.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

class InvertedAdaptiveScheduler : public Scheduler {
public:
    explicit InvertedAdaptiveScheduler(const ExperimentConfig& cfg)
        : cfg_(cfg)
        , window_(cfg.window_size)
        , current_chunk_(cfg.max_chunk)  // cold-start: begin with MAX_CHUNK
    {}

    void reset() override {
        window_.clear();
        current_chunk_ = cfg_.max_chunk;
        decision_idx_  = 0;
        trace_.clear();
    }

    int select_chunk(const CausalObserver& obs, int chunk_remaining) override {
        // Step 1: rebuild sliding window (identical to AdaptiveScheduler)
        window_.clear();
        int ws = obs.window_start(cfg_.window_size);
        int we = obs.num_completed();

        for (int i = ws; i < we; ++i)
            window_.push(static_cast<double>(obs.completed_cost(i)));

        double w_mean = window_.mean();
        double w_var  = window_.variance();
        int    w_n    = window_.size();

        // Step 2: apply INVERTED variance-based policy.
        if (w_n >= 2) {
            if (w_var > cfg_.var_high_thresh) {
                // High variance -> INCREASE chunk (inverted from adaptive)
                current_chunk_ = std::min(current_chunk_ + cfg_.chunk_step,
                                          cfg_.max_chunk);
            } else if (w_var < cfg_.var_low_thresh) {
                // Low variance -> DECREASE chunk (inverted from adaptive)
                current_chunk_ = std::max(current_chunk_ - cfg_.chunk_step,
                                          cfg_.min_chunk);
            }
        }

        // Step 3: clamp to remaining work
        int chunk = std::min(current_chunk_, chunk_remaining);

        // Step 4: record the decision in the trace
        SchedulerDecision d{};
        d.decision_idx         = decision_idx_++;
        d.task_start           = obs.num_completed();
        d.task_end             = obs.num_completed() + chunk;
        d.chunk_size           = chunk;
        d.window_start_task    = (w_n > 0) ? ws : -1;
        d.window_end_task      = we;
        d.window_n             = w_n;
        d.window_mean          = w_mean;
        d.window_variance      = w_var;
        d.phase_id_at_decision = -1;
        d.observation_lag_tasks = -1;
        d.action_lag_tasks      = -1;
        d.is_response_decision  = false;
        d.t_kernel_ms           = 0.0;
        d.t_total_chunk_ms      = 0.0;
        trace_.push_back(d);

        return chunk;
    }

    std::string name() const override { return "inverted_adaptive"; }

    const std::vector<SchedulerDecision>& trace() const override {
        return trace_;
    }

    void record_timing(double t_kernel_ms, double t_total_chunk_ms) override {
        if (!trace_.empty()) {
            trace_.back().t_kernel_ms      = t_kernel_ms;
            trace_.back().t_total_chunk_ms = t_total_chunk_ms;
        }
    }

private:
    const ExperimentConfig&         cfg_;
    SlidingWindow                   window_;
    int                             current_chunk_;
    int                             decision_idx_ = 0;
    std::vector<SchedulerDecision>  trace_;
};
