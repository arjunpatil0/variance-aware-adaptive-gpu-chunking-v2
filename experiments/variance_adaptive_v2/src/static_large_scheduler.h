#pragma once
// static_large_scheduler.h — STATIC-LARGE fixed-chunk scheduler.
//
// WHY it exists as a separate class: the three-arm methodology requires
// STATIC-LARGE to be independently implemented so its chunk size cannot
// accidentally adapt.  The ExperimentRunner passes it the configured
// static_large_chunk and that value never changes during execution.
//
// The STATIC-MATCHED arm is NOT a separate class; it uses this same
// StaticScheduler with a different chunk_size parameter (the mean adaptive
// chunk, derived from PASS 1).

#include "scheduler.h"
#include "config.h"
#include <algorithm>

class StaticScheduler : public Scheduler {
public:
    // label:      "static_large" or "static_matched" — for trace output
    // chunk_size: fixed chunk size for this arm
    StaticScheduler(const std::string& label, int chunk_size)
        : label_(label), fixed_chunk_(chunk_size) {}

    void reset() override {
        decision_idx_ = 0;
        trace_.clear();
    }

    // select_chunk: always returns fixed_chunk_, clamped to chunk_remaining.
    // The CausalObserver is accepted for interface compatibility but never read
    // (a static scheduler has no use for past observations).
    int select_chunk(const CausalObserver& /*obs*/, int chunk_remaining) override {
        int chunk = std::min(fixed_chunk_, chunk_remaining);

        SchedulerDecision d{};
        d.decision_idx         = decision_idx_++;
        d.chunk_size           = chunk;
        // window fields are not applicable to static scheduler
        d.window_start_task    = -1;
        d.window_end_task      = -1;
        d.window_n             = 0;
        d.window_mean          = 0.0;
        d.window_variance      = 0.0;
        d.phase_id_at_decision = -1;
        d.observation_lag_tasks = -1;
        d.action_lag_tasks     = -1;
        d.is_response_decision = false;
        d.t_kernel_ms          = 0.0;
        d.t_total_chunk_ms     = 0.0;
        trace_.push_back(d);

        return chunk;
    }

    std::string name() const override { return label_; }

    const std::vector<SchedulerDecision>& trace() const override {
        return trace_;
    }

    void record_timing(double t_kernel_ms, double t_total_chunk_ms) override {
        if (!trace_.empty()) {
            trace_.back().t_kernel_ms      = t_kernel_ms;
            trace_.back().t_total_chunk_ms = t_total_chunk_ms;
        }
    }

    int fixed_chunk() const { return fixed_chunk_; }

private:
    std::string                    label_;
    int                            fixed_chunk_;
    int                            decision_idx_ = 0;
    std::vector<SchedulerDecision> trace_;
};
