#pragma once
// scheduler.h — Abstract scheduler interface and CausalObserver.
//
// CausalObserver enforces the causal boundary: the adaptive scheduler may
// only inspect task costs for tasks that have already completed.  Future
// task costs are structurally inaccessible through this interface — not just
// "by convention."
//
// The host-side experiment runner constructs a new CausalObserver before
// each scheduling decision, with num_completed reflecting how many tasks
// have been processed so far.  Only AFTER the chunk kernel completes and
// cudaDeviceSynchronize() returns is the observer updated (num_completed
// incremented) for the next decision.

#include "config.h"
#include <vector>
#include <cassert>
#include <string>

// ---------------------------------------------------------------------------
// CausalObserver
// ---------------------------------------------------------------------------
// Wraps the task-cost array but exposes only the completed prefix.
// The adaptive scheduler's decide() method receives const CausalObserver&;
// it cannot reach beyond num_completed() without a deliberate assert failure.
class CausalObserver {
public:
    // costs: full task-cost array (size = total_tasks).
    // num_completed: how many tasks have been processed (causal boundary).
    CausalObserver(const int* costs, int num_completed)
        : costs_(costs), n_completed_(num_completed) {}

    int num_completed() const { return n_completed_; }

    // Safe cost accessor.  Aborts if task_idx >= num_completed().
    // This intentional hard failure makes accidental future-data access
    // immediately visible during testing rather than silently wrong.
    int completed_cost(int task_idx) const {
        assert(task_idx >= 0 && task_idx < n_completed_ &&
               "CausalObserver: attempt to read future task cost");
        return costs_[task_idx];
    }

    // Returns the inclusive start index of the sliding-window range given
    // the configured window size.
    // Range of valid indices: [window_start(w), n_completed_)
    int window_start(int window_size) const {
        int s = n_completed_ - window_size;
        return s < 0 ? 0 : s;
    }

private:
    const int* costs_;   // full array; NOT exposed directly
    int        n_completed_;
};

// ---------------------------------------------------------------------------
// Per-chunk decision record — written into the scheduler trace
// ---------------------------------------------------------------------------
struct SchedulerDecision {
    int    decision_idx;          // 0-based scheduling decision number
    int    task_start;            // first task in this chunk (inclusive)
    int    task_end;              // last task + 1 (exclusive)
    int    chunk_size;            // task_end - task_start
    int    window_start_task;     // first task index in the sliding window
    int    window_end_task;       // = task_start (the causal boundary)
    int    window_n;              // number of observations used
    double window_mean;           // mean of window task costs
    double window_variance;       // sample variance of window task costs
    int    phase_id_at_decision;  // phase containing task_start (-1 if unknown)

    // Detection lag fields — populated by statistical_analysis after the run.
    // -1 = not applicable (not a transition point) or not yet computed.
    int  observation_lag_tasks;   // tasks from regime boundary to window detection
    int  action_lag_tasks;        // tasks from regime boundary to chunk-size action
    bool is_response_decision;    // true if this decision is the first response

    // Timing — populated by ExperimentRunner after the kernel completes.
    double t_kernel_ms;           // CUDA event GPU time for this chunk
    double t_total_chunk_ms;      // wall-clock time including launch + sync
};

// ---------------------------------------------------------------------------
// Abstract Scheduler interface
// ---------------------------------------------------------------------------
class Scheduler {
public:
    virtual ~Scheduler() = default;

    // Reset internal state for a new repetition.
    virtual void reset() = 0;

    // Select the next chunk size.
    // Called once per chunk, BEFORE the kernel launches.
    // chunk_remaining: tasks left to process (chunk cannot exceed this).
    virtual int select_chunk(const CausalObserver& obs,
                             int chunk_remaining) = 0;

    // Scheduler type name for output labeling.
    virtual std::string name() const = 0;

    // The trace of all decisions made since the last reset().
    virtual const std::vector<SchedulerDecision>& trace() const = 0;

    // Record timing for the most recent decision (called after kernel completes).
    virtual void record_timing(double t_kernel_ms, double t_total_chunk_ms) = 0;
};
