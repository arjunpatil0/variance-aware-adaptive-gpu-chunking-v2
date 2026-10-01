#pragma once
// sliding_window.h — Fixed-capacity sliding window over double values.
//
// WHY: The adaptive scheduler uses a short-horizon window (not global history)
// to estimate local workload variance.  Using only recent observations makes
// the scheduler sensitive to regime changes without being anchored to distant
// history.
//
// This class is a pure host-side data structure — no CUDA involvement.
// The scheduler populates it with TASK COSTS (integers from the workload
// generator), not with wall-clock execution times.  Using task costs rather
// than timing avoids conflating workload variance with GPU timing noise.

#include <deque>
#include <cmath>
#include <cassert>
#include <vector>

class SlidingWindow {
public:
    explicit SlidingWindow(int capacity)
        : capacity_(capacity) {
        assert(capacity > 0);
    }

    // Push one observation.  If the window is full the oldest entry is evicted.
    void push(double value) {
        if ((int)data_.size() == capacity_) {
            data_.pop_front();
        }
        data_.push_back(value);
    }

    // Number of observations currently held (0 <= size() <= capacity()).
    int size()     const { return (int)data_.size(); }
    bool empty()   const { return data_.empty(); }
    bool full()    const { return (int)data_.size() == capacity_; }
    int capacity() const { return capacity_; }

    // Mean of observations currently in the window.
    // Returns 0.0 if empty.
    double mean() const {
        if (data_.empty()) return 0.0;
        double s = 0.0;
        for (double v : data_) s += v;
        return s / data_.size();
    }

    // Sample variance (unbiased, divides by n-1).
    // Returns 0.0 if fewer than 2 observations are present.
    // WHY sample variance (n-1): the window is a sample of a larger process,
    // not the complete population, so the unbiased estimator is appropriate.
    double variance() const {
        int n = (int)data_.size();
        if (n < 2) return 0.0;
        double m = mean();
        double s = 0.0;
        for (double v : data_) s += (v - m) * (v - m);
        return s / (n - 1);
    }

    // Reset to empty.
    void clear() { data_.clear(); }

    // Return all current observations (oldest first).
    std::vector<double> values() const {
        return std::vector<double>(data_.begin(), data_.end());
    }

private:
    int capacity_;
    std::deque<double> data_;
};
