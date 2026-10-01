// statistical_analysis.cpp — Implementation of all statistical computations.

#include "statistical_analysis.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

// Standard normal CDF via complementary error function (from <cmath>).
// erfc is available in C++11 and later; no external library required.
static double normal_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// 95% CI on mean(d) using normal approximation (symmetric, approximate).
// WHY approximate: the Wilcoxon test is nonparametric but a simple symmetric
// CI on the mean difference is still informative.  Document as approximate.
static void compute_ci_95(const std::vector<double>& diffs,
                           double& ci_lower, double& ci_upper)
{
    int n = (int)diffs.size();
    if (n == 0) { ci_lower = ci_upper = 0.0; return; }
    double s = 0.0;
    for (double d : diffs) s += d;
    double mean = s / n;
    double ss = 0.0;
    for (double d : diffs) ss += (d - mean) * (d - mean);
    double sd = (n > 1) ? std::sqrt(ss / (n - 1)) : 0.0;
    double se = sd / std::sqrt((double)n);
    ci_lower = mean - 1.96 * se;
    ci_upper = mean + 1.96 * se;
}

// ---------------------------------------------------------------------------
// compute_arm_stats
// ---------------------------------------------------------------------------
ArmStats compute_arm_stats(const std::string& name,
                            const std::vector<double>& totals_ms)
{
    ArmStats s;
    s.arm_name = name;
    s.n = (int)totals_ms.size();
    if (s.n == 0) return s;

    double sum = 0.0, sum_sq = 0.0;
    s.min_ms = totals_ms[0];
    s.max_ms = totals_ms[0];
    for (double v : totals_ms) {
        sum    += v;
        sum_sq += v * v;
        s.min_ms = std::min(s.min_ms, v);
        s.max_ms = std::max(s.max_ms, v);
    }
    s.mean_ms = sum / s.n;
    double var = (s.n > 1)
        ? (sum_sq / s.n - s.mean_ms * s.mean_ms) * s.n / (s.n - 1)
        : 0.0;
    s.std_ms = std::sqrt(var);
    return s;
}

// ---------------------------------------------------------------------------
// wilcoxon_signed_rank
// ---------------------------------------------------------------------------
WilcoxonResult wilcoxon_signed_rank(
    const std::string& arm_a_name,
    const std::string& arm_b_name,
    const std::vector<double>& a_ms,
    const std::vector<double>& b_ms,
    double alpha,
    double min_effect_fraction)
{
    assert(a_ms.size() == b_ms.size());
    int n = (int)a_ms.size();
    assert(n >= 2);

    // Paired differences: d = a - b
    // Negative d means a (adaptive) is faster.
    std::vector<double> diffs(n);
    for (int i = 0; i < n; ++i)
        diffs[i] = a_ms[i] - b_ms[i];

    // Remove exact zeros (they carry no rank information)
    std::vector<double> nonzero;
    nonzero.reserve(n);
    for (double d : diffs)
        if (d != 0.0) nonzero.push_back(d);
    int m = (int)nonzero.size();

    WilcoxonResult r;
    r.arm_a = arm_a_name;
    r.arm_b = arm_b_name;
    r.n     = n;

    // Mean paired difference and relative percentage
    double sum_d = 0.0;
    for (double d : diffs) sum_d += d;
    r.mean_diff_ms = sum_d / n;

    double ss_d = 0.0;
    for (double d : diffs) ss_d += (d - r.mean_diff_ms) * (d - r.mean_diff_ms);
    r.std_diff_ms = (n > 1) ? std::sqrt(ss_d / (n - 1)) : 0.0;

    // Relative %: positive = a is faster than b
    // mean(b) is the reference
    double mean_b = 0.0;
    for (double v : b_ms) mean_b += v;
    mean_b /= n;
    r.relative_pct = (mean_b != 0.0)
        ? 100.0 * (-r.mean_diff_ms) / mean_b
        : 0.0;

    compute_ci_95(diffs, r.ci_lower_95, r.ci_upper_95);

    if (m == 0) {
        // All differences are zero
        r.W_stat        = 0.0;
        r.Z_score       = 0.0;
        r.p_value       = 1.0;
        r.effect_size_r = 0.0;
        r.is_significant          = false;
        r.meets_effect_threshold  = false;
        r.is_improvement          = false;
        r.interpretation = "All paired differences are exactly zero.";
        return r;
    }

    // ------------------------------------------------------------------
    // Sort by absolute value, assign ranks (averaging ties)
    // ------------------------------------------------------------------
    std::vector<int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) {
        return std::abs(nonzero[i]) < std::abs(nonzero[j]);
    });

    std::vector<double> ranks(m, 0.0);
    double tie_correction = 0.0;
    int i = 0;
    while (i < m) {
        int j = i;
        double abs_val = std::abs(nonzero[idx[i]]);
        while (j < m && std::abs(nonzero[idx[j]]) == abs_val) ++j;
        // Average rank for tied group [i, j)
        double avg_rank = (i + 1 + j) / 2.0;  // 1-indexed
        int    t        = j - i;
        for (int k = i; k < j; ++k) ranks[k] = avg_rank;
        if (t > 1) tie_correction += (double)(t * t * t - t) / 48.0;
        i = j;
    }

    // W+ = sum of ranks for positive differences
    // W- = sum of ranks for negative differences
    double W_plus = 0.0, W_minus = 0.0;
    for (int k = 0; k < m; ++k) {
        if (nonzero[idx[k]] > 0.0) W_plus  += ranks[k];
        else                        W_minus += ranks[k];
    }
    double W = std::min(W_plus, W_minus);
    r.W_stat = W;

    // Normal approximation (valid for m >= 10; we have m <= n, n >= 20)
    double mu_W    = m * (m + 1.0) / 4.0;
    double sigma2_W = m * (m + 1.0) * (2.0 * m + 1.0) / 24.0 - tie_correction;
    double sigma_W  = std::sqrt(std::max(sigma2_W, 1e-12));

    // Continuity correction: subtract 0.5 from |W - mu_W|, clamping to 0
    double Z_raw = (W - mu_W);
    double Z_cc  = std::max(0.0, std::abs(Z_raw) - 0.5) / sigma_W;
    // Two-tailed
    r.Z_score       = Z_cc;
    r.p_value       = std::min(1.0, 2.0 * (1.0 - normal_cdf(Z_cc)));
    r.effect_size_r = r.Z_score / std::sqrt((double)n);

    // ------------------------------------------------------------------
    // Significance and effect thresholds
    // ------------------------------------------------------------------
    r.is_significant         = (r.p_value < alpha);
    // "Meets effect threshold" means adaptive is actually faster AND by >= 5%
    r.meets_effect_threshold = (r.relative_pct >= min_effect_fraction * 100.0);
    r.is_improvement         = r.is_significant && r.meets_effect_threshold;

    // Human-readable interpretation following pre-specified rules
    std::ostringstream interp;
    interp << arm_a_name << " vs " << arm_b_name << ": ";
    if (r.is_improvement) {
        interp << "IMPROVEMENT — p=" << r.p_value
               << " (< alpha=" << alpha << "), effect=" << r.relative_pct
               << "% (>= 5% threshold). Both criteria satisfied.";
    } else if (r.is_significant && !r.meets_effect_threshold) {
        interp << "Statistically significant (p=" << r.p_value
               << ") but effect=" << r.relative_pct
               << "% does not meet the 5% minimum practical threshold. "
               << "Result described as 'lower observed runtime' only.";
        r.caution = "Significance without practical effect: do not label as improvement.";
    } else if (!r.is_significant && r.meets_effect_threshold) {
        interp << "Effect=" << r.relative_pct
               << "% exceeds 5% threshold but p=" << r.p_value
               << " >= alpha=" << alpha << ". Not statistically significant. "
               << "Result described as 'lower observed runtime' only.";
        r.caution = "Effect without significance: do not label as improvement.";
    } else {
        interp << "No meaningful difference. p=" << r.p_value
               << ", effect=" << r.relative_pct << "%.";
    }
    r.interpretation = interp.str();

    return r;
}

// ---------------------------------------------------------------------------
// Helper: which phase does a task_index belong to?
// ---------------------------------------------------------------------------
int phase_of_task(int task_index, const std::vector<PhaseInfo>& phases)
{
    for (const auto& p : phases)
        if (task_index >= p.start_task && task_index < p.end_task)
            return p.phase_id;
    return -1;
}

bool is_high_variance_phase(int phase_id,
                             const std::vector<PhaseInfo>& phases,
                             double var_high_thresh)
{
    for (const auto& p : phases)
        if (p.phase_id == phase_id)
            return p.actual_variance > var_high_thresh;
    return false;
}

// ---------------------------------------------------------------------------
// compute_detection_lags
// ---------------------------------------------------------------------------
std::vector<DetectionLagResult> compute_detection_lags(
    std::vector<SchedulerDecision>& trace,
    const std::vector<PhaseInfo>&   phases,
    const ExperimentConfig&         cfg)
{
    std::vector<DetectionLagResult> results;
    if (phases.size() < 2) return results;

    // Annotate phase_id for every decision where it's unknown
    for (auto& d : trace) {
        if (d.phase_id_at_decision < 0)
            d.phase_id_at_decision = phase_of_task(d.task_start, phases);
    }

    // For each pair of adjacent phases there is one regime transition.
    for (int p = 0; p + 1 < (int)phases.size(); ++p) {

        int  boundary_task     = phases[p + 1].start_task;
        int  from_phase        = p;
        int  to_phase          = p + 1;
        bool new_is_high_var   = is_high_variance_phase(
                                     to_phase, phases, cfg.var_high_thresh);

        // Target chunk for the new regime
        double chunk_target = new_is_high_var
                              ? cfg.min_chunk
                              : cfg.max_chunk;

        // Find the chunk size active just before the boundary
        // (last decision whose task_end <= boundary_task)
        double chunk_prev = cfg.max_chunk;  // default if no prior decisions
        for (const auto& d : trace) {
            if (d.task_end <= boundary_task)
                chunk_prev = d.chunk_size;
        }

        double response_threshold =
            cfg.response_fraction * std::abs(chunk_target - chunk_prev);

        // Avoid trivial detections when chunk is already at target
        bool can_respond = (response_threshold > 0.0);

        int obs_lag     = -1;
        int action_lag  = -1;
        int resp_dec    = -1;

        for (auto& d : trace) {
            // Only consider decisions at or after the boundary
            if (d.task_start < boundary_task) continue;

            // ------------------------------------------------------------------
            // Observation lag: first decision where the window variance
            // has crossed into the new regime's territory.
            // window_variance is of task costs in [window_start_task, task_start).
            // ------------------------------------------------------------------
            if (obs_lag < 0) {
                bool window_reflects_new = false;
                if (d.window_n >= 2) {
                    if (new_is_high_var &&
                        d.window_variance > cfg.var_high_thresh)
                        window_reflects_new = true;
                    if (!new_is_high_var &&
                        d.window_variance < cfg.var_low_thresh)
                        window_reflects_new = true;
                }
                if (window_reflects_new)
                    obs_lag = d.task_start - boundary_task;
            }

            // ------------------------------------------------------------------
            // Action lag: first decision where chunk size has moved by
            // response_fraction toward the new-regime target.
            // ------------------------------------------------------------------
            if (action_lag < 0 && can_respond) {
                double movement = new_is_high_var
                    ? (chunk_prev - d.chunk_size)   // positive = decreased
                    : (d.chunk_size - chunk_prev);  // positive = increased
                if (movement >= response_threshold) {
                    action_lag = d.task_start - boundary_task;
                    resp_dec   = d.decision_idx;
                    // Annotate this decision in the trace
                    d.action_lag_tasks      = action_lag;
                    d.is_response_decision  = true;
                }
            }

            if (obs_lag >= 0) {
                // Also annotate the first observation-lag decision
                // (find it by index; it may have been skipped above)
                // We mark only action_lag decisions in-place above;
                // observation_lag goes into the DetectionLagResult only.
            }
        }

        // Backfill observation_lag_tasks on the first responding decision
        // (only if found)
        if (resp_dec >= 0) {
            for (auto& d : trace)
                if (d.decision_idx == resp_dec)
                    d.observation_lag_tasks = obs_lag;
        }

        DetectionLagResult res;
        res.transition_index        = p;
        res.from_phase              = from_phase;
        res.to_phase                = to_phase;
        res.boundary_task           = boundary_task;
        res.new_regime_high_variance= new_is_high_var;
        res.observation_lag_tasks   = obs_lag;
        res.action_lag_tasks        = action_lag;
        res.decision_index_responded= resp_dec;
        res.responded_within_window = (obs_lag >= 0 &&
                                       obs_lag <= cfg.window_size);
        results.push_back(res);
    }

    return results;
}
