// workload_generator.cpp — Implementation of WorkloadGenerator and presets.
//
// Phase distributions are chosen to produce clearly distinct variance regimes:
//   Uniform[a,b]  → σ² = (b-a)²/12  (low for narrow ranges)
//   Bimodal       → σ² large (two well-separated modes)
//   Log-normal    → heavy tail, high effective variance
//
// The exact same generated cost array is reused by all three scheduler arms
// within a repetition (reproducibility).

#include "workload_generator.h"
#include <random>
#include <cassert>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
WorkloadGenerator::WorkloadGenerator(const ExperimentConfig& cfg, unsigned int seed)
    : cfg_(cfg), seed_(seed)
{
    if (cfg_.phases.empty())
        throw std::runtime_error("ExperimentConfig.phases must be non-empty. "
                                 "Call a preset_experiment_X() first.");
}

// ---------------------------------------------------------------------------
// generate() — produces the task-cost array
// ---------------------------------------------------------------------------
std::vector<int> WorkloadGenerator::generate(std::vector<PhaseInfo>& phases_out)
{
    // One RNG seeded deterministically.
    // We advance the RNG phase-by-phase in the same order every call.
    std::mt19937 rng(seed_);

    std::vector<int> costs(cfg_.total_tasks, 0);
    phases_out.clear();

    for (int p = 0; p < (int)cfg_.phases.size(); ++p) {
        const PhaseConfig& pc = cfg_.phases[p];

        assert(pc.start_task >= 0 && pc.end_task <= cfg_.total_tasks);
        assert(pc.start_task < pc.end_task);

        switch (pc.distribution) {
            case DistributionKind::Uniform:
                fill_uniform(costs, pc.start_task, pc.end_task,
                             (int)pc.param1, (int)pc.param2, rng);
                break;

            case DistributionKind::Bimodal:
                fill_bimodal(costs, pc.start_task, pc.end_task,
                             (int)pc.param1, (int)pc.param2,
                             (int)pc.param3, (int)pc.param4, rng);
                break;

            case DistributionKind::LogNormal:
                fill_lognormal(costs, pc.start_task, pc.end_task,
                               pc.param1, pc.param2,
                               pc.clamp_min, pc.clamp_max, rng);
                break;

            default:
                throw std::runtime_error("Unknown DistributionKind");
        }

        phases_out.push_back(compute_stats(costs, pc, p));
    }

    return costs;
}

std::vector<int> WorkloadGenerator::generate()
{
    std::vector<PhaseInfo> dummy;
    return generate(dummy);
}

// ---------------------------------------------------------------------------
// Private fill helpers
// ---------------------------------------------------------------------------

void WorkloadGenerator::fill_uniform(std::vector<int>& costs,
                                     int from, int to,
                                     int lo, int hi,
                                     std::mt19937& rng)
{
    // WHY uniform_int_distribution rather than (rand % range): avoids modulo
    // bias and is reproducible across platforms with the same seed.
    std::uniform_int_distribution<int> dist(lo, hi);
    for (int i = from; i < to; ++i)
        costs[i] = dist(rng);
}

void WorkloadGenerator::fill_bimodal(std::vector<int>& costs,
                                     int from, int to,
                                     int lo1, int hi1,
                                     int lo2, int hi2,
                                     std::mt19937& rng)
{
    // WHY bimodal: creates high variance without a simple unimodal spread.
    // 50% of tasks are drawn from the low mode, 50% from the high mode.
    // This mimics workloads with two distinct task classes (e.g., read vs.
    // write, cached vs. uncached) and produces σ² ≈ ((hi2-lo2)/4)² + gap²/4.
    std::uniform_int_distribution<int> low_mode(lo1, hi1);
    std::uniform_int_distribution<int> high_mode(lo2, hi2);
    std::bernoulli_distribution coin(0.5);

    for (int i = from; i < to; ++i)
        costs[i] = coin(rng) ? low_mode(rng) : high_mode(rng);
}

void WorkloadGenerator::fill_lognormal(std::vector<int>& costs,
                                       int from, int to,
                                       double mu, double sigma,
                                       int clamp_min, int clamp_max,
                                       std::mt19937& rng)
{
    // WHY log-normal: models bursty/heavy-tailed workloads where most tasks
    // are inexpensive but rare tasks are very expensive — realistic for
    // workloads with occasional cache misses, branch mispredictions, etc.
    std::lognormal_distribution<double> dist(mu, sigma);

    for (int i = from; i < to; ++i) {
        int v = static_cast<int>(std::round(dist(rng)));
        v = std::max(clamp_min, std::min(clamp_max, v));
        costs[i] = v;
    }
}

// ---------------------------------------------------------------------------
// Compute per-phase statistics for metadata / detection-lag reference
// ---------------------------------------------------------------------------
PhaseInfo WorkloadGenerator::compute_stats(const std::vector<int>& costs,
                                           const PhaseConfig& pc,
                                           int phase_id) const
{
    PhaseInfo info;
    info.phase_id   = phase_id;
    info.name       = pc.name;
    info.start_task = pc.start_task;
    info.end_task   = pc.end_task;

    int n = pc.end_task - pc.start_task;
    double sum = 0.0, sum_sq = 0.0;
    int lo = INT_MAX, hi = INT_MIN;

    for (int i = pc.start_task; i < pc.end_task; ++i) {
        double v = costs[i];
        sum    += v;
        sum_sq += v * v;
        lo = std::min(lo, costs[i]);
        hi = std::max(hi, costs[i]);
    }

    double mean = sum / n;
    // Population variance (we own the full phase, not a sample)
    double var  = sum_sq / n - mean * mean;

    info.actual_mean     = mean;
    info.actual_variance = var;
    info.actual_min      = lo;
    info.actual_max      = hi;

    switch (pc.distribution) {
        case DistributionKind::Uniform:
            info.distribution_desc = "Uniform[" + std::to_string((int)pc.param1)
                                   + "," + std::to_string((int)pc.param2) + "]";
            break;
        case DistributionKind::Bimodal:
            info.distribution_desc = "Bimodal(U["
                + std::to_string((int)pc.param1) + "," + std::to_string((int)pc.param2)
                + "]+U[" + std::to_string((int)pc.param3) + ","
                + std::to_string((int)pc.param4) + "])";
            break;
        case DistributionKind::LogNormal:
            info.distribution_desc = "LogNormal(mu=" + std::to_string(pc.param1)
                + ",sigma=" + std::to_string(pc.param2)
                + ",clamp=[" + std::to_string(pc.clamp_min)
                + "," + std::to_string(pc.clamp_max) + "])";
            break;
        default:
            info.distribution_desc = "Unknown";
    }

    return info;
}

// ---------------------------------------------------------------------------
// Experiment presets
// ---------------------------------------------------------------------------
// Phase boundaries are derived from total_tasks; if total_tasks is changed
// via CLI the boundaries scale accordingly.  The preset sets them to equal
// quarters; the caller may override after calling the preset if desired.

static void set_equal_phase_boundaries(ExperimentConfig& cfg)
{
    int N = cfg.total_tasks;
    int q = N / 4;
    // Adjust to make sure phases cover all N tasks (handle rounding)
    for (int i = 0; i < 4; ++i) {
        cfg.phases[i].start_task = i * q;
        cfg.phases[i].end_task   = (i == 3) ? N : (i + 1) * q;
    }
}

// -- Experiment A: Mostly uniform (all phases low variance) -----------------
// WHY: Establishes that the adaptive scheduler does NOT aggressively change
// chunk sizes when the workload is genuinely stable.
void preset_experiment_A(ExperimentConfig& cfg)
{
    cfg.experiment_name = "A_uniform";
    cfg.phases.resize(4);
    // Phase 0: Uniform[250,350] σ²=(100²/12)≈833
    cfg.phases[0] = {"A0_uniform_low",  0, 0, DistributionKind::Uniform,
                     250, 350, 0, 0, 0, 0};
    // Phase 1: Uniform[280,380] similarly low
    cfg.phases[1] = {"A1_uniform_low",  0, 0, DistributionKind::Uniform,
                     280, 380, 0, 0, 0, 0};
    // Phase 2: Uniform[260,360]
    cfg.phases[2] = {"A2_uniform_low",  0, 0, DistributionKind::Uniform,
                     260, 360, 0, 0, 0, 0};
    // Phase 3: Uniform[270,370]
    cfg.phases[3] = {"A3_uniform_low",  0, 0, DistributionKind::Uniform,
                     270, 370, 0, 0, 0, 0};
    set_equal_phase_boundaries(cfg);
}

// -- Experiment B: Gradually increasing variance ----------------------------
// WHY: Tests whether the scheduler detects a monotonically drifting workload.
void preset_experiment_B(ExperimentConfig& cfg)
{
    cfg.experiment_name = "B_increasing_variance";
    cfg.phases.resize(4);
    // Phase 0: Uniform[300,400] σ²≈833
    cfg.phases[0] = {"B0_low_var",   0, 0, DistributionKind::Uniform,
                     300, 400, 0, 0, 0, 0};
    // Phase 1: Uniform[200,500] σ²≈7500
    cfg.phases[1] = {"B1_med_var",   0, 0, DistributionKind::Uniform,
                     200, 500, 0, 0, 0, 0};
    // Phase 2: Bimodal U[100,300]+U[700,900] σ² very high
    cfg.phases[2] = {"B2_high_var",  0, 0, DistributionKind::Bimodal,
                     100, 300, 700, 900, 0, 0};
    // Phase 3: LogNormal bursty
    cfg.phases[3] = {"B3_bursty",    0, 0, DistributionKind::LogNormal,
                     5.5, 0.9, 0, 0, 100, 3000};
    set_equal_phase_boundaries(cfg);
}

// -- Experiment C: Low → High → Low → Bursty  (primary experiment) ----------
// WHY: The alternating structure creates multiple detectable regime transitions
// so that detection-lag can be measured for both low-to-high and high-to-low.
void preset_experiment_C(ExperimentConfig& cfg)
{
    cfg.experiment_name = "C_low_high_low_bursty";
    cfg.phases.resize(4);
    // Phase 0: Uniform[200,400]  σ²=(200²/12)≈3333  — Low variance
    cfg.phases[0] = {"C0_low",    0, 0, DistributionKind::Uniform,
                     200, 400, 0, 0, 0, 0};
    // Phase 1: Bimodal U[100,300]+U[700,900]  σ²≈100000 — High variance
    //   mean≈500, two modes separated by 400
    cfg.phases[1] = {"C1_high",   0, 0, DistributionKind::Bimodal,
                     100, 300, 700, 900, 0, 0};
    // Phase 2: Uniform[350,450]  σ²=(100²/12)≈833 — Low variance again
    cfg.phases[2] = {"C2_low",    0, 0, DistributionKind::Uniform,
                     350, 450, 0, 0, 0, 0};
    // Phase 3: LogNormal(mu=5.5,sigma=0.9) clamped [100,3000] — Bursty
    cfg.phases[3] = {"C3_bursty", 0, 0, DistributionKind::LogNormal,
                     5.5, 0.9, 0, 0, 100, 3000};
    set_equal_phase_boundaries(cfg);
}

// -- Experiment D: Bursty throughout ----------------------------------------
// WHY: Tests worst-case adaptive behavior when variance is consistently high.
void preset_experiment_D(ExperimentConfig& cfg)
{
    cfg.experiment_name = "D_bursty";
    cfg.phases.resize(4);
    // All phases use log-normal with varying parameters
    cfg.phases[0] = {"D0_burst_mild",  0, 0, DistributionKind::LogNormal,
                     5.0, 0.7, 0, 0, 100, 2000};
    cfg.phases[1] = {"D1_burst_heavy", 0, 0, DistributionKind::LogNormal,
                     5.5, 1.2, 0, 0, 100, 5000};
    cfg.phases[2] = {"D2_burst_mild",  0, 0, DistributionKind::LogNormal,
                     5.0, 0.7, 0, 0, 100, 2000};
    cfg.phases[3] = {"D3_burst_heavy", 0, 0, DistributionKind::LogNormal,
                     5.5, 1.2, 0, 0, 100, 5000};
    set_equal_phase_boundaries(cfg);
}
