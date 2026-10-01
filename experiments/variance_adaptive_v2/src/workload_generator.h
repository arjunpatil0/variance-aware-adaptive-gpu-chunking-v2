#pragma once
// workload_generator.h — Generates task-cost arrays with multiple phases.
//
// WHY: The workload is pre-generated before any scheduler runs so that all
// three scheduler arms (STATIC-LARGE, STATIC-MATCHED, ADAPTIVE) process the
// exact same sequence of task costs within each repetition.
//
// The adaptive scheduler is causal and never reads future task costs;
// the generator knowing the full array in advance is not a violation of
// that principle — it merely ensures comparability across arms.

#include "config.h"
#include <vector>
#include <string>
#include <random>    // std::mt19937, std::uniform_int_distribution, etc.

// ---------------------------------------------------------------------------
// Per-phase statistics recorded alongside the generated workload.
// Written to workload.csv and used by statistical_analysis for detection lag.
// ---------------------------------------------------------------------------
struct PhaseInfo {
    int    phase_id;
    std::string name;
    int    start_task;   // inclusive
    int    end_task;     // exclusive
    double actual_mean;
    double actual_variance;
    double actual_min;
    double actual_max;
    std::string distribution_desc;
};

// ---------------------------------------------------------------------------
// WorkloadGenerator
// ---------------------------------------------------------------------------
class WorkloadGenerator {
public:
    explicit WorkloadGenerator(const ExperimentConfig& cfg, unsigned int seed);

    // Generate task costs. Returns the cost array and per-phase statistics.
    // Deterministic: same (cfg, seed) always produces the same array.
    std::vector<int>      generate(std::vector<PhaseInfo>& phases_out);

    // Convenience: generate costs only (phases discarded).
    std::vector<int>      generate();

private:
    const ExperimentConfig& cfg_;
    unsigned int seed_;

    void fill_uniform  (std::vector<int>& costs, int from, int to,
                        int lo, int hi, std::mt19937& rng);
    void fill_bimodal  (std::vector<int>& costs, int from, int to,
                        int lo1, int hi1, int lo2, int hi2, std::mt19937& rng);
    void fill_lognormal(std::vector<int>& costs, int from, int to,
                        double mu, double sigma,
                        int clamp_min, int clamp_max, std::mt19937& rng);

    PhaseInfo compute_stats(const std::vector<int>& costs,
                            const PhaseConfig& pc, int phase_id) const;
};

// ---------------------------------------------------------------------------
// Helpers: build the four experiment presets
// ---------------------------------------------------------------------------
// Each preset populates cfg.phases and sets cfg.experiment_name.
// Call one of these before constructing WorkloadGenerator.

// Experiment A — Mostly uniform (all phases low variance)
void preset_experiment_A(ExperimentConfig& cfg);

// Experiment B — Gradually increasing variance
void preset_experiment_B(ExperimentConfig& cfg);

// Experiment C — Low → High → Low → Bursty  (primary test)
void preset_experiment_C(ExperimentConfig& cfg);

// Experiment D — Bursty / heavy-tailed throughout
void preset_experiment_D(ExperimentConfig& cfg);
