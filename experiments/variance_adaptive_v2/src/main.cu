// main.cu — Entry point for the variance_adaptive_v2 experiment.
//
// Compiles all source files through nvcc (delegates .cpp to MSVC).
//
// Usage:
//   variance_adaptive_v2.exe --calibrate [--results-dir DIR]
//   variance_adaptive_v2.exe --experiment [A|B|C|D] [options]
//
// Options:
//   --experiment A|B|C|D   Workload preset (default: C)
//   --tasks N              Total task count (default: 800000)
//   --reps N               Final repetitions (default: 20)
//   --warmup N             Warmup repetitions (default: 3)
//   --seed N               Base workload seed (default: 42)
//   --multiplier N         Cost multiplier override (skip calibration)
//   --results-dir DIR      Output directory base (default: results)
//   --calibrate            Run calibration sweep and exit

// Include all source files here so nvcc compiles everything as a SINGLE
// translation unit. This matches the old experiment's pattern (one .cu file)
// and avoids nvcc's separate compilation mode which uses a different CUDA
// runtime init path that triggers cudaErrorCallRequiresNewerDriver (801)
// on CUDA 13.1 toolkit + driver 555.97.
#include "cuda_executor.cu"
#include "workload_generator.cpp"
#include "statistical_analysis.cpp"
#include "result_serializer.cpp"
#include "experiment_runner.cpp"

#include "config.h"
#include "workload_generator.h"
#include "experiment_runner.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ---------------------------------------------------------------------------
// Simple argument parser
// ---------------------------------------------------------------------------
static std::string get_arg(int argc, char* argv[],
                            const std::string& flag,
                            const std::string& default_val = "")
{
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == flag)
            return argv[i + 1];
    return default_val;
}

static bool has_flag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    printf("variance_adaptive_v2 starting...\n");

    // Parse common options
    std::string exp_name   = get_arg(argc, argv, "--experiment", "C");
    int total_tasks        = std::stoi(get_arg(argc, argv, "--tasks",      "800000"));
    int num_final_reps     = std::stoi(get_arg(argc, argv, "--reps",       "20"));
    int num_warmup_reps    = std::stoi(get_arg(argc, argv, "--warmup",     "3"));
    int base_seed          = std::stoi(get_arg(argc, argv, "--seed",       "42"));
    std::string results_dir= get_arg(argc, argv, "--results-dir",          "results");
    bool do_calibrate      = has_flag(argc, argv, "--calibrate");
    bool has_multiplier    = has_flag(argc, argv, "--multiplier");
    int  forced_mult       = has_multiplier
                             ? std::stoi(get_arg(argc, argv, "--multiplier", "1"))
                             : 1;

    // Build config
    ExperimentConfig cfg;
    cfg.total_tasks      = total_tasks;
    cfg.workload_seed    = (unsigned int)base_seed;
    cfg.num_final_reps   = num_final_reps;
    cfg.num_warmup_reps  = num_warmup_reps;
    cfg.results_base_dir = results_dir;

    // Apply preset (populates phases and experiment_name)
    if      (exp_name == "A") preset_experiment_A(cfg);
    else if (exp_name == "B") preset_experiment_B(cfg);
    else if (exp_name == "C") preset_experiment_C(cfg);
    else if (exp_name == "D") preset_experiment_D(cfg);
    else {
        fprintf(stderr, "Unknown experiment '%s'. Use A, B, C, or D.\n",
                exp_name.c_str());
        return 1;
    }

    // Per-experiment output directory (avoids overwriting previous runs)
    std::string out_dir = results_dir + "/" + cfg.experiment_name;
    cfg.results_base_dir = results_dir;

    if (do_calibrate) {
        // Calibration mode: determine cost_multiplier and variance thresholds.
        // Must run before final experiments if no --multiplier override.
        ExperimentRunner calibrator(cfg, out_dir);
        calibrator.run_calibration();
        // Print the resulting multiplier so the user can pass --multiplier
        printf("\nRe-run with: --multiplier %d --experiment %s\n",
               calibrator.config().cost_multiplier, exp_name.c_str());
        return 0;
    }

    // Set cost multiplier (from calibration or override)
    if (has_multiplier) {
        cfg.cost_multiplier = forced_mult;
        // Default variance thresholds — suitable for the raw-cost distributions.
        // Users should run --calibrate first for best results.
        cfg.var_low_thresh  = 5000.0;
        cfg.var_high_thresh = 30000.0;
        printf("Using forced multiplier=%d with default thresholds.\n",
               forced_mult);
    } else {
        // Load calibration if it exists; otherwise use defaults
        cfg.cost_multiplier = 50;     // conservative default
        cfg.var_low_thresh  = 5000.0;
        cfg.var_high_thresh = 30000.0;
        printf("No --calibrate run detected; using default multiplier=%d "
               "and default thresholds. Run --calibrate for best results.\n",
               cfg.cost_multiplier);
    }

    if (has_flag(argc, argv, "--profile-occupancy")) {
        printf("Running occupancy profiling subset...\n");
        // Generate a small uniform workload just to have valid memory to read/write
        WorkloadGenerator gen(cfg, cfg.workload_seed);
        std::vector<PhaseInfo> dummy_phases;
        std::vector<int> task_costs = gen.generate(dummy_phases);
        CudaExecutor executor(task_costs, cfg.threads_per_block, cfg.cost_multiplier);

        int spread[] = {1000, 2000, 3000, 5000, 7500, 10000};
        for (int chunk : spread) {
            if (chunk < cfg.min_chunk || chunk > cfg.max_chunk) continue;
            for (int i = 0; i < 5; ++i) { // 5 samples per chunk size
                executor.launch_chunk(0, chunk);
            }
        }
        return 0;
    }

    if (cfg.num_final_reps < 20) {
        fprintf(stderr,
            "WARNING: --reps %d is below the minimum 20 required for "
            "valid statistical analysis.\n", cfg.num_final_reps);
    }

    printf("\nStarting experiment %s | tasks=%d | reps=%d | warmup=%d\n",
           cfg.experiment_name.c_str(), cfg.total_tasks,
           cfg.num_final_reps, cfg.num_warmup_reps);

    ExperimentRunner runner(cfg, out_dir);
    runner.run();

    return 0;
}
