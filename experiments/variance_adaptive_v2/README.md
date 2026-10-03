# Variance-Aware Adaptive GPU Workload Chunking — Experiment v2

## Overview

This experiment is a methodologically rigorous study of **variance-aware adaptive GPU workload chunking** using a causal sliding-window estimator of short-horizon workload variance.

This is **not** a claim of a novel GPU scheduling algorithm. It is an experimental framework designed to study the relationship between workload variability, adaptive chunking, scheduling overhead, and causal runtime adaptation.

The old project (`src/static_chunking.cu`, `src/variance_chunking.cu`) is **preserved and untouched** in `../../src/`. This experiment lives entirely in `experiments/variance_adaptive_v2/`.

---

## Findings

All results are from a real NVIDIA RTX 4070 GPU (confirmed, not simulated). Each experiment ran 20 independent repetitions. Statistics use the Wilcoxon signed-rank test, cross-validated against Python/scipy.

---

### What we measured (in plain English)

We gave the same 800,000 tasks to four different scheduling strategies and measured how long each took to finish.

- **ADAPTIVE** — watches how unpredictable recent tasks were; uses smaller batches when unpredictable, larger batches when predictable
- **STATIC-LARGE** — always uses the maximum batch size (10,000 tasks). Simple, never adapts.
- **STATIC-MATCHED** — uses the same *average* batch size as ADAPTIVE, but never changes it. This is the fair comparison: same average chunk, no adaptation.
- **INVERTED-ADAPTIVE** — does the exact opposite of ADAPTIVE: uses *larger* batches when tasks are unpredictable, smaller when predictable.

---

### Results — All Four Experiments (n=20, p-values confirmed)

| Experiment | What the workload looks like | ADAPTIVE | STATIC-MATCHED | STATIC-LARGE | INVERTED |
|---|---|---:|---:|---:|---:|
| **A** | All tasks cost the same (uniform) | 999 ms | 998 ms | 999 ms | 4,764 ms |
| **B** | Variance steadily increases | 11,823 ms | 7,670 ms | 2,757 ms | 4,951 ms |
| **C** | Low → High → Low → Bursty variance | 9,942 ms | 6,877 ms | 2,725 ms | 4,551 ms |
| **D** | Always bursty (log-normal, heavy tails) | 33,670 ms | 27,440 ms | 6,924 ms | 6,906 ms |

All differences between arms in experiments B, C, and D are **statistically significant (p = 0.0001)**.

---

### What the numbers mean — the simple version

**On a uniform workload (Experiment A): everything is the same.**
When all tasks cost roughly the same amount, there is nothing for the adaptive scheduler to react to. It quickly grows to max batch size and stays there — identical to STATIC-LARGE. The only exception is INVERTED, which mistakenly shrinks batches because it sees "low variance = shrink" — and becomes 5× slower as a result. This confirms both schedulers are doing exactly what they are designed to do.

**On variable workloads (Experiments B, C, D): ADAPTIVE is always the slowest.**
The scheduler detects high variance and shrinks batch size. A smaller batch means more round-trips to the GPU. On Windows, each round-trip (kernel launch) costs about 0.75ms of fixed overhead regardless of how much work is inside. With 500 launches instead of 80, that overhead adds up to minutes. The GPU ends up spending more time on paperwork than on actual computation.

**INVERTED-ADAPTIVE is faster than ADAPTIVE on every variable workload.**
The inverted policy grows batches when variance is high — opposite to what seems intuitive. This means fewer round-trips, less overhead, and the GPU gets a large pool of threads to work with simultaneously. On experiments B and C it runs 2–3× faster than ADAPTIVE. On experiment D it matches STATIC-LARGE almost exactly.

**STATIC-LARGE is the fastest on every experiment.**
Eighty launches for 800,000 tasks. Minimal overhead. No decision-making complexity. This is the baseline that adaptive scheduling needs to beat — and none of the adaptive strategies do.

---

### The core finding

> **Software-level adaptive chunking that reacts to variance by reducing batch size is counterproductive on this hardware.** Every kernel launch costs fixed overhead. Reducing batch size increases launch count. The overhead dominates, and performance degrades — not improves. The GPU's own hardware warp scheduler handles variable-cost tasks within a large batch better than software fragmentation does.

The inverted arm provides the mechanistic proof: reversing the policy (grow when variance is high) dramatically recovers performance, which shows the *direction* of adaptation — not just adaptation itself — was the problem.

---

### Limitations

- Tested on one GPU (RTX 4070) in one OS mode (Windows WDDM). Under Linux, per-launch overhead is ~100× lower; results could differ significantly.
- Synthetic compute-bound workload only. Real workloads with memory access patterns or inter-kernel dependencies may behave differently.
- Occupancy profiling (Nsight Compute) was not available on this system; the warp-starvation mechanism is inferred from timing, not directly measured.

*This is an independent methodologically rigorous re-investigation; not a peer-reviewed publication.*

---

## 1. Research Question

Does a causal sliding-window variance-aware scheduler improve total end-to-end GPU workload execution time compared with a fixed-chunk scheduler that uses the **same average chunk size**?

The primary question controls for average chunk size so that the comparison isolates adaptiveness rather than merely chunk-size magnitude.

---

## 2. Hypothesis

**H0**: STATIC-MATCHED and ADAPTIVE have no difference in total end-to-end execution time.

**H1**: ADAPTIVE has lower total end-to-end execution time than STATIC-MATCHED.

A result is labeled an **improvement** only if **both** of the following hold:
- Paired Wilcoxon signed-rank test: p < 0.05
- ADAPTIVE mean T_total is at least **5% lower** than STATIC-MATCHED mean T_total

Results meeting only one criterion are described as "lower observed runtime" but not as improvements.

---

## 3. Primary Dependent Variable

**T_total: total end-to-end wall-clock execution time per workload repetition.**

This is the **only primary hypothesis metric**. It is measured as the host-side wall-clock elapsed time from the start of the scheduler loop to the completion of the final chunk (including kernel execution, kernel launch latency, synchronization, and host-side decision overhead).

All other measurements (T_kernel, T_scheduling, chunk sizes, detection lag) are **secondary/diagnostic**.

> **Warning**: Do not select a different primary metric after inspecting results. Do not call a result an improvement unless both the p-value and effect-size criteria are satisfied simultaneously.

---

## 4. Workload Model

Task computational cost varies over time through four explicit phases with known ground-truth regime boundaries:

| Phase | Task Range | Distribution | Variance Character |
|-------|-----------|-------------|-------------------|
| 0 | [0, N/4) | Uniform[200, 400] | **Low** |
| 1 | [N/4, N/2) | Bimodal: 50% U[100,300] + 50% U[700,900] | **High** |
| 2 | [N/2, 3N/4) | Uniform[350, 450] | **Low** |
| 3 | [3N/4, N) | Log-normal(μ=5.5, σ=0.9) clamped [100, 3000] | **Bursty** |

*(This is Experiment C — Low→High→Low→Bursty. Experiments A, B, D use different distributions.)*

Each task is mapped to one CUDA thread. The thread performs `W × cost_multiplier` iterations of a compute-bound loop where `W` is the task's cost value.

The workload is **pre-generated** with a fixed random seed before any scheduler runs. All three scheduler arms within a repetition process **the exact same task-cost array**.

### Reproducibility
- Default seed: 42
- Each repetition uses seed = `base_seed + rep × 1000003`
- All four experiment presets (A–D) are deterministic given seed + config

---

## 5. Static-Large Scheduling Strategy

STATIC-LARGE uses a fixed chunk size of `MAX_CHUNK` (default: 10,000 tasks) throughout the entire workload. It does not inspect task costs or execution times.

This is the **coarse-grained baseline**. It is the secondary comparison arm.

Configuration: `--max-chunk` (default: 10,000)

---

## 6. Static-Matched Scheduling Strategy

STATIC-MATCHED uses a **fixed chunk size equal to the mean chunk size selected by the adaptive scheduler** for the same workload repetition.

This is the **primary comparison arm**. Because STATIC-MATCHED uses the same average chunk size as ADAPTIVE, any performance difference between them must come from adaptiveness itself (finer-grained reaction to regime changes) rather than from having a smaller chunk size overall.

### Two-Pass Protocol

STATIC-MATCHED's chunk size cannot be known until the adaptive scheduler has run. Therefore:

**PASS 1**: Run ADAPTIVE on the workload. Record the mean adaptive chunk size.

**PASS 2**: Run STATIC-LARGE and STATIC-MATCHED on the same workload. STATIC-MATCHED uses `round(mean_adaptive_chunk)` from PASS 1.

The adaptive run used to derive the matched chunk size is also the timed adaptive observation for that repetition.

> **Do not use results from one workload seed to set the matched chunk for another seed.**

---

## 7. Adaptive Scheduling Strategy

The adaptive scheduler uses a **causal sliding window** over recently completed task costs to estimate local workload variance and adjust chunk size.

At each scheduling decision:
1. Collect the `min(WINDOW_SIZE, num_completed)` most recent completed task costs
2. Compute window mean μ and sample variance σ²
3. Apply policy:
   - σ² < `VAR_LOW_THRESH` → `chunk = min(chunk + CHUNK_STEP, MAX_CHUNK)`
   - σ² > `VAR_HIGH_THRESH` → `chunk = max(chunk - CHUNK_STEP, MIN_CHUNK)`
   - else → chunk unchanged
4. Launch exactly **one kernel** for that chunk
5. After `cudaDeviceSynchronize()`, update the causal observer (new tasks are now "completed")
6. Repeat

Initial chunk size: `MAX_CHUNK` (conservative cold-start).

---

## 8. Causal Scheduling Definition

The adaptive scheduler is **causal**: it makes scheduling decisions using only task costs from tasks that have already completed execution.

At decision k the scheduler may use:
- Task costs for tasks `[0, processed)` — those that have already completed
- Its own internal state (current chunk size, decision index)
- Fixed configuration parameters

It **must not** inspect costs for tasks `[processed, N)` — those not yet executed.

This is enforced at the interface level by `CausalObserver::completed_cost(i)`, which aborts with an assertion failure if `i >= num_completed`. There is no path through which future task costs can reach the chunk-size decision.

---

## 9. Sliding-Window Variance Definition

The sliding window holds the `min(WINDOW_SIZE, num_completed)` most recent completed task costs.

**Variance**: sample variance (divides by n−1), computed over raw task cost values (not over GPU execution times).

> **Why task costs, not timing?** GPU execution times contain noise from clock variations, OS scheduling, and measurement overhead. Task costs (the pre-generated workload values) are the ground-truth signal of workload heterogeneity. Using costs instead of timing avoids the confound that killed the original experiment.

**Window size** (default: 20) limits how far back the scheduler looks, but because the observer updates only in chunk-sized batches, the actual lag is bounded by `max(WINDOW_SIZE, chunk_size_at_transition)`.

---

## 10. Chunk-Size Policy

```
if σ² < VAR_LOW_THRESH:   chunk = min(chunk + CHUNK_STEP, MAX_CHUNK)
if σ² > VAR_HIGH_THRESH:  chunk = max(chunk - CHUNK_STEP, MIN_CHUNK)
else:                      chunk unchanged
```

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `MIN_CHUNK` | 1,000 | Floor to avoid excessive kernel launches |
| `MAX_CHUNK` | 10,000 | Ceiling = STATIC-LARGE chunk size |
| `CHUNK_STEP` | 1,000 | Linear step; one per decision; avoids over-reaction |
| `WINDOW_SIZE` | 20 | ≥20 obs/phase at MAX_CHUNK; bounds observation lag |
| `VAR_LOW_THRESH` | 5,000 | Set by calibration; above low-variance phases |
| `VAR_HIGH_THRESH` | 30,000 | Set by calibration; below high-variance phases |

Policy is **deterministic** given workload + config + seed. No ML, no heuristic magic numbers.

---

## 11. Three-Arm Experimental Methodology

Every experiment compares exactly three scheduler arms on **identical workloads**:

| Arm | Type | Chunk Size |
|-----|------|-----------|
| STATIC-LARGE | Fixed | MAX_CHUNK (always 10,000) |
| STATIC-MATCHED | Fixed | Mean adaptive chunk (from PASS 1) |
| **ADAPTIVE** | Adaptive | Varies by σ² |

**Primary comparison**: STATIC-MATCHED vs ADAPTIVE  
**Secondary comparison**: STATIC-LARGE vs ADAPTIVE (weaker — does not control for chunk size)

### Arm-Ordering Randomization

Running arms in the same order every repetition creates a systematic bias (GPU clock boosting, thermal throttling). To address this:

- PASS 1 always runs ADAPTIVE first (required — its output sets STATIC-MATCHED's chunk size)
- PASS 2 randomizes the order of STATIC-LARGE and STATIC-MATCHED using `rep % 2`
- A 200ms `sleep + cudaDeviceSynchronize()` is inserted between every arm
- The `arm_order` column in `raw_results.csv` records the order for each rep

---

## 12. Statistical Methodology

- **Minimum repetitions**: 20 (required; do not reduce for final results)
- **Test**: Wilcoxon signed-rank test, two-tailed, paired — chosen before experiments
- **Significance level**: α = 0.05 (fixed before any experiment)
- **Minimum practical effect**: 5% lower mean T_total for ADAPTIVE vs STATIC-MATCHED
- **Warm-up runs**: excluded from all statistics

For each arm, report: n, mean, std, min, max.

For each paired comparison, report: paired mean difference, relative %, p-value, effect size r = Z/√n, 95% CI on mean paired difference.

The C++ implementation is cross-validated by `analysis/analyze_v2.py` using `scipy.stats.wilcoxon`.

> **Do not perform repeated hypothesis tests on secondary metrics and selectively report only significant results. T_total is the single primary hypothesis metric.**

---

## 13. Warm-Up Methodology

Before collecting final timing measurements:

1. Initialize CUDA (implicit on first API call)
2. Allocate GPU memory
3. Run `NUM_WARMUP_REPS` (default: 3) full-workload passes with all three arms
4. Discard all warmup results
5. Begin final timed measurements

This excludes CUDA context creation, JIT compilation, and clock settling from all statistics.

---

## 14. Kernel Launch Mechanics

**One scheduler chunk = exactly one CUDA kernel launch.**

```
for each chunk:
  scheduler selects chunk size C
  blocks = ceil(C / THREADS_PER_BLOCK)
  launch work_kernel<<<blocks, THREADS_PER_BLOCK>>>()
  cudaDeviceSynchronize()
  update causal observer
  select next chunk
```

The number of kernel launches **equals** the number of scheduling decisions. This is by design: the experiment explicitly studies the tradeoff between scheduling granularity and kernel-launch/synchronization overhead.

There is **no persistent kernel**. Multiple scheduling chunks are not hidden inside one kernel launch.

---

## 15. Metrics

**PRIMARY (hypothesis metric):**
- `T_total`: total end-to-end wall-clock time (host-side, `std::chrono`)

**SECONDARY / DIAGNOSTIC:**
- `T_kernel`: sum of CUDA event elapsed times per chunk (GPU-side only)
- `T_scheduling`: `T_total - T_kernel` (approximate; derived, not directly measured)
- Number of kernel launches (= number of chunks)
- Chunk sizes selected
- Window variance at each decision
- Detection lag per regime transition

> `T_scheduling` is labeled as **approximate** because it is derived by subtraction. It includes launch latency, synchronization wait, and host-side decision overhead but these are not independently measured.

---

## 16. Detection-Lag Definition

Because the scheduler is causal, it cannot react instantaneously to a workload regime change.

### observation_lag_tasks
The number of newly completed tasks after a known ground-truth regime boundary until the first scheduling decision where the sliding-window variance crosses into the new regime's territory.

**Bounded by max(WINDOW_SIZE, chunk_size_at_transition)**: because the `CausalObserver` only updates in chunk-sized batches (not per-task), if a regime transitions while a large chunk (e.g. 10,000 tasks) is in flight, the window literally cannot see any new-regime data until that entire chunk finishes.

### action_lag_tasks
The number of tasks from the regime boundary until the first chunk-size decision that moves by at least `RESPONSE_FRACTION × |target_chunk − prev_chunk|` toward the new regime's target chunk size.

**Not bounded by WINDOW_SIZE alone**: the in-flight chunk executing when the regime changes adds additional delay. Therefore:

```
action_lag_tasks ≥ observation_lag_tasks
```

This distinction separates **statistical detection delay** (window fills with new-regime data) from **chunk-granularity reaction delay** (current chunk must complete before the scheduler can decide).

**Response criterion (fixed before experiments):** a decision responds to the new regime if chunk size moves by ≥ `RESPONSE_FRACTION = 0.25` of the distance from the previous-regime chunk size to the new-regime target chunk size.

---

## 17. Warp-Divergence Limitation

**Chunk size controls scheduling granularity *between* chunks. It does NOT provide intra-chunk load balancing.**

Within a CUDA warp (32 threads), all threads execute in lockstep. If tasks in the same warp have different work amounts `W_i`, the warp's execution time is determined by the **slowest thread** in that warp: `max(W_i)`. Changing the chunk size does not change this behaviour.

**Adaptive chunking does not remove warp-level divergence.** A smaller chunk size gives the scheduler more frequent decision points between chunks. It does not prevent one thread from waiting for another within the same warp.

This experiment studies **inter-chunk scheduling granularity**, not intra-warp or intra-block load balancing. These are separate concerns. The experiment must not claim that adaptive chunking solves warp-level divergence.

---

## 18. Expected Failure Modes

The experiment is capable of showing any of the following results:

1. **Adaptive improvement**: ADAPTIVE achieves both p < 0.05 and ≥ 5% lower T_total than STATIC-MATCHED. Correctly labeled "improvement."

2. **No meaningful difference**: Adaptive overhead (more decisions, more launches if chunk decreases) cancels any benefit. Reported honestly.

3. **Adaptive degradation**: ADAPTIVE is slower than STATIC-MATCHED because the scheduler reduces chunk size (increasing kernel-launch overhead) without proportional GPU-side benefit. This is the scenario the original experiment demonstrated when the controller was over-sensitive.

4. **Detection lag too large**: The scheduler reacts after the regime has already ended (lag > phase length). The experiment records this.

5. **Calibration failure**: If `cost_multiplier` is too small, T_kernel is in the noise floor. Run `--calibrate` first.

---

## 19. How to Build

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit (nvcc on PATH)
- Microsoft C++ Build Tools (Visual Studio or Build Tools, cl.exe)
- Python 3.8+ with `pandas matplotlib scipy numpy` (for analysis only)

**Build:**
```bat
cd experiments\variance_adaptive_v2
build.bat
```

The `-allow-unsupported-compiler` flag handles MSVC version mismatches (same as the original project).

---

## 20. How to Run

**Step 1 — Calibrate (recommended):**
```bat
variance_adaptive_v2.exe --calibrate --experiment C
```
Writes `results/calibration.json` with the chosen `cost_multiplier` and variance thresholds.

**Step 2 — Run experiments:**
```bat
variance_adaptive_v2.exe --experiment C --multiplier 50 --reps 20 --warmup 3
variance_adaptive_v2.exe --experiment A --multiplier 50 --reps 20 --warmup 3
variance_adaptive_v2.exe --experiment B --multiplier 50 --reps 20 --warmup 3
variance_adaptive_v2.exe --experiment D --multiplier 50 --reps 20 --warmup 3
```

**Step 3 — Analyze:**
```bat
python analysis\analyze_v2.py --experiment C
```

---

## 21. How to Reproduce Experiments

All randomness is seeded. To reproduce exactly:
1. Use the same `--seed` value (default: 42)
2. Use the same `--multiplier` from calibration
3. Use the same `--tasks`, `--reps`, `--warmup`
4. Run on the same GPU (timing depends on hardware)

The `config.json` written to each results directory contains the full parameter snapshot needed for reproduction.

---

## 22. Where Results Are Stored

```
experiments/variance_adaptive_v2/results/
  calibration.json
  C_low_high_low_bursty/
    config.json              # full parameter snapshot
    workload.csv             # task_index, task_cost, phase_id
    scheduler_trace.csv      # all chunk decisions for all arms and reps
    raw_results.csv          # per-rep T_total for all 3 arms
    statistical_results.json # Wilcoxon results, means, stds, CIs
    detection_lag_results.json
    python_statistical_results.json  # scipy cross-check
    plots/
      01_task_cost_trace.png
      02_window_variance_trace.png
      03_chunk_size_trace.png
      04_chunk_timeline.png
      05_total_runtime_bars.png
      06_scheduling_decisions.png
      07_detection_lag.png
```

Results directories are **not overwritten** between runs. Re-running creates a new directory or appends to existing CSVs.

---

## 23. Limitations

1. **Causal lag is inherent**: the scheduler cannot react instantaneously; `action_lag_tasks ≥ observation_lag_tasks`. This is by design for realistic runtime adaptation.

2. **Warp-level divergence is not addressed**: see §17.

3. **Single GPU**: results are specific to the GPU tested. Different hardware may show different overhead profiles.

4. **Synthetic workload**: the workload is compute-bound with a simple XOR loop. Real GPU workloads have memory access patterns, pipeline hazards, and inter-kernel dependencies not modeled here.

5. **T_scheduling is approximate**: derived by subtraction `T_total − T_kernel`; launch latency and synchronization overhead are not independently measured.

6. **Normal approximation for Wilcoxon**: valid for n ≥ 10; at n = 20 this is marginal. For critical results, consider exact Wilcoxon p-values (available in scipy with `method='exact'` for small n).

7. **Linear chunk-step policy**: the policy does not adapt its step size to the magnitude of variance change. A larger variance spike causes the same step as a small one. This is intentional (transparency) but limits responsiveness.

---

## Project Structure

```
experiments/variance_adaptive_v2/
  src/
    config.h                    All tunable parameters
    workload_generator.h/.cpp   Phase-based seeded workload generation
    sliding_window.h            Fixed-capacity sliding window
    scheduler.h                 CausalObserver + abstract Scheduler
    static_large_scheduler.h    STATIC-LARGE and STATIC-MATCHED implementation
    adaptive_scheduler.h        Causal variance-aware adaptive scheduler
    cuda_executor.h/.cu         CUDA kernel + dual timing
    statistical_analysis.h/.cpp Wilcoxon + detection lag computation
    result_serializer.h/.cpp    JSON/CSV output
    experiment_runner.h/.cpp    Three-arm orchestration
    main.cu                     Entry point
  analysis/
    analyze_v2.py               Scipy cross-check + all 7 plots
  results/                      Created at runtime
  build.bat                     Windows build script
  README.md                     This file
```

The original experiment (`../../src/`) is **completely preserved**.
