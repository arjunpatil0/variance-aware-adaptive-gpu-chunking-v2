# Interview Preparation: Complete Technical Deep Dive

**How to use this document:** Read it front to back. Every section builds on the previous one. By the end, you will be able to reconstruct every concept, every decision, and every piece of code in this project from memory — and explain it to someone who has never seen a GPU.

---

## Table of Contents

- [Part I: Concepts From Scratch](#part-i-concepts-from-scratch)
  - [1. What is a GPU and why does it exist?](#1-what-is-a-gpu-and-why-does-it-exist)
  - [2. What is CUDA?](#2-what-is-cuda)
  - [3. What is a thread, a warp, a block, and a grid?](#3-what-is-a-thread-a-warp-a-block-and-a-grid)
  - [4. What is a kernel launch and why does it cost time?](#4-what-is-a-kernel-launch-and-why-does-it-cost-time)
  - [5. What is chunking and why is it needed?](#5-what-is-chunking-and-why-is-it-needed)
  - [6. What is variance (the math)?](#6-what-is-variance-the-math)
  - [7. What is a sliding window?](#7-what-is-a-sliding-window)
  - [8. What is a causal system?](#8-what-is-a-causal-system)
  - [9. What is a Wilcoxon signed-rank test?](#9-what-is-a-wilcoxon-signed-rank-test)
  - [10. What is warp divergence?](#10-what-is-warp-divergence)
- [Part II: The Problem Statement](#part-ii-the-problem-statement)
  - [11. The problem in one paragraph](#11-the-problem-in-one-paragraph)
  - [12. Why this problem matters (real-world use cases)](#12-why-this-problem-matters-real-world-use-cases)
  - [13. What existed before this project?](#13-what-existed-before-this-project)
- [Part III: Resume Line-by-Line Breakdown](#part-iii-resume-line-by-line-breakdown)
  - [14. Project 1: "Causal GPU Scheduling Framework..."](#14-project-1-causal-gpu-scheduling-framework)
  - [15. Project 2: "Variance-Aware Adaptive GPU Scheduler"](#15-project-2-variance-aware-adaptive-gpu-scheduler)
- [Part IV: How Every Piece Was Built](#part-iv-how-every-piece-was-built)
  - [16. How the scheduler was made — from zero to working code](#16-how-the-scheduler-was-made)
  - [17. How the workload was generated](#17-how-the-workload-was-generated)
  - [18. How the GPU kernel works — line by line](#18-how-the-gpu-kernel-works)
  - [19. How the experiment runner orchestrates everything](#19-how-the-experiment-runner-orchestrates-everything)
  - [20. How the statistical analysis works — every step](#20-how-the-statistical-analysis-works)
  - [21. How calibration works and why it exists](#21-how-calibration-works-and-why-it-exists)
- [Part V: Problems Encountered and How They Were Solved](#part-v-problems-encountered-and-how-they-were-solved)
- [Part VI: Every Decision and Why](#part-vi-every-decision-and-why)
- [Part VII: The Results — What They Mean and How We Know](#part-vii-the-results)
- [Part VIII: Interview Questions You Will Be Asked](#part-viii-interview-questions)

---

# Part I: Concepts From Scratch

## 1. What is a GPU and why does it exist?

A CPU (Central Processing Unit) is the main brain of a computer. It has a small number of powerful cores — typically 4 to 16. Each core can handle complex logic: conditionals, function calls, deeply nested loops. It is optimised for doing one thing very well, very fast.

A GPU (Graphics Processing Unit) was originally designed to draw images on a screen. Drawing an image means computing the colour of every pixel — millions of them — independently. Each pixel's colour does not depend on any other pixel. So the GPU was built with thousands of small, simple cores that can all work at the same time.

The RTX 4070 (the GPU used in this project) has **5,888 shader cores**. Each one is individually weaker than a CPU core, but together they can do thousands of computations simultaneously.

**When to use a GPU:** When you have a large number of independent computations that all follow roughly the same logic. This is called "data parallelism." Examples: matrix multiplication, image filtering, physics simulation, neural network training.

**When NOT to use a GPU:** When tasks have complex dependencies on each other, or when each task needs a different code path.

**An interviewer will ask:** "Why did you use a GPU for this project?"

**Your answer:** "The workload consists of 800,000 independent tasks. Each task runs the same loop body with a different iteration count. This is textbook data parallelism — thousands of identical operations on different data. A GPU can run thousands of these simultaneously."

---

## 2. What is CUDA?

CUDA is NVIDIA's programming framework for writing code that runs on NVIDIA GPUs. It extends C++ with a few special keywords.

The most important keyword is `__global__`. A function marked `__global__` is called a **kernel**. You write it in C++, but it runs on the GPU, not the CPU.

```cpp
__global__ void my_kernel(int* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        data[tid] = data[tid] * 2;  // Each thread doubles its element
    }
}
```

To launch a kernel, you use a special syntax:
```cpp
my_kernel<<<num_blocks, threads_per_block>>>(data, n);
```

The CPU (called the "host") tells the GPU (called the "device") to run `my_kernel` with `num_blocks × threads_per_block` threads total. Each thread gets a unique ID through `threadIdx.x` and `blockIdx.x`.

**The CPU and GPU have separate memory.** You must:
1. Allocate memory on the GPU: `cudaMalloc(&d_data, size)`
2. Copy data from CPU to GPU: `cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice)`
3. Launch the kernel
4. Wait for the GPU to finish: `cudaDeviceSynchronize()`
5. Copy results back if needed

**An interviewer will ask:** "What happens when you call `cudaDeviceSynchronize()`?"

**Your answer:** "It blocks the CPU thread until all previously launched GPU work has completed. Without it, the CPU would continue executing while the GPU is still computing, and any timing measurement would be wrong — you'd measure the time to *submit* work, not the time to *complete* work."

---

## 3. What is a thread, a warp, a block, and a grid?

This is the CUDA execution hierarchy. You must know this cold.

**Thread:** The smallest unit of execution. One thread runs on one GPU core and processes one piece of data. In this project, one thread processes one task (one loop with a specific iteration count).

**Warp:** A group of 32 threads that execute **in lockstep**. This is the most important hardware concept to understand. All 32 threads in a warp execute the same instruction at the same time. If 31 threads are done but 1 thread is still looping, all 31 sit idle waiting for the 1 to finish.

This project uses this fact: chunk size = 10,000 threads → 10,000 / 32 = 312.5 → 313 warps per kernel launch.

**Block:** A group of threads (up to 1,024) assigned to one Streaming Multiprocessor (SM). Threads within a block can share fast memory and synchronise with each other. This project uses blocks of 256 threads (8 warps per block).

**Grid:** All blocks in one kernel launch. Grid = 10,000 tasks / 256 threads per block = ~40 blocks per launch.

**Why 256 threads per block?** (An interviewer will ask this.)

**Answer:** "256 is a standard choice that balances occupancy and register pressure. Each SM on the RTX 4070 can hold up to 1,536 threads (48 warps). At 256 threads per block, we can fit 6 blocks per SM (6 × 256 = 1,536), maximising occupancy. Going to 512 or 1,024 per block reduces the number of blocks an SM can hold simultaneously, which can hurt latency hiding if the kernel uses too many registers."

---

## 4. What is a kernel launch and why does it cost time?

When the CPU tells the GPU to run a kernel, several things happen before any computation begins:

1. The CUDA runtime packages the kernel arguments into a launch configuration
2. This configuration is submitted to the GPU driver
3. On Windows WDDM mode, the driver batches this submission through the Windows Display Driver Model — an OS-level abstraction layer that adds latency
4. The GPU's hardware work distributor allocates blocks to SMs
5. Only then do threads begin executing

This entire sequence takes a fixed amount of time regardless of how much work is inside the kernel. We measured it:

```
launch_overhead_ms: 0.751630
```

This is from `results/calibration.json`. It means every single kernel launch costs 0.75ms before any useful computation happens.

**Why this matters enormously:** If you launch a kernel with 10,000 tasks that take 13.5ms to compute, the overhead is 0.75 / (0.75 + 13.5) = 5.3% — acceptable. But if you launch a kernel with 1,000 tasks that take 1.35ms, the overhead is 0.75 / (0.75 + 1.35) = 36% — devastating. And if you do 800 launches instead of 80, you pay 800 × 0.75ms = 600ms of pure overhead.

**An interviewer will ask:** "What is WDDM and why does it affect your results?"

**Answer:** "WDDM is Windows Display Driver Model. On Windows, the GPU driver operates through an OS-managed display driver layer that adds buffering and scheduling overhead to every kernel launch. On Linux with TCC mode (Tesla Compute Cluster), launches bypass this layer and cost microseconds instead of ~750 microseconds. Our entire finding — that adaptive chunk shrinking hurts performance — is dominated by this per-launch overhead. Under TCC mode, the results could be fundamentally different, and this is explicitly listed as a limitation."

---

## 5. What is chunking and why is it needed?

"Chunking" means breaking a large set of tasks into smaller groups (chunks) and processing one group at a time.

**Why not process all 800,000 tasks in one kernel launch?**

For the purpose of this experiment, you could. But then you would have zero decision points — no opportunity for the scheduler to observe what happened and adjust. The entire point of the project is to test whether making adjustments *between* batches helps.

In practice, chunking exists because:
1. **Memory constraints:** A GPU has limited memory. If each task needs significant shared memory or registers, you cannot fit all tasks at once.
2. **Stream scheduling:** In production systems, you interleave computation with data transfers using CUDA streams. Chunking enables this overlap.
3. **Fault tolerance:** If a kernel crashes, you lose one chunk, not the entire workload.
4. **Resource sharing:** In multi-user GPU environments (cloud, HPC), you cannot monopolise the GPU with one massive launch.

**In this project specifically:** Chunking is the mechanism under study. The scheduler decides how many tasks go into each chunk. The question is whether changing chunk size based on workload characteristics improves total execution time.

**An interviewer will ask:** "Why didn't you just use one kernel launch for everything?"

**Answer:** "That would eliminate the experimental variable entirely. The research question is whether inter-launch scheduling decisions add value. But practically, production GPU schedulers *do* use chunking because of memory limits, stream interleaving, and multi-tenant resource sharing. The question of optimal chunk size under workload variance is real."

---

## 6. What is variance (the math)?

**Mean (average):** Sum of all values divided by count.

For values {200, 800, 200, 800}: mean = (200+800+200+800)/4 = 500.

**Variance:** Measures how spread out values are around the mean. It is the average of the squared deviations from the mean.

**Sample variance formula** (used in this project):

$$\sigma^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

For {200, 800, 200, 800}:
1. Mean = 500
2. Deviations: (200-500)=-300, (800-500)=300, (200-500)=-300, (800-500)=300
3. Squared deviations: 90000, 90000, 90000, 90000
4. Sum = 360,000
5. Divide by n-1 = 3: **variance = 120,000**

For {490, 510, 500, 500}:
1. Mean = 500
2. Deviations: -10, 10, 0, 0
3. Squared: 100, 100, 0, 0
4. Sum = 200
5. Divide by 3: **variance = 66.7**

Both sets have the same mean (500) but radically different variance. High variance = unpredictable. Low variance = uniform.

**Why divide by n-1 instead of n?** This is called Bessel's correction. When you compute variance from a sample (not the full population), dividing by n slightly underestimates the true variance. Dividing by n-1 corrects this bias. The V1 code used `/WINDOW` (population variance) — this was one of many methodological issues.

**How variance is computed in the actual code** (V2, `adaptive_scheduler.h`):

```cpp
// Inside select_chunk():
window_.clear();
int ws = obs.window_start(cfg_.window_size);  // = max(0, completed - 20)
int we = obs.num_completed();
for (int i = ws; i < we; ++i)
    window_.push(static_cast<double>(obs.completed_cost(i)));

double w_var = window_.variance();
```

The `variance()` method in `sliding_window.h`:
```cpp
double variance() const {
    if (n_ < 2) return 0.0;
    double m = mean();
    double ss = 0.0;
    for (int i = 0; i < n_; ++i) {
        double d = data_[(start_ + i) % capacity_] - m;
        ss += d * d;
    }
    return ss / (n_ - 1);  // Bessel's correction
}
```

**An interviewer will ask:** "Why did you compute variance on task costs instead of execution times?"

**Answer:** "The V1 experiment used execution-time variance and it created a feedback loop. GPU timing includes random noise from OS scheduling, driver batching, and thermal throttling. When the scheduler reacted to noisy timing by shrinking chunks, it created more launches, which added more noise, which triggered more shrinking. The controller spiralled. V2 computes variance on the raw task costs — the ground-truth computational work — which are deterministic and noise-free. This breaks the feedback loop."

---

## 7. What is a sliding window?

A sliding window is a buffer that holds the N most recent values. When you add a new value and the buffer is full, the oldest value is discarded.

```
Window size = 3

Observe 10 → [10]
Observe 20 → [10, 20]
Observe 30 → [10, 20, 30]    ← full
Observe 40 → [20, 30, 40]    ← 10 dropped
Observe 50 → [30, 40, 50]    ← 20 dropped
```

At every step, you can compute the mean and variance of only the recent values. This lets you detect *changes* in the workload character. If the last 20 tasks were expensive, the window's mean is high even if the first 780,000 tasks were cheap.

**Why window size = 20?**

This was chosen as a balance:
- Too small (e.g., 3): Variance estimate is noisy. Three data points cannot distinguish "high variance" from "two outliers."
- Too large (e.g., 200): Slow to react. If a regime change happens, 200 old values from the previous regime dilute the signal. With 200,000 tasks per phase, a window of 200 means the scheduler needs 200 tasks (0.1% of the phase) before the window reflects the new regime. With 20 tasks, it needs 0.01%.
- 20 provides a statistically meaningful variance estimate (degrees of freedom = 19) while reacting within ~20 tasks of a regime change.

**An interviewer will ask:** "Why not use an exponential moving average (EMA) instead of a sliding window?"

**Answer:** "An EMA weights all history, with exponentially decaying influence. A sliding window cleanly discards data older than N observations. For our purpose — detecting regime changes — the sliding window is better because it provides a sharp transition. An EMA would take longer to 'forget' the old regime because it never fully discards old data. The window also makes the variance calculation straightforward and numerically stable, which matters for reproducibility."

---

## 8. What is a causal system?

A system is **causal** if it only uses information from the past and present when making decisions. It cannot use information about the future.

In a real GPU scheduler, this is inherently true — you cannot see tasks that have not arrived yet. But in a *simulated experiment*, it is easy to accidentally cheat:

**Accidental violation 1:** Computing the mean cost across the entire 800,000-task array before the experiment starts, and using that mean to set thresholds. This leaks future information into the scheduler's calibration.

**Accidental violation 2:** Passing the full `task_costs` array directly to the scheduler, which could then peek at future tasks even if it "promises" not to.

**How this project enforces causality mechanically:**

```cpp
// From scheduler.h (lines 26-55)
class CausalObserver {
public:
    CausalObserver(const int* costs, int num_completed)
        : costs_(costs), n_completed_(num_completed) {}

    int completed_cost(int task_idx) const {
        assert(task_idx >= 0 && task_idx < n_completed_ &&
               "CausalObserver: attempt to read future task cost");
        return costs_[task_idx];
    }

    int num_completed() const { return n_completed_; }
};
```

The `assert` on line 43 **crashes the program immediately** if any code tries to read a task that has not been completed yet. This is not a convention or a comment — it is a runtime enforcement mechanism.

Before every scheduling decision:
```cpp
// In experiment_runner.cpp, run_arm():
CausalObserver obs(task_costs.data(), processed);
int chunk = sched.select_chunk(obs, remaining);
```

A fresh `CausalObserver` is created with the current `processed` count. The scheduler literally cannot access `task_costs[processed]` or beyond.

**An interviewer will ask:** "How do you prove the scheduler doesn't cheat?"

**Answer:** "The `CausalObserver` wraps the full array but exposes only the completed prefix. Attempting to access any index at or beyond `num_completed()` triggers an assertion failure that terminates the program. A fresh observer is constructed before every scheduling decision with the exact number of completed tasks. The scheduler never receives the raw array. This is a compile-time-enforced contract through the abstract `Scheduler` interface — `select_chunk()` takes `const CausalObserver&`, not `const vector<int>&`."

---

## 9. What is a Wilcoxon signed-rank test?

This is a statistical test for determining whether two sets of paired measurements are different.

**Why not a t-test?** A t-test assumes your data is normally distributed (bell-curve shaped). GPU timing data is not: it has occasional spikes from OS scheduling, thermal throttling, or WDDM batching delays. The Wilcoxon test makes no assumption about the distribution shape. It only asks: "Is the median of the paired differences zero?"

**How it works, step by step** (using the actual implementation in `statistical_analysis.cpp`):

Given 20 repetitions, each producing a timing for ADAPTIVE and STATIC-MATCHED:

```
Step 1: Compute paired differences.
        d[i] = adaptive[i] - matched[i]  for each repetition i

Step 2: Remove any d[i] = 0 (ties carry no information).
        Count remaining as m.

Step 3: Sort |d[i]| (absolute values) in ascending order.
        Assign ranks 1 through m.
        If two |d[i]| values are equal, give them the average rank.

Step 4: Compute W+ = sum of ranks where d[i] > 0
        Compute W- = sum of ranks where d[i] < 0
        W = min(W+, W-)

Step 5: Under the null hypothesis (no difference), W follows a known distribution.
        For m ≥ 10, the normal approximation is used:
        
        mean_W = m(m+1)/4
        var_W  = m(m+1)(2m+1)/24 - (tie correction)
        Z = (|W - mean_W| - 0.5) / sqrt(var_W)    (0.5 is continuity correction)

Step 6: p-value = 2 × (1 - Φ(Z))    (two-tailed)
        where Φ is the standard normal CDF.
```

If p < 0.05 (our pre-registered threshold), we reject the null hypothesis — there IS a statistically significant difference.

**Our actual result:** W = 0.0, Z = 3.90, p = 0.0000957.

W = 0 means that in ALL 20 repetitions, the difference had the same sign. ADAPTIVE was slower than MATCHED in every single repetition, with no exceptions. This is the strongest possible signal.

**An interviewer will ask:** "Why p < 0.001 and not p < 0.05?"

**Answer:** "Our pre-registered alpha was 0.05, but the observed p-value of 0.0001 far exceeds that threshold. With W = 0 (all 20 differences in the same direction), we have extremely strong evidence. I would say the result is significant at p < 0.001, but our threshold was p < 0.05 — and the effect meets both. I do not adjust the threshold after seeing the data."

---

## 10. What is warp divergence?

In CUDA, 32 threads form a warp and execute the same instruction simultaneously. If some threads in a warp take a different code path (an `if/else` branch), the hardware must execute both branches serially — with inactive threads masked off.

But in this project, the divergence is not in code paths. It is in **loop iteration counts**.

```cuda
for (int i = 0; i < W; ++i)       // W varies per thread
    acc += (i ^ W) & 0xFF;
```

Thread 0 might have W = 200 (done in 200 iterations). Thread 31 might have W = 3,000 (done in 3,000 iterations). But they are in the same warp. Thread 0 finishes at iteration 200 and then sits idle — doing nothing — for the remaining 2,800 iterations while thread 31 is still working.

This is the **actual performance problem** that workload variance creates on a GPU. It is not about total compute time (the same total work is done either way). It is about wasted thread time — idle cycles on cores that have finished their work but are trapped in a warp with slower threads.

**An interviewer will ask:** "How would you reduce warp divergence in this workload?"

**Answer:** "Sort the tasks by cost before launching each chunk. If cheap tasks (cost ~200) are grouped together in the same warp, and expensive tasks (cost ~3,000) are grouped together in a different warp, then no thread is ever waiting for a much-slower neighbour. The cheap warp finishes quickly and its SM is freed for new work. The expensive warp runs longer, but no cycles are wasted. This is exactly what the SORTING-ADAPTIVE scheduler does — and it is the correct use of the variance signal."

---

# Part II: The Problem Statement

## 11. The problem in one paragraph

GPU workloads in practice have tasks with unequal computational costs. A scheduler that breaks work into fixed-size chunks cannot react to changes in workload variability. The intuitive solution is to monitor workload variance in real time and adjust chunk size accordingly — smaller chunks during high-variance phases (more decision points) and larger chunks during low-variance phases (less overhead). This project builds that system, tests it under rigorous experimental conditions, and discovers that the intuitive strategy is wrong: adaptive chunk shrinking makes performance worse, not better, because the per-launch overhead of more frequent kernel launches outweighs any benefit of finer-grained scheduling. The project then diagnoses the root cause and proposes a corrected strategy.

## 12. Why this problem matters (real-world use cases)

This is not an academic exercise. The chunk size decision happens in every GPU-accelerated system:

| System | How chunking appears | Why variance matters |
|--------|---------------------|---------------------|
| **PyTorch / TensorFlow inference** | Batching incoming requests into GPU batches | Request complexity varies (short text vs. long document) |
| **NVIDIA Triton Inference Server** | Dynamic batching with configurable max batch size | Heterogeneous model queries with different compute costs |
| **Real-time ray tracing** | Dividing screen tiles into GPU work packets | Some tiles hit complex geometry; others are empty sky |
| **Graph neural networks** | Processing node neighbourhoods in batches | Node degree varies enormously (power-law graphs) |
| **Molecular dynamics (GROMACS)** | Dividing particle interactions into GPU tiles | Non-bonded interactions vary with local particle density |
| **Genomics (BWA-MEM2)** | Batching sequence alignments for GPU | Read lengths and alignment complexity vary per read |

In all cases, someone must decide how many tasks to group per GPU launch. This project provides empirical evidence about when adaptive sizing helps and when it hurts.

## 13. What existed before this project?

**Prior work in the literature:**
- CUDA Dynamic Parallelism (CDP): allows kernels to launch child kernels, enabling recursive work decomposition. But CDP has high overhead per child launch.
- OpenMP `schedule(dynamic)` and `schedule(guided)`: CPU-side dynamic chunk scheduling. Not directly applicable to GPU kernel launches.
- Persistent kernels with work queues: one kernel launch, threads pull tasks from a shared queue using atomics. Eliminates launch overhead entirely. This is the theoretically optimal architecture but was not explored in V1.
- NVIDIA CUDA Graphs: pre-recorded sequences of launches to reduce per-launch overhead. Requires static launch patterns.

**What V1 of this project did:**
The original experiment (V1) built a variance-aware scheduler that used GPU execution timing to detect variance and adjust chunk size. It reported a 4.9% improvement. But it had fatal methodological flaws: no controlled baseline, timing feedback loop, n=5 trials, population variance instead of sample variance. V2 was the complete rewrite to test whether the claimed improvement was real.

---

# Part III: Resume Line-by-Line Breakdown

## 14. Project 1: "Causal GPU Scheduling Framework with Automated Statistical Validation"

### Line: "Engineered a causal-observer scheduling interface"

**What is a causal-observer?** A wrapper around the task cost array that exposes only completed tasks. See Section 8 for the full explanation and exact code.

**What is a scheduling interface?** An abstract C++ class (`Scheduler` in `scheduler.h`) that defines the contract any scheduling strategy must follow:

```cpp
class Scheduler {
public:
    virtual void reset() = 0;
    virtual int select_chunk(const CausalObserver& obs, int chunk_remaining) = 0;
    virtual std::string name() const = 0;
    virtual const std::vector<SchedulerDecision>& trace() const = 0;
    virtual void record_timing(double t_kernel_ms, double t_total_chunk_ms) = 0;
};
```

Why `virtual` and `= 0`? This makes them "pure virtual" — any class that inherits from `Scheduler` MUST implement these methods. The `ExperimentRunner` only holds `Scheduler*` pointers — it does not know or care which specific scheduler is running. This lets us add new schedulers (ADAPTIVE, INVERTED, SORTING) without changing the runner code.

**"Engineered"** means: this was designed and implemented from scratch, not imported from a library or framework.

### Line: "three-arm benchmarking protocol (static, size-matched static, adaptive)"

**What is an "arm"?** Borrowed from clinical trial terminology. An "arm" is one treatment group in a controlled experiment. Just as a drug trial has a treatment arm and a placebo arm, our experiment has:

1. **ADAPTIVE arm:** The treatment under test
2. **STATIC-LARGE arm:** The naive baseline (always max chunk)
3. **STATIC-MATCHED arm:** The controlled baseline (same average chunk as adaptive, but fixed)

**Why "size-matched"?** See Section 6 of the CODEBASE_GUIDE. Without STATIC-MATCHED, you cannot distinguish "adaptiveness helped" from "smaller chunks helped." This is the key experimental design insight.

### Line: "to isolate scheduling strategy from chunk-size as independent variables"

**What are independent variables?** In experimental design, an independent variable is one you intentionally change to observe its effect. Here, two variables could explain performance differences:

1. **Chunk size magnitude:** Maybe chunks of 2,000 are just better than chunks of 10,000, regardless of whether they adapt.
2. **Adaptiveness:** Maybe changing chunk size over time helps, regardless of the average chunk size.

STATIC-MATCHED isolates variable #2 by holding variable #1 constant. If ADAPTIVE beats STATIC-MATCHED (same average chunk, but one adapts and the other doesn't), then adaptiveness itself is the cause. If ADAPTIVE loses to STATIC-MATCHED, adaptiveness itself is the problem.

### Line: "Automated hardware calibration to guarantee compute-bound measurement conditions"

**What does "compute-bound" mean?** A workload is compute-bound if the GPU spends most of its time computing, not waiting for memory or communication. If the GPU spends 90% of its time on actual computation and 10% on overhead, measurement noise from overhead is small. If it spends 10% on computation and 90% on overhead, you are measuring overhead, not your scheduler's effect.

**What is calibration?** Before running the experiment, we sweep through different `cost_multiplier` values to find one where actual GPU computation time exceeds 10× the launch overhead:

From `results/calibration.json`:
```
launch_overhead_ms: 0.751630
multiplier_sweep:
  multiplier=1    → t_kernel=0.020ms   (0.03× overhead — measuring noise)
  multiplier=10   → t_kernel=0.042ms   (0.06× overhead — still noise)
  multiplier=50   → t_kernel=0.179ms   (0.24× overhead — still noise)
  multiplier=100  → t_kernel=0.351ms   (0.47× overhead — mostly noise)
  multiplier=500  → t_kernel=1.698ms   (2.3×  overhead — marginal)
  multiplier=1000 → t_kernel=3.382ms   (4.5×  overhead — getting there)
  multiplier=2000 → t_kernel=6.756ms   (9.0×  overhead — almost)
  multiplier=4000 → t_kernel=13.491ms  (18.0× overhead — YES) ← chosen
```

At `cost_multiplier=4000`, `T_kernel / launch_overhead = 13.49 / 0.75 = 18×`. The GPU spends 18× more time computing than on launch overhead. This guarantees we are measuring real compute-bound behaviour.

**"Automated"** means the calibration is a built-in mode of the program (`--calibrate` flag), not a manual process.

### Line: "diagnosed a CUDA driver/toolkit version mismatch blocking real-GPU execution"

During development, the binary compiled successfully but produced error code 801 (`cudaErrorCallRequiresNewerDriver`) at runtime. This meant the CUDA toolkit version (13.1) generated PTX or SASS code that the installed GPU driver (version 555.97) could not understand.

**How it was diagnosed:** The `CudaExecutor` constructor tries `cudaMalloc`. When it fails, it prints the CUDA error string and falls back to simulation mode. The error message identified code 801 specifically.

**How it was resolved:** The compilation was changed from separate compilation (multiple `.cu` files compiled independently and linked) to single-translation-unit compilation (all code included into `main.cu`). This avoids `nvcc`'s device linker (`nvlink`), which was the component triggering the driver mismatch. The exact build command:

```bat
nvcc -allow-unsupported-compiler src/main.cu -o variance_adaptive_v2.exe
```

`-allow-unsupported-compiler` permits `nvcc` to use an MSVC version newer than its official compatibility list.

### Line: "resolving it to validate results on actual hardware rather than a simulated fallback"

The simulation fallback estimated GPU timing with:
```cpp
double sim_ms = 12.6 + (total_warp_cost * cost_mult * 0.00000005);
```

This was useful for validating the scheduling logic, statistical analysis, and output pipeline. But the simulation cannot capture real WDDM launch overhead, thermal throttling, or warp scheduling effects. The results are only scientifically valid with `execution_mode: "cuda"`, which was confirmed in every `config.json` after the driver issue was resolved.

### Line: "Validated on real GPU hardware (n=20 reps)"

20 repetitions, each with a different random seed (`seed = 42 + rep × 1,000,003`), all confirmed via `execution_mode: "cuda"` in the output `config.json`.

### Line: "used the framework to uncover a statistically significant confound (p<0.001)"

**What is a confound?** A hidden variable that makes you think one thing caused the effect when actually something else did.

The original V1 experiment claimed adaptive scheduling improved performance by 4.9%. The confound was **chunk-size magnitude**. V1 compared ADAPTIVE (average chunk ~50,000) against STATIC (chunk = 100,000). The improvement could have come from smaller chunks being better in general — not from adaptation.

V2's STATIC-MATCHED arm revealed that adaptation does not help — it hurts (p = 0.0001). This is the confound that was "uncovered": what V1 attributed to adaptive intelligence was actually just the effect of average chunk size.

### Line: "missed by the original single-run benchmark"

V1 ran n=5 trials. At n=5, the Wilcoxon test has almost no statistical power. A 4.9% difference at n=5 cannot be distinguished from random noise. V2's n=20 provides 80%+ power to detect a 5% effect size, making the test meaningful.

---

## 15. Project 2: "Variance-Aware Adaptive GPU Scheduler"

### Line: "Built a runtime scheduler in CUDA that adapts GPU workload chunking"

This refers to V1's `variance_chunking.cu`. The scheduler runs on the CPU and adjusts chunk size between kernel launches. "Runtime" means the decisions happen during execution, not ahead of time.

### Line: "using live execution-time variance as a feedback signal"

V1 used GPU execution timing (measured with `std::chrono`) as the variance signal. Each chunk's execution time was added to a sliding window of size 5. Variance was computed as population variance (`/WINDOW`, not `/(WINDOW-1)`). The exact code:

```cpp
// variance_chunking.cu lines 120-163
window.push_back(ms);           // ms = chunk execution time
if (window.size() > WINDOW)
    window.erase(window.begin());

double mean = 0.0;
for (double t : window) mean += t;
mean /= WINDOW;

double variance = 0.0;
for (double t : window)
    variance += (t - mean) * (t - mean);
variance /= WINDOW;  // population variance, not sample
```

### Line: "cutting high-variance execution episodes by 33%"

V1's tuned scheduler achieved 117.4ms vs. 123.5ms on the static baseline. (123.5 - 117.4) / 123.5 ≈ 4.9% — not 33%. The 33% figure likely refers to reduction in tail-latency variance of individual chunks, not total runtime. V2 showed this improvement is not statistically reliable.

### Line: "with no loss in average throughput (2.12 ms vs. 2.25 ms baseline)"

These are per-chunk average execution times from V1. The claim is valid within V1's methodology but is based on n=5 and without a size-matched control.

### Line: "sliding-window variance tracking and threshold-based chunk resizing"

The exact V1 thresholds and logic:

```cpp
const int WINDOW = 5;
const double VAR_HIGH = 0.80;    // ms² — threshold for shrinking
const double VAR_LOW  = 0.10;    // ms² — threshold for growing
int chunk = 100000;
const int min_chunk = 25000;
const int max_chunk = 200000;
const int chunk_step = 10000;
const int COOLDOWN = 3;

if (cooldown_remaining > 0) {
    cooldown_remaining--;
} else if (variance > VAR_HIGH && chunk > min_chunk) {
    chunk = (chunk * 3) / 4;      // shrink by 25%
    cooldown_remaining = COOLDOWN;
} else if (variance < VAR_LOW && chunk < max_chunk) {
    chunk += chunk_step;           // grow by 10,000
    cooldown_remaining = COOLDOWN;
}
```

Note: V1 shrinks multiplicatively (`*3/4`) but grows additively (`+10,000`). This asymmetry was a design decision — shrinking should be faster than growing to react quickly to sudden variance spikes.

### Line: "benchmarked across 1000+ kernel launches"

With 5,000,000 tasks and a starting chunk of 100,000, there are at least 50 launches. Across 5 trials with both static and adaptive, total launches exceed 1,000.

---

# Part IV: How Every Piece Was Built

## 16. How the scheduler was made

### The thought process, step by step:

**Step 1: Define the interface first.**

Before writing any scheduling logic, define what a scheduler must be able to do:
- Accept an observation of past data → return a chunk size decision
- Reset between repetitions
- Record what decisions it made (for analysis)

This became the abstract `Scheduler` class.

**Step 2: Design the observation mechanism.**

The scheduler needs data about past tasks. But it must not see the future. So we created `CausalObserver` — a wrapper that exposes only completed tasks, with a hard `assert` on future access.

**Step 3: Choose the variance signal.**

V1 used execution time. This was wrong (feedback loop). V2 uses raw task costs — the integer that determines how many loop iterations each task performs. These are deterministic: the same task always has the same cost regardless of GPU clock speed, OS scheduling, or thermal state.

**Step 4: Choose the policy.**

The simplest possible policy:
```
if variance > HIGH → shrink chunk by STEP
if variance < LOW  → grow chunk by STEP
otherwise          → hold
```

Why linear stepping (STEP = 1,000) instead of proportional (multiply by 0.75 like V1)?

Proportional: `chunk = chunk * 0.75` → 10000 → 7500 → 5625 → 4219 → 3164 → 2373 → 1780 → 1335 → 1001. Converges to min_chunk in 9 steps. Overshoots if variance is moderate.

Linear: `chunk -= 1000` → 10000 → 9000 → 8000 → 7000 → ... → 1000. Takes 9 steps, but each step is predictable and the scheduler can stop at any intermediate value. Less likely to overshoot.

The choice was linear for simplicity and predictability. The scheduler should walk toward its target, not jump.

**Step 5: Choose the thresholds.**

`VAR_LOW = 5,000` and `VAR_HIGH = 30,000` were set during calibration based on the actual phase variances:

| Phase | Distribution | Measured Variance |
|-------|-------------|------------------|
| C0 (Uniform 200-400) | Low | ~3,333 |
| C1 (Bimodal 100-300/700-900) | High | ~120,000 |
| C2 (Uniform 350-450) | Low | ~833 |
| C3 (Log-normal) | Bursty | ~486,000 |

- `VAR_LOW = 5,000` captures C0 and C2 (both well below 5,000)
- `VAR_HIGH = 30,000` triggers on C1 and C3 (both far above 30,000)
- The gap between 5,000 and 30,000 is a "dead zone" where the scheduler holds steady, preventing oscillation.

**Step 6: Choose the window size.**

Window = 20. See Section 7 for the reasoning.

---

## 17. How the workload was generated

The workload is an array of 800,000 integers. Each integer represents the computational cost of one task. The array is divided into four phases of 200,000 tasks each.

For Experiment C (the primary experiment):

```cpp
// From workload_generator.cpp — preset_experiment_C()
// Phase 0: Uniform[200, 400] — low variance, stable
// Phase 1: Bimodal — 50% Uniform[100,300] + 50% Uniform[700,900] — high variance
// Phase 2: Uniform[350, 450] — low variance again
// Phase 3: Log-normal(μ=5.5, σ=0.9) clamped [100, 3000] — bursty spikes
```

**Why these specific distributions?**

- **Uniform:** The simplest distribution. All values equally likely between min and max. Creates a known, calculable variance: Var = (b-a)² / 12.
- **Bimodal:** Two clusters far apart. Creates extremely high variance because values cluster far from the mean. Simulates workloads with two distinct task types (fast preprocessing + slow processing).
- **Log-normal:** The logarithm of each value is normally distributed. Produces occasional extreme outliers — a few tasks cost 3,000 while most cost ~200. This matches real-world phenomena: file sizes, network latency, query processing times all follow log-normal distributions.

**Reproducibility:** The random number generator is `std::mt19937` (Mersenne Twister) seeded with `42 + rep × 1,000,003`. Using a prime stride ensures different repetitions are statistically independent. All four arms within one repetition share the SAME workload (same seed, same array). This eliminates workload variation as a confound between arms.

---

## 18. How the GPU kernel works — line by line

```cuda
// From cuda_executor.cu (lines 44-65)
__global__ void work_kernel(
    const int* task_costs,   // Array of costs, uploaded from CPU
    int task_offset,         // Where this chunk starts
    int chunk_size,          // How many tasks in this chunk
    volatile int* out,       // Output array (volatile = don't optimise away)
    int cost_multiplier)     // Scaling factor (4000 in final experiments)
{
    // Step 1: Each thread computes its unique global index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 2: Guard — threads beyond chunk_size do nothing
    if (tid >= chunk_size) return;

    // Step 3: Look up this task's cost and scale it
    int W = task_costs[task_offset + tid] * cost_multiplier;

    // Step 4: Compute loop — W iterations of bitwise operations
    int acc = 0;
    for (int i = 0; i < W; ++i)
        acc += (i ^ W) & 0xFF;

    // Step 5: Write result to prevent dead-code elimination
    out[task_offset + tid] = acc;
}
```

**Why `volatile int* out`?** Without `volatile`, the compiler sees that `acc` is computed, written to `out`, and never read again. A smart compiler would delete the entire loop and the write — "dead code elimination." `volatile` tells the compiler: "This memory location may be read by something you cannot see. You must perform the write." This ensures the computation actually happens.

**Why `(i ^ W) & 0xFF`?** XOR (`^`) produces a bit pattern that depends on both `i` and `W`, making the computation non-trivial. `& 0xFF` masks to the low 8 bits, keeping `acc` from overflowing. The specific operation does not matter — it just needs to be cheap, non-optimisable, and linearly scaling with `W`.

**How the kernel is launched:**

```cpp
// From cuda_executor.cu, launch_chunk()
int n_blocks = (chunk_size + tpb_ - 1) / tpb_;  // ceiling division
// For chunk=10000, tpb=256: n_blocks = (10000+255)/256 = 40

cudaEventRecord(ev_start_);
work_kernel<<<n_blocks, tpb_>>>(d_tasks_, task_offset, chunk_size, d_out_, cost_mult_);
cudaEventRecord(ev_stop_);
cudaDeviceSynchronize();           // Wait for GPU to finish

float gpu_ms;
cudaEventElapsedTime(&gpu_ms, ev_start_, ev_stop_);  // T_kernel
// Host wall-clock T_total_chunk is measured with std::chrono around the above
```

**Two timing measurements per chunk:**
1. `t_kernel_ms` (CUDA events): GPU-side execution only. Excludes launch overhead.
2. `t_total_chunk_ms` (std::chrono wall clock): Everything. Launch + GPU + sync.

The gap between them (t_total - t_kernel) is approximately the launch overhead (~0.75ms).

---

## 19. How the experiment runner orchestrates everything

The `ExperimentRunner` in `experiment_runner.cpp` controls the entire multi-arm protocol.

**For each of 20 repetitions:**

```
1. Generate workload with seed = 42 + rep × 1,000,003
2. Upload task costs to GPU (one upload, shared by all arms)

3. PASS 1: Run ADAPTIVE arm
   → Record T_total, trace of all decisions, and mean chunk size
   → Compute matched_chunk = round(mean of all adaptive chunk sizes)

4. Sleep 200ms + cudaDeviceSynchronize() (thermal settling)

5. PASS 2 (order randomised by rep % 2):
   → Run STATIC-LARGE (chunk = 10,000)
   → Sleep 200ms
   → Run STATIC-MATCHED (chunk = matched_chunk from step 3)

6. Sleep 200ms

7. PASS 3: Run INVERTED-ADAPTIVE arm

8. Write one row to raw_results.csv and flush to disk
```

**Why ADAPTIVE must run first:** STATIC-MATCHED's chunk size depends on ADAPTIVE's average chunk. You must run ADAPTIVE before you know what MATCHED's chunk size should be. INVERTED is independent and can run in any order.

**Why order randomisation:** If STATIC-LARGE always runs before STATIC-MATCHED, any ordering effect (GPU warming up, thermal throttling) would systematically bias one arm. By alternating the order (even reps: LARGE first; odd reps: MATCHED first), the bias cancels across 20 reps.

**Why 200ms sleep:** GPU clock boosting. When the GPU runs heavy compute, it boosts its clock frequency. If arm B runs immediately after arm A, arm B benefits from the already-boosted clock, making it appear faster than it would in isolation. 200ms of sleep + `cudaDeviceSynchronize()` lets the GPU return to idle clocks and idle temperature.

**Evidence that thermal effects matter:** Without the settling pause, we observed 3-5% systematic bias favouring the second arm in every repetition. With the pause, arm ordering effects are within measurement noise (~0.5%).

---

## 20. How the statistical analysis works — every step

### The Wilcoxon signed-rank test (actual code walkthrough)

From `statistical_analysis.cpp` (lines 73-217), with actual Experiment C numbers:

```
Input: 20 pairs of (adaptive_ms, matched_ms)

Step 1: Paired differences d[i] = adaptive[i] - matched[i]
        All 20 d[i] are positive (adaptive is always slower)
        d values range from ~2500 to ~3600

Step 2: No zeros → m = 20

Step 3: Sort |d[i]| ascending, assign ranks 1..20

Step 4: W+ = 1+2+3+...+20 = 210  (all positive → all contribute to W+)
        W- = 0
        W = min(210, 0) = 0

Step 5: mean_W = 20 × 21 / 4 = 105
        var_W  = 20 × 21 × 41 / 24 = 717.5
        sigma_W = sqrt(717.5) = 26.79
        Z = (|0 - 105| - 0.5) / 26.79 = 104.5 / 26.79 = 3.901

Step 6: p = 2 × (1 - Φ(3.901)) = 2 × 0.0000479 = 0.0000957
```

**Result: p = 0.0000957 < 0.05 → statistically significant.**

Effect size r = Z / √n = 3.901 / √20 = 0.872 → "large" effect (> 0.5).

### Cross-validation against scipy

From `analyze_v2.py`:
```python
from scipy.stats import wilcoxon
stat, p_scipy = wilcoxon(df['t_total_adaptive_ms'], df['t_total_static_matched_ms'])
```

The Python script independently computes the Wilcoxon test and checks:
```
C++ p = 0.0001, scipy p = 0.0000 → delta = 0.0001 → [OK]
```

The small delta is due to the C++ implementation using a normal approximation while scipy uses an exact distribution. Both agree the result is highly significant.

---

## 21. How calibration works and why it exists

**Problem:** If `cost_multiplier` is too small, each task computes for microseconds. The measured time is dominated by launch overhead and OS scheduling noise. You would be benchmarking the Windows kernel, not your GPU scheduler.

**Solution:** Sweep multipliers until GPU compute time ≥ 10× launch overhead.

```cpp
// From experiment_runner.cpp, run_calibration()
vector<int> candidates = {1, 10, 50, 100, 500};
int m = 500;
while (t_kernel < 10.0 * overhead) {
    m *= 2;          // double until we clear the threshold
    candidates.push_back(m);
}
```

The 10× ratio ensures that launch overhead contributes at most 9% of total chunk time. This means 91%+ of what you measure is actual computation — the scheduler's effect on performance is real, not drowned in noise.

---

# Part V: Problems Encountered and How They Were Solved

## Problem 1: CUDA Driver Mismatch (Error 801)

**Symptom:** `variance_adaptive_v2.exe` compiled successfully but produced `cudaError 801: cudaErrorCallRequiresNewerDriver` at the first `cudaMalloc` call.

**Root cause investigation:**
1. Checked `nvcc --version`: CUDA Toolkit 13.1
2. Checked `nvidia-smi`: Driver 555.97
3. CUDA 13.1 generates PTX code targeting compute capability 8.9+ (Ada Lovelace). Driver 555.97 does not contain a JIT compiler for this PTX version.
4. The error occurred specifically with separate compilation (`-rdc=true -dlink`), not with single-TU compilation.

**Solution:** Switch to single-translation-unit compilation. All `.cpp` and `.cu` files are `#include`d into `main.cu`. Build command: `nvcc -allow-unsupported-compiler src/main.cu -o variance_adaptive_v2.exe`.

**Why this works:** Without separate compilation, `nvcc` resolves all device symbols in one pass and generates self-contained SASS (machine code) directly, bypassing the PTX→SASS JIT compilation step that triggered the driver mismatch.

**Evidence it worked:** After recompilation, `cudaMalloc` succeeded. `config.json` reports `execution_mode: "cuda"`. Kernel timing shows 13.49ms per 10,000-task chunk — consistent with real GPU execution, not simulation.

---

## Problem 2: V1 Timing Feedback Loop

**Symptom:** The V1 adaptive scheduler would shrink chunks until hitting `min_chunk = 25,000` on every workload, even uniform ones.

**Root cause:** V1 used `std::chrono` wall-clock execution time as the variance signal. But execution time includes:
- CUDA launch overhead (~0.75ms, random ±0.1ms)
- OS thread scheduling jitter
- GPU clock boost/throttle transients
- WDDM command batching delays

This noise was interpreted as "high variance" by the scheduler, triggering chunk shrinkage. Smaller chunks meant more launches, which meant more launch-overhead noise, which looked like even higher variance, triggering more shrinkage. A positive feedback loop.

**Solution:** V2 computes variance on pre-generated task costs — deterministic integers that do not change between runs or under different system conditions. The variance signal reflects genuine workload heterogeneity, not measurement noise.

**Evidence:** On Experiment A (uniform workload), V2's ADAPTIVE scheduler grows to MAX_CHUNK and stays there (variance ≈ 3,333 < VAR_LOW = 5,000). V1 would have shrunk to MIN_CHUNK due to timing noise. The variance signal is stable and predictable.

---

## Problem 3: No Controlled Baseline in V1

**Symptom:** V1 reported a 4.9% improvement but compared ADAPTIVE (average chunk ~50,000) against STATIC (chunk = 100,000). The improvement could be entirely from smaller chunks, not from adaptation.

**Root cause:** The experiment did not control for chunk-size magnitude as an independent variable.

**Solution:** V2 introduced STATIC-MATCHED — a static scheduler whose chunk size equals the adaptive scheduler's mean chunk size. This isolates adaptiveness as the variable under test.

**Result:** ADAPTIVE is 44.6% SLOWER than STATIC-MATCHED (p = 0.0001). The V1 "improvement" was not real.

---

## Problem 4: Statistical Underpowering (n=5)

**Symptom:** V1's 4.9% improvement at n=5 could easily be random noise.

**Root cause:** With only 5 data points, the Wilcoxon test has very low statistical power (~30% for a 5% effect). You would fail to detect a real difference most of the time, and apparent differences are often noise.

**Solution:** V2 uses n=20, which provides ~80% power for detecting a 5% effect (the pre-registered minimum practical effect threshold). The result at n=20 is definitive: p = 0.0001 with W = 0.

---

## Problem 5: Experiment D Running Too Long

**Symptom:** Experiment D (all bursty phases) took over 30 minutes. Previous experiments took 10-15 minutes.

**Root cause:** Experiment D's log-normal phases have extremely high variance (486,000). The ADAPTIVE scheduler shrinks to MIN_CHUNK (1,000) for nearly the entire workload. With 800,000 tasks ÷ 1,000 = 800 launches, and each launch taking ~0.75ms overhead + 1.35ms compute + 200ms inter-arm pause would be wrong — the pauses are between arms, not between chunks. The actual issue: at multiplier 4000, 800 chunks of 1,000 tasks each take about 800 × (0.75 + 1.35) = 1,680ms per arm. But ADAPTIVE takes 33,670ms — much more. The tasks in D have high individual costs (mean ~494 × 4000 = 1,976,000 iterations per task). Each 1,000-task chunk genuinely takes a long time.

**Solution:** Let it run. D is an extreme stress test. It completed in ~28 minutes and produced valid results.

---

## Problem 6: `is_improvement = false` despite p = 0.0001

**Symptom:** The statistical output says `is_significant: true` but `is_improvement: false`. This seems contradictory.

**Root cause:** The test checks whether ADAPTIVE is FASTER than MATCHED. The effect is -44.6% — meaning ADAPTIVE is 44.6% SLOWER. The test correctly reports "significant but not an improvement." This is not a bug; it is the framework correctly reporting that the hypothesis (adaptive helps) is disproven.

**How to explain in interview:** "Our framework was honest. We pre-registered the hypothesis that adaptive scheduling would be faster. The result was the opposite — it was significantly slower. The framework reported this accurately: statistically significant (p < 0.001, the effect is real) but not an improvement (the effect goes in the wrong direction). This is exactly how rigorous science works — you report what you find, not what you hoped for."

---

# Part VI: Every Decision and Why

| Decision | Why this, not the alternative |
|----------|------------------------------|
| **Single .cu file compilation** | Separate compilation triggered driver error 801. Single TU avoids device linking. |
| **256 threads per block** | Standard for compute-bound kernels. Maximises occupancy on RTX 4070 (6 blocks/SM × 256 = 1,536 = SM max). |
| **Window size = 20** | Large enough for stable variance estimate (df=19). Small enough to react within 0.01% of a 200,000-task phase. |
| **`var_low=5,000` `var_high=30,000`** | Set from calibration. Captures the actual phase variances with a dead zone to prevent oscillation. |
| **Linear step (±1,000) not proportional** | Predictable convergence. Proportional (×0.75) overshoots on moderate variance. |
| **Sample variance (n-1) not population (n)** | Mathematically correct for finite samples. V1 used /n (wrong). |
| **Wilcoxon not t-test** | GPU timing is not normally distributed. Wilcoxon is non-parametric — no distributional assumption. |
| **n=20 not n=5** | Power analysis: n=20 gives ~80% power for 5% effect. n=5 gives ~30%. |
| **200ms inter-arm pause** | GPU clock boost decays with time constant ~100ms. 200ms ensures >90% decay. |
| **Rep%2 order randomisation** | Eliminates systematic ordering bias. Even crude alternation cancels bias across 20 reps. |
| **CausalObserver with assert** | Mechanical enforcement, not convention. Catches bugs at development time, not at publication time. |
| **Task cost variance, not timing variance** | Breaks the V1 feedback loop. Task costs are deterministic ground truth. |
| **STATIC-MATCHED as primary comparison** | Controls for chunk-size magnitude. Only way to isolate adaptiveness as a variable. |
| **INVERTED-ADAPTIVE arm** | Mechanism test. If opposite policy recovers performance, the direction of adaptation is the cause. |
| **Cross-validating C++ stats against scipy** | Catches implementation bugs in the C++ Wilcoxon. Both must agree within 0.01. |
| **volatile int* out** | Prevents dead-code elimination. Without it, the compiler removes the entire loop. |
| **Flush CSV after every rep** | Crash safety. If the experiment crashes on rep 14, you still have reps 0-13. |

---

# Part VII: The Results

## Experiment C (Primary) — n=20, RTX 4070, CUDA confirmed

| Arm | Mean T_total | Std | Launches |
|-----|:---:|:---:|:---:|
| ADAPTIVE | 9,941.6 ms | 251.0 ms | ~500 |
| STATIC-MATCHED | 6,876.6 ms | 116.9 ms | ~100 |
| STATIC-LARGE | 2,724.7 ms | 22.9 ms | 80 |
| INVERTED | 4,550.7 ms | 36.5 ms | ~80 |

**Primary test (MATCHED vs ADAPTIVE):** p = 0.0001, W = 0, effect = -44.6%
→ ADAPTIVE is 44.6% SLOWER. Not an improvement.

**Mechanism test (INVERTED vs ADAPTIVE):** p = 0.0001, effect = -118.5%
→ INVERTED is 2.2× faster. Confirms the chunk-shrinking policy is the problem.

## All Four Experiments

| Exp | ADAPTIVE | MATCHED | LARGE | INVERTED |
|-----|---:|---:|---:|---:|
| A (uniform) | 999 ms | 998 ms | 999 ms | 4,764 ms |
| B (increasing) | 11,823 ms | 7,670 ms | 2,757 ms | 4,951 ms |
| C (alternating) | 9,942 ms | 6,877 ms | 2,725 ms | 4,551 ms |
| D (all bursty) | 33,670 ms | 27,440 ms | 6,924 ms | 6,906 ms |

**How to explain:** "ADAPTIVE is the worst on every variable workload because it shrinks chunks during high-variance phases, producing 500+ launches instead of 80. Each launch costs 0.75ms of fixed overhead. 500 × 0.75 = 375ms of overhead alone — overhead that STATIC-LARGE and INVERTED do not pay. The GPU's own warp scheduler handles task cost variation within a large chunk better than software fragmentation does."

---

# Part VIII: Interview Questions You Will Be Asked

Based on IBM ISL, IBM Research, and systems engineering interview patterns:

## Architecture & Ownership Questions

**Q: "Walk me through the end-to-end data flow of this system."**
A: "User runs the exe with `--experiment C --multiplier 4000 --reps 20`. `main.cu` parses args, creates an `ExperimentConfig`, calls `preset_experiment_C()` to fill the phase distributions, then constructs an `ExperimentRunner`. The runner's `run()` method writes `config.json`, runs 3 warmup reps (discarded), then 20 final reps. Each rep generates a workload array (800K ints), uploads it to GPU via `CudaExecutor`, runs ADAPTIVE (pass 1), derives matched_chunk from the mean of adaptive's choices, runs STATIC-LARGE and STATIC-MATCHED in randomised order (pass 2), runs INVERTED (pass 3), and flushes one row to `raw_results.csv`. After all 20 reps, the runner computes arm statistics and three Wilcoxon tests, writes `statistical_results.json`. Then `analyze_v2.py` cross-checks against scipy and generates 7 diagnostic plots."

**Q: "Which modules did YOU write from scratch?"**
A: "Every `.h`, `.cpp`, and `.cu` file in `experiments/variance_adaptive_v2/src/` — 13 files total. The V2 framework shares no code with V1. I also wrote `analyze_v2.py`, `build_dashboard.py`, and the calibration system. The only external dependencies are the CUDA runtime, the C++ standard library, and Python's scipy/matplotlib/pandas."

**Q: "Why did you pick this tech stack over standard alternatives?"**
A: "CUDA because the target hardware is an NVIDIA GPU. C++ because CUDA extends C++ and the entire scheduling + timing loop must run with minimal overhead — Python would add milliseconds of interpreter latency to each decision point. The C++ Wilcoxon implementation avoids a Python bridge at runtime. Python is used only for post-hoc analysis and plotting, where millisecond latency doesn't matter."

## The "Why Not" Pattern

**Q: "Why didn't you use OpenMP dynamic scheduling?"**
A: "OpenMP's `schedule(dynamic, chunk_size)` adjusts work distribution across CPU threads, but it does not control GPU kernel launch granularity. Our experiment specifically measures the cost of repeated GPU kernel launches — which OpenMP does not address. Additionally, OpenMP cannot enforce the causal boundary or the two-pass arm protocol."

**Q: "Why a sliding window and not an exponential moving average?"**
A: See Section 7 — sharp regime detection vs. slow EMA forgetting.

**Q: "Why not use CUDA Streams or CUDA Graphs to reduce launch overhead?"**
A: "CUDA Streams allow overlapping computation and data transfer but do not reduce per-launch overhead for compute-bound kernels on WDDM. CUDA Graphs pre-record a fixed sequence of launches, reducing overhead — but the scheduler's chunk sizes change at runtime, so the graph would need to be rebuilt every iteration, negating the benefit. The correct solution for eliminating launch overhead is a persistent kernel with a work queue — which is the next architectural step."

**Q: "Why not just use one persistent kernel?"**
A: "A persistent kernel with an atomic work queue would eliminate launch overhead entirely — one launch, 0.75ms overhead total. But the research question was specifically about inter-launch scheduling decisions. Using a persistent kernel would answer a different question. However, now that we know inter-launch adaptation hurts, the persistent kernel is the clear next step for a production system."

## Verification & Ground Truth

**Q: "How did you verify your timing numbers were not measurement artifacts?"**
A: "Three mechanisms: (1) Calibration ensures compute time ≥ 10× launch overhead — measurement noise is at most 9% of the signal. (2) Two independent timing methods (CUDA events for GPU time, std::chrono for wall clock) provide cross-checks. (3) The C++ Wilcoxon implementation is cross-validated against scipy to within 0.01. If any math bug existed, the cross-check would catch it."

**Q: "How did you isolate host-to-device PCIe latency from GPU execution time?"**
A: "CUDA events (`cudaEventElapsedTime`) measure GPU-side time only — they exclude host/PCIe latency. The wall-clock `std::chrono` measurement includes everything. The difference (t_total_chunk - t_kernel ≈ 0.75ms) is the launch + sync overhead. Task cost data is uploaded once per repetition, not per chunk, so upload latency does not affect chunk-by-chunk timing."

**Q: "How do you know the warmup and thermal effects play a role?"**
A: "We measured it. Without inter-arm pauses, the second arm in each rep was systematically 3-5% faster than when run in isolation. GPU Boost 4.0 on the RTX 4070 increases clock frequency under sustained load. 200ms of idle + cudaDeviceSynchronize() allows the GPU to return to base clocks. With pauses, ordering effects drop to ~0.5% — within noise. The 3 warmup reps ensure the GPU's power management has stabilised (initial boost-to-base transients take ~1 second)."

## Retrospective

**Q: "Knowing what you know now, what is the single biggest design flaw?"**
A: "The fundamental assumption that reacting to variance by changing chunk size is the right lever. The correct response to high variance is not smaller chunks (more launches) but sorted tasks within large chunks (less warp divergence). If I were starting over, I would implement the sorting scheduler from day one and compare it against unsorted STATIC-LARGE."

**Q: "If you had another 3 months, what would you build?"**
A: "Three things: (1) A persistent kernel with an atomic work queue — eliminates all launch overhead, which is the dominant cost. (2) A within-chunk task sorting scheduler — directly addresses warp divergence, which is the actual GPU-level cost of workload variance. (3) Repeating the experiment on Linux TCC mode to see if the findings change when launch overhead drops from 750μs to ~5μs."
