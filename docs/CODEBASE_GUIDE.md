# Codebase Guide: Variance-Aware Adaptive GPU Chunking

**Who this is for:** Someone with no prior knowledge of GPUs, parallel computing, or systems
programming who wants to understand every decision in this codebase from the ground up.

**How to read this:** Start at Section 1. Every section assumes you read the previous one.
Do not skip ahead; each idea is built on the one before it.

---

## Table of Contents

1. The Problem This Project Solves (in Plain English)
2. Background Concepts You Must Understand First
   - 2.1 What is a GPU?
   - 2.2 What is a "Task" and a "Workload"?
   - 2.3 What is a "Chunk" and why does it exist?
   - 2.4 What is Variance (the Math)?
   - 2.5 What is a Sliding Window?
   - 2.6 What is a Causal System?
3. The Research Question
4. The Experimental Design (Why Three Arms?)
5. The Project is Actually Two Projects
6. Project V1: The Original Experiment (What Failed and Why)
7. Project V2: The Rigorous Rewrite
   - 7.1 Complete File Map
   - 7.2 How the Program Starts: `main.cu`
   - 7.3 What the Workload Looks Like: `workload_generator.h/.cpp`
   - 7.4 The Causal Boundary: `scheduler.h`
   - 7.5 The Data Container: `sliding_window.h`
   - 7.6 The Config Object: `config.h`
   - 7.7 The Adaptive Brain: `adaptive_scheduler.h`
   - 7.8 The Inverted Brain: `inverted_adaptive_scheduler.h`
   - 7.9 The Static Schedulers: `static_large_scheduler.h`
   - 7.10 The GPU Interface: `cuda_executor.h/.cu`
   - 7.11 The Orchestrator: `experiment_runner.h/.cpp`
   - 7.12 The Statistics Engine: `statistical_analysis.h/.cpp`
   - 7.13 Writing Results: `result_serializer.h/.cpp`
8. The Two Core Flows, Step by Step
   - Flow A: Running an Experiment End to End
   - Flow B: How a Single Scheduling Decision is Made
9. The Analysis Pipeline (Python)
   - 9.1 `analyze_v2.py`
   - 9.2 `build_dashboard.py`
   - 9.3 `profile_occupancy.py`
10. The Actual Results and What They Mean
11. Key Design Decisions and Why They Were Made
12. Known Gaps, Limitations, and Technical Debt
13. "If I Want to Change X, Edit Y" — Modification Map
14. Glossary
15. Ten Questions to Test Your Understanding

---

## 1. The Problem This Project Solves (in Plain English)

Imagine you have a very large list of tasks — 800,000 tasks — and each task requires a
different amount of computation. Some tasks are quick; some take a long time. You want to
run all of them on your GPU as fast as possible.

The GPU cannot run all 800,000 tasks in a single instruction. You have to group them into
**chunks** and send one chunk at a time. After each chunk finishes, you decide the size of
the next chunk.

**The question this project asks:** Can we make the program run faster by watching how
variable (unpredictable) the recent tasks were, and using that to choose a smarter chunk
size for the next batch?

The intuition behind the idea is:
- If recent tasks were all roughly the same cost → use a large chunk (less overhead,
  more tasks processed per round trip).
- If recent tasks were wildly different in cost → use a smaller chunk (get a decision
  point sooner to react).

The project builds this system, tests it rigorously, and discovers a surprising result:
the adaptive strategy is actually **slower**, not faster — and then investigates why.

---

## 2. Background Concepts You Must Understand First

### 2.1 What is a GPU?

Your computer has two kinds of processors:

**CPU (Central Processing Unit):** The main brain of your computer. It has a small number
of cores (typically 4–16 on a consumer machine). Each core is extremely powerful and fast
at running complex logic: conditionals, loops, function calls. A CPU core runs one task at
a time with great sophistication.

**GPU (Graphics Processing Unit):** Originally designed to compute images (pixels on a
screen), it has thousands of cores. Each core is individually much weaker than a CPU core,
but there are so many of them that for tasks you can run in parallel (the same operation
on many pieces of data at once), the GPU wins dramatically.

The GPU in this project is an **NVIDIA RTX 4070**, which has 5,888 shader processors.

**How you use a GPU from code:** You write a special function called a **kernel**. You
launch this function with a specific number of threads. All threads run the kernel code at
the same time (in parallel). Each thread is assigned a unique ID (`threadIdx`, `blockIdx`)
which it uses to know which piece of data to work on.

**Launching a kernel costs time even before it starts computing.** On Windows with a
standard GPU driver, this overhead is approximately 0.75 milliseconds per launch. This is
called **launch overhead**. If you launch many small kernels instead of fewer large ones,
you spend more total time on overhead. This fact is central to understanding the project's
results.

### 2.2 What is a "Task" and a "Workload"?

In this project, a **task** is a single unit of computation assigned to one GPU thread.

Each task has a **cost**: an integer that controls how many loop iterations that thread
will execute. A task with cost 200 runs a loop 200 × cost_multiplier times. A task with
cost 900 runs it 900 × cost_multiplier times.

The **workload** is the complete array of 800,000 task costs. It is generated on the CPU
before the GPU is involved. The costs are not uniform — they vary according to a
distribution that changes in four distinct phases, creating a realistic scenario of
fluctuating computational demand.

### 2.3 What is a "Chunk" and why does it exist?

You cannot pass all 800,000 tasks to the GPU at once in a single kernel launch for the
purposes of this experiment. The scheduler needs to make decisions between batches to
potentially adjust the batch size.

A **chunk** is a contiguous slice of tasks — for example, tasks 0 through 9,999. The
scheduler picks a chunk size, launches a kernel for that many tasks, waits for it to
finish, then decides the next chunk size.

Each kernel launch processes exactly one chunk. One chunk = one kernel launch.

**Why does chunk size matter?**

- **Large chunks:** Fewer kernel launches → less launch overhead. But the scheduler gets
  fewer opportunities to adjust between high-variance and low-variance phases.
- **Small chunks:** More frequent decision points. But more launches → more overhead.
  If the overhead per launch is large relative to the actual computation per chunk, small
  chunks make everything slower.

This tradeoff is the central tension of the entire project.

### 2.4 What is Variance (the Math)?

**Mean (average):** If you have a set of numbers, the mean is their sum divided by the
count. For example, the mean of {10, 20, 30} is (10+20+30)/3 = 20.

**Variance:** A measure of how spread out the numbers are around the mean.

The formula for **sample variance** (the version used in this project) is:

$$\sigma^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n - 1}$$

Where:
- $x_i$ is each individual value
- $\bar{x}$ is the mean of all values
- $n$ is the number of values

Step by step on the example {10, 20, 30}:
1. Mean $\bar{x}$ = 20
2. Deviations from mean: (10-20)=-10, (20-20)=0, (30-20)=10
3. Squared deviations: 100, 0, 100
4. Sum: 200
5. Divide by n-1 = 2: variance = **100**

Now consider {19, 20, 21}:
1. Mean = 20
2. Deviations: -1, 0, 1
3. Squared: 1, 0, 1
4. Sum: 2
5. Divide by 2: variance = **1**

Both sets have the same mean. But the first set (high variance = 100) is very spread out,
while the second (low variance = 1) is tightly clustered. **Variance tells you how
unpredictable the values are.**

Why divide by n-1 instead of n? This is called **Bessel's correction** and it makes the
sample variance an unbiased estimate of the true population variance. For this project,
the practical effect is minor, but the formula is mathematically correct.

**In this project:** The scheduler computes the variance of the most recent completed task
costs. A high variance means the tasks are wildly heterogeneous (some very cheap, some
very expensive). A low variance means the tasks are roughly uniform.

### 2.5 What is a Sliding Window?

A **sliding window** is a fixed-size buffer that always holds the N most recent values
you have observed.

Example with N=3:
- You observe: 10 → window = [10]
- You observe: 20 → window = [10, 20]
- You observe: 30 → window = [10, 20, 30]
- You observe: 40 → window = [20, 30, 40]  (10 dropped off the back)
- You observe: 50 → window = [30, 40, 50]

The window "slides" forward through time. At each point, you can compute the mean and
variance of only the recent observations, ignoring old ones. This lets you react to
regime changes: if the last 20 tasks were high-variance, the window's variance is high
even if the earlier 780,000 tasks were uniform.

In code, this is implemented in `src/sliding_window.h`.

### 2.6 What is a Causal System?

A system is **causal** if it can only use information from the past and present — never
from the future — when making a decision.

This seems obvious for a real-world scheduler (you cannot see the future), but it is
easy to accidentally violate in an experiment:

**Non-causal (wrong):** "Look at the execution time of the next chunk to decide the
chunk size." (This uses future information.)

**Also non-causal (a common bug):** "Compute the mean cost across the entire workload
and use that to set thresholds." (The entire workload includes future tasks you haven't
processed yet.)

**Causal (correct):** "Only look at tasks that have already finished execution to decide
the next chunk size."

This project enforces causality by creating a `CausalObserver` object — a wrapper around
the task costs array that **blocks access to any task beyond the last completed one**.
Attempting to read a future task cost causes the program to terminate immediately with an
assertion failure. This is not just a convention — it is enforced mechanically in code.

---

## 3. The Research Question

> Does a causal sliding-window variance-aware scheduler improve total end-to-end GPU
> workload execution time compared with a fixed-chunk scheduler that uses the **same
> average chunk size**?

The phrase "same average chunk size" is critical. If the adaptive scheduler uses an
average chunk of 2,000 and beats a static scheduler using 10,000, you cannot conclude
that *adaptiveness* helped — you could just conclude that *smaller chunks* helped.

The experiment neutralizes this by creating a **STATIC-MATCHED** arm: a fixed-chunk
scheduler whose chunk size is set to exactly the mean chunk size the adaptive scheduler
chose. If adaptive still wins over static-matched, then the *adaptiveness itself* — the
ability to change chunk size over time — is responsible for the improvement.

---

## 4. The Experimental Design (Why Three Arms, Now Four?)

The project compares four "arms" — four different scheduling strategies applied to the
identical workload within each repetition:

| Arm | Strategy | Chunk Size |
|-----|----------|------------|
| **ADAPTIVE** | Watches variance; shrinks when high, grows when low | Varies each decision |
| **STATIC-LARGE** | Never adapts; always uses the maximum | 10,000 (fixed) |
| **STATIC-MATCHED** | Never adapts; uses the mean of what adaptive chose | ~2,000–5,000 (fixed per rep) |
| **INVERTED-ADAPTIVE** | Watches variance; **grows** when high, **shrinks** when low | Varies (opposite policy) |

The **primary comparison** is STATIC-MATCHED vs ADAPTIVE. This is the controlled test
of whether adaptiveness helps.

The **secondary comparison** is STATIC-LARGE vs ADAPTIVE. This is a rougher test;
it shows the cost of many small launches versus few large ones.

The **tertiary comparison (mechanism test)** is INVERTED-ADAPTIVE vs ADAPTIVE. This
was added to test a specific hypothesis: if ADAPTIVE is slower because small chunks
starve the GPU of parallel work (warp occupancy starvation), then INVERTED-ADAPTIVE
(which grows chunks when variance is high, keeping the GPU fed) should recover
performance.

**Why 20 repetitions?**

Each repetition runs with a different random seed — a different draw of the workload from
the same statistical distribution. With only 3 repetitions, a single unusual result
would dominate. With 20, you get a stable picture. Specifically, the **Wilcoxon signed-
rank test** — the statistical test used here — requires at least ~10–20 samples to have
meaningful power.

**Why Wilcoxon and not a t-test?**

A t-test assumes your data is normally distributed (bell curve shaped). GPU timings are
not perfectly normal — they have occasional outliers from OS scheduling or thermal events.
The Wilcoxon signed-rank test makes no distributional assumption. It tests whether the
median of the paired differences is zero. It is more robust to outliers.

---

## 5. The Project is Actually Two Projects

The repository contains two separate, independent experiments:

```
variance_gpu_project/
├── src/                      ← V1: The original failed experiment
│   ├── static_chunking.cu
│   └── variance_chunking.cu
├── data/                     ← V1's recorded output CSVs (15 files)
├── analysis/
│   └── gpu_scheduling_analysis.ipynb  ← V1's Jupyter notebook
├── results/                  ← V1's plots
└── experiments/
    └── variance_adaptive_v2/ ← V2: Complete rigorous rewrite (untracked in Git)
        ├── src/              ← All C++ and CUDA source
        ├── analysis/         ← Python analysis scripts
        └── results/          ← Generated outputs
```

**V1** was the original project submitted for academic evaluation. It had serious
methodological flaws (described in Section 6).

**V2** is the complete rewrite. All 19 source files in `experiments/variance_adaptive_v2/`
were written from scratch. It is entirely in the `experiments/` folder and has never been
committed to Git (the directory appears as "untracked" in `git status`). V1 is preserved
and untouched in `src/`.

---

## 6. Project V1: The Original Experiment (What Failed and Why)

**Files:** `src/static_chunking.cu`, `src/variance_chunking.cu`

**What V1 did:** It measured actual GPU execution time for each chunk, added those
measurements to a sliding window, and used the variance of those timing measurements to
decide the next chunk size.

**Why this was wrong:**

1. **Timing noise feedback loop:** GPU execution time is noisy. Clock boosts, OS
   scheduling, WDDM driver batching, and thermal variation all add random fluctuations
   to timing. When the scheduler responded to this noise by shrinking chunk size, it
   created more launches, which added more overhead, which made timing more variable,
   which triggered more shrinking. The controller spiraled into using tiny chunks.

2. **No controlled comparison:** V1 compared ADAPTIVE against STATIC (10,000 chunks) and
   TUNED-ADAPTIVE (different thresholds). It never compared against a static scheduler
   using the *same average chunk size*. This means the results could not distinguish
   "smaller chunks helped" from "adaptive behavior helped."

3. **n=5 trials:** The reported results are averages of only 5 trials. With 5 data
   points, no statistical test has meaningful power. A 4.9% improvement reported with
   n=5 cannot be distinguished from random noise.

4. **No causal enforcement:** V1 used timing-based adaptation that technically looked at
   the output of recently completed chunks, which is causally valid. However, the core
   measurement — using elapsed GPU time rather than pre-generated task costs — meant the
   scheduler was reacting to measurement artifacts rather than the actual workload.

**V1 Results (from `README.md`):**

| Method | Mean Runtime (ms) |
|--------|:-----------------:|
| Static Chunking | 123.5 |
| Original Adaptive | 1459.9 |
| Tuned Adaptive | **117.4** |

The original adaptive was 12× slower than static due to excessive fragmentation. The
tuned version achieved a claimed 4.9% improvement. **These results are unreliable due to
the above methodological issues.**

---

## 7. Project V2: The Rigorous Rewrite

### 7.1 Complete File Map

```
experiments/variance_adaptive_v2/
├── src/
│   ├── main.cu                       Entry point; parses args; launches experiment
│   ├── config.h                      All tunable parameters in one place
│   ├── scheduler.h                   CausalObserver + abstract Scheduler interface
│   ├── sliding_window.h              Fixed-size ring buffer for variance computation
│   ├── workload_generator.h/.cpp     Generates the phased synthetic workload
│   ├── adaptive_scheduler.h          The ADAPTIVE arm (shrink when high-variance)
│   ├── inverted_adaptive_scheduler.h The INVERTED arm (grow when high-variance)
│   ├── static_large_scheduler.h      Both STATIC-LARGE and STATIC-MATCHED arms
│   ├── cuda_executor.h/.cu           The GPU kernel + launch timing machinery
│   ├── experiment_runner.h/.cpp      The 4-arm, multi-rep orchestrator
│   ├── statistical_analysis.h/.cpp   Wilcoxon test + detection lag computation
│   └── result_serializer.h/.cpp      Writes CSV/JSON output files
├── analysis/
│   ├── analyze_v2.py                 Python cross-check + 7 plots
│   ├── build_dashboard.py            HTML dashboard generator
│   └── profile_occupancy.py         Nsight Compute occupancy wrapper
├── results/
│   ├── calibration.json             Calibration sweep output
│   ├── dashboard.html               HTML summary page
│   ├── A_uniform/                   Outputs for Experiment A
│   ├── B_increasing_variance/       Outputs for Experiment B
│   ├── C_low_high_low_bursty/       Outputs for Experiment C (primary)
│   └── D_bursty/                    Outputs for Experiment D (partial)
├── variance_adaptive_v2.exe         The compiled main binary
├── variance_adaptive_v2_prof.exe    Compiled with --profile-occupancy flag
└── build.bat                        Windows build script
```

---

### 7.2 How the Program Starts: `main.cu`

**File:** `src/main.cu` (164 lines)

`main.cu` is the **entry point** — the first function that runs when you execute the
program. In C/C++, every program starts at a function called `main()`.

Because the project uses CUDA (NVIDIA's GPU programming framework), the file must have
the `.cu` extension so the `nvcc` compiler knows to handle GPU code inside it.

**The single-translation-unit trick:** Rather than compiling each `.cpp` file separately
and linking them together, `main.cu` directly `#include`s all other source files:

```cpp
#include "cuda_executor.cu"
#include "workload_generator.cpp"
#include "statistical_analysis.cpp"
#include "result_serializer.cpp"
#include "experiment_runner.cpp"
```

This means the entire project is compiled as if it were one giant file. The reason for
this unusual pattern is documented in the comment at the top:

> "This matches the old experiment's pattern and avoids nvcc's separate compilation
> mode which triggers cudaErrorCallRequiresNewerDriver (801) on this specific
> CUDA + driver + MSVC version combination."

Translation: there was a driver compatibility bug. Compiling as one file was the
reliable workaround.

**What `main()` does (in order):**

1. **Parse command-line arguments.** Arguments like `--experiment C`, `--reps 20`,
   `--multiplier 4000` are read from `argv[]` using simple string comparison.

2. **Build an `ExperimentConfig` object.** This is a struct (`config.h`) that holds
   every parameter the experiment needs.

3. **Apply a preset.** `preset_experiment_A(cfg)`, `preset_experiment_B(cfg)`, etc.
   populate the phase distributions. The default is `C` (Low→High→Low→Bursty).

4. **Branch on mode:**
   - `--calibrate`: Run the calibration sweep and exit.
   - `--profile-occupancy`: Run a tiny workload for Nsight Compute and exit.
   - Normal: Run `ExperimentRunner::run()` and exit.

---

### 7.3 What the Workload Looks Like: `workload_generator.h/.cpp`

**Files:** `src/workload_generator.h`, `src/workload_generator.cpp`

The workload generator creates the array of 800,000 integer task costs. Each integer
represents how much computation one GPU thread will perform.

**The four presets (A, B, C, D):**

Each preset defines four phases. For Experiment C (the primary experiment):

| Phase | Tasks | Distribution | Raw Cost Range | Variance Character |
|-------|-------|-------------|----------------|--------------------|
| C0 | 0–199,999 | Uniform[200, 400] | 200 to 400 | Low (~3,333) |
| C1 | 200,000–399,999 | Bimodal: 50% U[100,300] + 50% U[700,900] | 100 to 900 | Very High (>100,000) |
| C2 | 400,000–599,999 | Uniform[350, 450] | 350 to 450 | Low (~833) |
| C3 | 600,000–799,999 | Log-normal(μ=5.5, σ=0.9) clamped [100, 3000] | 100 to 3000 | Bursty |

**What is a bimodal distribution?** It is a distribution with two humps. Half the tasks
are cheap (100–300), half are expensive (700–900). The mean is ~500, but very few tasks
actually cost ~500. This creates high variance because values cluster far from the mean.

**What is log-normal?** If you take the logarithm of each value and get a normal (bell
curve) distribution, the original values are log-normal. This produces occasional very
large spikes — a few tasks cost 3000 while most cost 150–500.

**Reproducibility:** The random number generator (`std::mt19937`) is seeded with
`base_seed + rep * 1000003`. Using a prime stride (1,000,003) ensures different
repetitions produce genuinely independent workloads rather than correlated ones.

The workload is generated on the CPU. It is then **uploaded to the GPU once per
repetition** and all four scheduler arms share that same uploaded array. This ensures
every arm faces literally identical work.

---

### 7.4 The Causal Boundary: `scheduler.h`

**File:** `src/scheduler.h`

This file defines two things: `CausalObserver` and the abstract `Scheduler` interface.

**`CausalObserver` — the enforcement mechanism:**

```cpp
class CausalObserver {
public:
    CausalObserver(const int* costs, int num_completed)
        : all_costs_(costs), num_completed_(num_completed) {}

    int completed_cost(int task_index) const {
        assert(task_index < num_completed_);  // Hard stop if violated
        return all_costs_[task_index];
    }

    int num_completed() const { return num_completed_; }
    int window_start(int window_size) const {
        return std::max(0, num_completed_ - window_size);
    }
};
```

The object is constructed with a pointer to the full task costs array AND the number
of tasks completed so far. The `assert` on `completed_cost()` crashes the program if
any code tries to read a future task cost.

Before every scheduling decision, a **fresh** `CausalObserver` is created:

```cpp
// In experiment_runner.cpp, inside run_arm():
CausalObserver obs(task_costs.data(), processed);
int chunk = sched.select_chunk(obs, remaining);
```

`processed` is the number of tasks done so far. The observer gives the scheduler a
view of only those tasks. The scheduler cannot ask for `obs.completed_cost(processed)`
or beyond — the assert kills the process.

**`Scheduler` — the abstract interface:**

This defines what any scheduler must be able to do:

```cpp
class Scheduler {
public:
    virtual void reset() = 0;
    virtual int select_chunk(const CausalObserver& obs, int remaining) = 0;
    virtual std::string name() const = 0;
    virtual const std::vector<SchedulerDecision>& trace() const = 0;
    virtual void record_timing(double t_kernel_ms, double t_total_chunk_ms) = 0;
};
```

`= 0` means "any subclass **must** implement this." This is C++'s way of defining a
contract. The `ExperimentRunner` only ever holds `Scheduler*` references — it doesn't
care whether it's ADAPTIVE, INVERTED, or STATIC. It calls `select_chunk()` and gets
back a number.

**`SchedulerDecision` — what gets recorded per chunk:**

A struct that captures everything about one scheduling decision: the task range chosen,
the window variance that drove the decision, the chunk size selected, GPU timing, and
later, detection lag information.

---

### 7.5 The Data Container: `sliding_window.h`

**File:** `src/sliding_window.h`

This implements the sliding window described in Section 2.5.

It is a **ring buffer** (also called circular buffer): a fixed-size array where new
values overwrite the oldest ones. No memory allocation happens after construction.

Key methods:
- `push(value)` — add a value; if full, it overwrites the oldest
- `mean()` — returns the current mean
- `variance()` — returns the current sample variance
- `size()` — how many values are currently in the window
- `clear()` — reset to empty (called when a scheduler resets between repetitions)

The variance computation uses the two-pass formula: first compute the mean, then
compute the sum of squared deviations, then divide by n−1. This is numerically stable
enough for the values in this project.

---

### 7.6 The Config Object: `config.h`

**File:** `src/config.h`

This is a plain C++ struct that holds every tunable parameter. Rather than scattering
magic numbers throughout the code, all of them live here with names and default values.

Key parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `total_tasks` | 800,000 | Total number of tasks in the workload |
| `num_final_reps` | 20 | Number of timed repetitions |
| `num_warmup_reps` | 3 | Warmup runs before timing begins |
| `min_chunk` | 1,000 | Smallest chunk size allowed |
| `max_chunk` | 10,000 | Largest chunk size allowed (= STATIC-LARGE chunk) |
| `chunk_step` | 1,000 | How much to grow/shrink per decision |
| `window_size` | 20 | How many past task costs to include in variance |
| `var_low_thresh` | 5,000 | Variance below this → grow chunk |
| `var_high_thresh` | 30,000 | Variance above this → shrink chunk |
| `cost_multiplier` | 50 | Scales how much compute each task unit represents |
| `threads_per_block` | 256 | CUDA block size |
| `inter_arm_sleep_ms` | 200 | Milliseconds to pause between arms |
| `alpha` | 0.05 | Statistical significance threshold |
| `min_effect_fraction` | 0.05 | Must be 5% better to count as improvement |

The preset functions (`preset_experiment_A`, `preset_experiment_C`, etc.) live in
`workload_generator.cpp` and fill in the `phases` vector on this config.

---

### 7.7 The Adaptive Brain: `adaptive_scheduler.h`

**File:** `src/adaptive_scheduler.h` (163 lines)

This implements the ADAPTIVE arm. It extends the abstract `Scheduler`.

**State the scheduler holds:**
- `current_chunk_` — the current chunk size (starts at `max_chunk`)
- `window_` — a `SlidingWindow` object
- `trace_` — a vector of `SchedulerDecision` records (one per chunk launch)

**The `select_chunk()` method — the brain:**

```
Step 1: Rebuild the sliding window from the CausalObserver.
        Loop from window_start to num_completed, pushing each completed task cost.

Step 2: Apply the policy:
        If variance < VAR_LOW_THRESH  → grow chunk by CHUNK_STEP (up to MAX_CHUNK)
        If variance > VAR_HIGH_THRESH → shrink chunk by CHUNK_STEP (down to MIN_CHUNK)
        Otherwise                     → hold chunk unchanged

Step 3: Clamp to remaining work (don't request more tasks than exist).

Step 4: Record the decision in the trace.

Step 5: Return the chunk size.
```

**Important subtlety about the window:** Notice that Step 1 **rebuilds the entire window
from scratch** on every call. It does not maintain an incremental state. This is slightly
inefficient but simple to reason about and verify for correctness. The window holds at
most 20 values, so the rebuild is trivially fast.

**The `mean_chunk_size()` method:** After the scheduler finishes one full run (all chunks
processed), this computes the mean of all chunk sizes from the trace. The
`ExperimentRunner` uses this to set the `STATIC-MATCHED` arm's fixed chunk size.

---

### 7.8 The Inverted Brain: `inverted_adaptive_scheduler.h`

**File:** `src/inverted_adaptive_scheduler.h`

This is structurally identical to `adaptive_scheduler.h`. The single difference is that
the policy in `select_chunk()` is inverted:

```
ADAPTIVE (normal):
  If variance < low  → GROW chunk
  If variance > high → SHRINK chunk

INVERTED-ADAPTIVE:
  If variance < low  → SHRINK chunk
  If variance > high → GROW chunk
```

**Why invert the policy?** To test the **occupancy hypothesis**:

ADAPTIVE shrinks chunks during high-variance phases. The hypothesis is that this
starves the GPU warp scheduler of concurrent work, causing worse performance.
INVERTED-ADAPTIVE grows chunks during high-variance phases, giving the GPU more
threads to schedule, which should recover occupancy.

If the result shows INVERTED is significantly faster than ADAPTIVE during
high-variance workloads, the occupancy hypothesis is supported. The results
confirmed exactly this.

---

### 7.9 The Static Schedulers: `static_large_scheduler.h`

**File:** `src/static_large_scheduler.h`

This implements both STATIC-LARGE and STATIC-MATCHED. They are the same class,
parameterized at construction with their fixed chunk size:

```cpp
StaticScheduler static_large("static_large", cfg_.static_large_chunk);  // = 10,000
StaticScheduler static_matched("static_matched", matched_chunk);         // = adaptive mean
```

`select_chunk()` always returns the fixed chunk size, clamped to the remaining work.
It never looks at the `CausalObserver` at all. There is no adaptation.

These serve as the controlled baselines.

---

### 7.10 The GPU Interface: `cuda_executor.h/.cu`

**Files:** `src/cuda_executor.h`, `src/cuda_executor.cu` (210 lines)

This is the boundary between the scheduler (CPU-side logic) and the GPU.

**The CUDA kernel — `work_kernel`:**

```cuda
__global__ void work_kernel(
    const int* task_costs,   // Array of task costs (on GPU)
    int task_offset,         // Where this chunk starts in the array
    int chunk_size,          // How many tasks in this chunk
    volatile int* out,       // Output array (volatile prevents dead-code removal)
    int cost_multiplier)     // Scales work amount
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunk_size) return;  // Guard: threads beyond chunk do nothing

    int W = task_costs[task_offset + tid] * cost_multiplier;
    int acc = 0;
    for (int i = 0; i < W; ++i)
        acc += (i ^ W) & 0xFF;     // Compute-bound loop
    out[task_offset + tid] = acc;
}
```

`__global__` means this function runs on the GPU. Each thread computes its own `tid`,
looks up its task cost, and runs `W` iterations of a simple loop. The XOR and AND
operations are chosen to be cheap but non-trivial — the compiler cannot optimize the
loop away.

**Why `volatile int* out`?** Without `volatile`, a smart compiler might notice that
`out[...]` is written but never read again in this function and simply delete the write
(and the entire loop, since `acc` would then be dead). `volatile` forces the write to
actually happen, preserving the computation.

**The two timing measurements:**

1. **`t_kernel_ms`** — CUDA event elapsed time: timestamps recorded immediately before
   and after the kernel launch using `cudaEventRecord`. This measures only the GPU
   execution, excluding launch latency and host-side synchronization.

2. **`t_total_chunk_ms`** — Host wall-clock: measured using `std::chrono` from before
   the kernel launch to after `cudaDeviceSynchronize()`. This includes everything:
   launch overhead, actual GPU work, and the time spent waiting for the GPU.

`T_total` (the primary metric) is built from `t_total_chunk_ms` summed across all chunks.

**The CPU simulation fallback:**

If `cudaMalloc` fails (indicating the GPU is unavailable), the executor silently switches
to simulation mode. It estimates execution time using:

```cpp
double sim_ms = 12.6 + (total_warp_cost * cost_mult * 0.00000005);
```

The 12.6ms base represents the measured launch overhead. This was used during early
development when the GPU driver was not working. All actual results in this project
use the real GPU (`execution_mode: "cuda"` in config.json).

---

### 7.11 The Orchestrator: `experiment_runner.h/.cpp`

**Files:** `src/experiment_runner.h`, `src/experiment_runner.cpp` (450 lines)

This is the control center. It coordinates all four arms, manages timing, writes output,
and computes statistics.

**`run()` — the main loop:**

```
1. Write config.json (parameter snapshot for reproduction)
2. Run 3 warmup repetitions (discarded — warms up GPU, settles clocks)
3. For each of 20 final repetitions:
   a. Call run_one_repetition(rep, warmup=false)
   b. Store all four timing values
   c. Pause 200ms between reps (thermal settling)
4. Compute statistics for all four arms
5. Run three Wilcoxon tests (primary, secondary, tertiary)
6. Write statistical_results.json
7. Write detection_lag_results.json
```

**`run_one_repetition()` — one pass through the workload:**

```
1. Generate workload with seed = base_seed + rep * 1000003
2. Upload task costs to GPU (one upload, reused by all arms)
3. PASS 1: Run ADAPTIVE arm → get t_adaptive, trace, and mean_chunk
4. Derive matched_chunk = round(mean of adaptive chunk sizes)
5. Compute detection lags from adaptive trace
6. Pause 200ms
7. PASS 2 (randomized order):
   - If rep is even: run STATIC-LARGE then STATIC-MATCHED
   - If rep is odd: run STATIC-MATCHED then STATIC-LARGE
   - Pause 200ms between them
8. PASS 3: Run INVERTED-ADAPTIVE arm
9. Serialize all results immediately to CSV (crash-safe)
10. Return RepetitionResult struct
```

**`run_arm()` — one scheduler running through the whole workload:**

```
1. Reset the scheduler (clear sliding window, reset chunk to MAX)
2. Start wall clock
3. While there are unprocessed tasks:
   a. Build CausalObserver with current processed count
   b. Ask scheduler for next chunk size
   c. Launch CUDA kernel for that chunk
   d. Record timing back into the scheduler's trace
   e. Advance processed counter
4. Stop wall clock → this is t_total for this arm
5. Annotate trace with phase IDs
6. Write trace rows to scheduler_trace.csv
7. Return t_total
```

**The calibration sweep — `run_calibration()`:**

Before running experiments, you should find a `cost_multiplier` value where the GPU
actually has time to compute something meaningful. If `cost_multiplier` is too small,
each task takes less than a microsecond and the entire measured time is just launch
overhead — you'd be measuring overhead, not computation.

The calibration runs a loop testing multipliers 1, 10, 50, 100, 500, 1000, 2000, 4000...
until `T_kernel >= 10 * launch_overhead`. For this hardware (RTX 4070, WDDM mode,
launch overhead ~0.75ms), the required multiplier was 4000.

---

### 7.12 The Statistics Engine: `statistical_analysis.h/.cpp`

**Files:** `src/statistical_analysis.h`, `src/statistical_analysis.cpp`

This file contains:
1. `compute_arm_stats()` — computes mean, std, min, max for one arm's timings
2. `wilcoxon_signed_rank()` — the statistical test
3. `compute_detection_lags()` — measures scheduler reaction speed

**The Wilcoxon Signed-Rank Test — step by step:**

Given two sets of timings (e.g., ADAPTIVE and STATIC-MATCHED, 20 values each):

```
Step 1: Compute paired differences.
        d_i = ADAPTIVE_i - STATIC-MATCHED_i   (for each repetition i)

Step 2: Discard ties (where d_i = 0). Count remaining as N.

Step 3: Sort |d_i| (absolute values) in ascending order.
        Assign ranks 1 through N (smallest absolute difference = rank 1).
        For ties in |d_i|, use the average rank.

Step 4: W = sum of ranks where d_i > 0 (adaptive was slower).

Step 5: Compute Z-score:
        Expected mean  = N(N+1)/4
        Expected variance = N(N+1)(2N+1)/24
        Z = (W - mean) / sqrt(variance)
        (Apply continuity correction: subtract 0.5 from |W - mean|)

Step 6: p-value = 2 * (1 - Φ(|Z|))
        where Φ is the standard normal CDF.
        (Two-tailed: we test whether either arm is better, not specifically which.)

Step 7: Report effect size r = Z / sqrt(N), and mean paired difference with 95% CI.
```

A **p-value < 0.05** means: if there were truly no difference between the two arms,
the probability of observing a difference at least this large by chance is less than 5%.
We reject the null hypothesis.

An **effect size r > 0.5** is considered "large." An effect of 44% or 79% relative
difference is practically enormous.

This implementation is cross-validated by `analyze_v2.py` using `scipy.stats.wilcoxon`.

**Detection lag computation:**

For each known phase boundary (e.g., task 200,000 where phase C0 ends and C1 begins):

- `observation_lag_tasks`: How many tasks passed before the sliding window's variance
  entered the new regime's territory?
- `action_lag_tasks`: How many tasks passed before the chunk size moved by at least 25%
  toward the new regime's target size?

These quantities reveal whether the scheduler reacts in time to be useful. If the
phase is only 200,000 tasks long but the scheduler takes 15,000 tasks to react, that
means 7.5% of the phase is spent with the wrong chunk size.

---

### 7.13 Writing Results: `result_serializer.h/.cpp`

**Files:** `src/result_serializer.h`, `src/result_serializer.cpp`

This handles writing all outputs to disk. It is constructed once per experiment with
the output directory path, and it opens several files at construction:

- `raw_results.csv` — one row per repetition, columns: rep, seed, t_adaptive_ms,
  t_static_large_ms, t_static_matched_ms, t_inverted_adaptive_ms, plus chunk counts
- `scheduler_trace.csv` — one row per chunk launch across all reps and arms

After the experiment, it writes:
- `config.json` — full parameter snapshot
- `workload.csv` — the task costs and phase IDs for rep 0
- `statistical_results.json` — mean, std, Wilcoxon test results for all arms
- `detection_lag_results.json` — observation and action lags per phase transition
- `calibration.json` — only written by `--calibrate` mode

**Important detail: crash safety.** After every repetition, `raw_results.csv` is
flushed to disk immediately. If the experiment crashes on rep 14 of 20, you still have
reps 0–13 saved and can analyze partial results.

---

## 8. The Two Core Flows, Step by Step

### Flow A: Running Experiment C with 20 Reps

```
User runs: variance_adaptive_v2.exe --experiment C --multiplier 4000 --reps 20 --warmup 3

main():
  Parse args → exp_name="C", cost_multiplier=4000, etc.
  Create ExperimentConfig cfg
  preset_experiment_C(cfg) → fills phases: Uniform, Bimodal, Uniform, LogNormal
  Create ExperimentRunner runner(cfg, "results/C_low_high_low_bursty")
  runner.run()
    │
    ├─ Write config.json
    │
    ├─ Warmup loop (3 reps, results discarded):
    │    For each warmup rep:
    │      run_one_repetition(-(w+1), warmup=true)
    │        Generate workload (seed changes each rep)
    │        Upload to GPU
    │        Run ADAPTIVE → throw away result
    │        Run STATIC-LARGE → throw away result
    │        Run STATIC-MATCHED → throw away result
    │        Run INVERTED → throw away result
    │
    ├─ Final measurement loop (20 reps):
    │    For rep 0..19:
    │      run_one_repetition(r, warmup=false)
    │        Generate workload (seed = 42 + r * 1000003)
    │        Upload to GPU
    │        Run ADAPTIVE → record t_adaptive, get mean_chunk=2025
    │        Pause 200ms
    │        If r even: Run STATIC-LARGE, pause, Run STATIC-MATCHED
    │        If r odd:  Run STATIC-MATCHED, pause, Run STATIC-LARGE
    │        Pause 200ms
    │        Run INVERTED → record t_inverted
    │        Write one row to raw_results.csv (flush)
    │        Return RepetitionResult
    │      Store t values in times_adaptive[], times_matched[], etc.
    │      Pause 200ms
    │
    ├─ compute_arm_stats for each arm
    ├─ wilcoxon_signed_rank for primary, secondary, tertiary
    ├─ Write statistical_results.json
    ├─ Write detection_lag_results.json
    └─ Print summary to console
```

### Flow B: How One Scheduling Decision is Made (ADAPTIVE)

```
Scenario: 400,000 tasks have been processed. The scheduler must decide the next chunk.

run_arm() has:
  processed = 400,000
  total = 800,000

Step 1: Build observer.
  CausalObserver obs(task_costs.data(), 400000)
  obs can only read costs[0..399999]. Costs[400000..] are forbidden.

Step 2: Call scheduler.
  adaptive.select_chunk(obs, remaining=400000)

    Inside select_chunk():
      window_start = max(0, 400000 - 20) = 399980
      window_end   = 400000 (exclusive)

      Clear window.
      Push costs[399980], costs[399981], ..., costs[399999]  → 20 values

      These 20 tasks are in Phase C1 (Bimodal).
      Each cost is either ~200 or ~800.
      Sample variance ≈ 110,000 (very high)

      110,000 > VAR_HIGH_THRESH (30,000) → SHRINK
      current_chunk = max(current_chunk - 1000, 1000)

      If current_chunk was 5000 → now 4000.
      chunk = min(4000, 400000) = 4000

      Record SchedulerDecision{task_start=400000, chunk_size=4000, window_variance=110000}

      Return 4000

Step 3: Launch kernel.
  CudaExecutor::launch_chunk(400000, 4000)
    Record CUDA event ev_start
    Launch work_kernel<<<16, 256>>>(d_tasks, 400000, 4000, d_out, 4000)
    Record CUDA event ev_stop
    cudaDeviceSynchronize()   ← Wait for GPU to finish
    Measure t_kernel from events, t_total from wall clock
    Return ChunkResult{t_kernel_ms=3.2, t_total_chunk_ms=3.95}

Step 4: Record timing.
  adaptive.record_timing(3.2, 3.95)
  → Updates the last SchedulerDecision in the trace

Step 5: Advance.
  processed = 400,000 + 4,000 = 404,000

Step 6: Loop. Next decision uses obs with num_completed = 404,000.
```

---

## 9. The Analysis Pipeline (Python)

### 9.1 `analyze_v2.py`

**File:** `analysis/analyze_v2.py` (~483 lines)

Run with: `python analysis\analyze_v2.py --experiment C`

It finds the results directory (e.g., `results/C_low_high_low_bursty`), loads:
- `raw_results.csv` — the per-rep timings
- `scheduler_trace.csv` — the per-chunk decisions
- `statistical_results.json` — the C++ Wilcoxon output
- `config.json` — the parameters

**Step 1 — Cross-check:** It runs `scipy.stats.wilcoxon` independently on the same
timing data. It then checks whether the C++ p-values are within 0.01 of the scipy p-
values. If they diverge, it prints `[MISMATCH]`. This catches bugs in the C++ math.
All current results pass this check.

**Step 2 — Generate 7 plots:**

| Plot | What it shows |
|------|--------------|
| `01_task_cost_trace.png` | The raw workload: cost of each task, colored by phase |
| `02_window_variance_trace.png` | How variance evolved over time as the scheduler saw the workload |
| `03_chunk_size_trace.png` | The chunk sizes chosen by each arm over time |
| `04_chunk_timeline.png` | A Gantt-chart-like view of chunk boundaries |
| `05_total_runtime_bars.png` | Mean ± std T_total for all 4 arms (the headline result) |
| `06_scheduling_decisions.png` | Adaptive + Inverted chunk size traces overlaid on phase bands |
| `07_detection_lag.png` | How quickly the scheduler reacted at each phase boundary |

### 9.2 `build_dashboard.py`

**File:** `analysis/build_dashboard.py`

Run with: `python analysis\build_dashboard.py`

Scans all results directories (A, B, C, D), embeds the bar chart and chunk trace images
as base64 data URIs (so the HTML file is self-contained — no external dependencies),
and builds a table of the three Wilcoxon comparisons per experiment. Outputs to
`results/dashboard.html`.

### 9.3 `profile_occupancy.py`

**File:** `analysis/profile_occupancy.py`

Run with: `python analysis\profile_occupancy.py --experiment C`

Checks if NVIDIA Nsight Compute (`ncu`) is installed. If not, prints:

> "Nsight Compute (ncu) is not available on the system. Skipping occupancy profiling."

If `ncu` is available, it runs the `variance_adaptive_v2_prof.exe` binary (compiled with
the `--profile-occupancy` flag) under Nsight Compute, extracting the metric
`sm__warps_active.avg.pct_of_peak_sustained_active` at chunk sizes
{1000, 2000, 3000, 5000, 7500, 10000}. This tells you what fraction of the GPU's warp
schedulers were active at each chunk size — directly measuring the occupancy starvation
that the INVERTED experiment was designed to detect.

**On this hardware, Nsight Compute is not installed.** The tool exits cleanly.

---

## 10. The Actual Results and What They Mean

All results are from real GPU execution (`execution_mode: "cuda"`, RTX 4070).

### Experiment C (Low→High→Low→Bursty) — n=20

| Arm | Mean T_total (ms) | Std (ms) |
|-----|:-----------------:|:--------:|
| ADAPTIVE | 9,941.6 | 251.0 |
| STATIC-MATCHED | 6,876.6 | 116.9 |
| STATIC-LARGE | 2,724.7 | 22.9 |
| INVERTED-ADAPTIVE | 4,550.7 | 36.5 |

| Comparison | p-value | Relative Difference | Significant? |
|------------|:-------:|:-------------------:|:------------:|
| MATCHED vs ADAPTIVE (primary) | 0.0001 | −44.6% | Yes |
| LARGE vs ADAPTIVE (secondary) | 0.0001 | −264.9% | Yes |
| INVERTED vs ADAPTIVE (tertiary) | 0.0001 | −118.5% | Yes |

**What this means:**

ADAPTIVE is the **worst** scheduler. It is 44.6% slower than STATIC-MATCHED (which
uses the exact same average chunk size but without adapting). This means adaptive behavior
— the very thing the project set out to show is beneficial — actually **hurts**.

Why? The ADAPTIVE scheduler detects high variance in Phase C1 (the bimodal phase) and
responds by shrinking chunk size from 10,000 down to 1,000 over several decisions. With
1,000 tasks per chunk, there are 800 kernel launches for the bimodal phase alone. Each
launch incurs ~0.75ms overhead on WDDM mode. That's ~600ms of pure overhead for the
bimodal phase — before any computation.

INVERTED-ADAPTIVE does the opposite: when variance is high, it grows chunks (up to
10,000). It takes 80 launches for the whole workload regardless of variance. Less overhead
→ faster. It finishes the workload in 4,550ms versus ADAPTIVE's 9,941ms.

**The key insight:** On WDDM (Windows Display Driver Model), kernel launch overhead is
the dominant cost at these compute scales. Any strategy that increases the number of
kernel launches will be slower. "React to variance by shrinking chunks" directly increases
kernel launches → directly hurts performance.

### Experiment B (Increasing Variance) — n=7 (interrupted)

| Arm | Mean T_total (ms) |
|-----|:-----------------:|
| ADAPTIVE | 11,823.0 |
| STATIC-MATCHED | 7,669.7 |
| STATIC-LARGE | 2,757.1 |
| INVERTED-ADAPTIVE | 4,950.9 |

Same pattern. ADAPTIVE is worst; STATIC-LARGE is best; INVERTED is between the two.

### Experiment A (Uniform workload) — n=20

| Arm | Mean T_total (ms) |
|-----|:-----------------:|
| ADAPTIVE | 998.8 |
| STATIC-MATCHED | 998.0 |
| STATIC-LARGE | 998.7 |
| INVERTED-ADAPTIVE | 4,764.1 |

With a uniform workload (no variance to react to), ADAPTIVE, MATCHED, and LARGE are
essentially identical. The adaptive scheduler quickly grows to MAX_CHUNK (10,000) and
stays there — it degenerates into STATIC-LARGE behavior. All three finish in ~1 second.

INVERTED-ADAPTIVE, however, detects low variance and keeps shrinking chunk size to
MIN_CHUNK (1,000), creating 10× more launches → 10× slower. This confirms INVERTED is
genuinely doing the opposite of the intended optimal behavior on uniform workloads.

---

## 11. Key Design Decisions and Why They Were Made

**Decision 1: Use task costs (not GPU timings) for variance computation.**

The original experiment used GPU timing. This created a feedback loop where timing noise
triggered chunk shrinkage, which increased overhead, which made timing noisier. Using
pre-generated task costs eliminates this noise source entirely. The scheduler now reacts
to true workload heterogeneity, not measurement error.

**Decision 2: STATIC-MATCHED as the primary comparison arm.**

This is the most intellectually honest comparison. Without it, you could claim "adaptive
is better" when actually "smaller chunks happen to be better for this workload" —
conflating adaptiveness with magnitude. By matching the average chunk size, you isolate
the specific value of changing chunk size over time.

**Decision 3: Single translation unit compilation (`#include "*.cpp"` in `main.cu`).**

A driver compatibility bug (`cudaError 801`) occurred with standard separate compilation.
Compiling everything as one unit through `nvcc` eliminates the device linking step that
triggered the bug. This is unusual but legitimate — it is effectively how small CUDA
projects were structured before separate compilation support matured.

**Decision 4: The INVERTED-ADAPTIVE arm.**

Added as a mechanism test after the initial results showed ADAPTIVE loses. By applying
the opposite policy and observing that it is significantly faster, we can conclude the
mechanism causing ADAPTIVE to lose is specifically the reduction in chunk size (and
corresponding increase in kernel launches) during high-variance phases — not some other
artifact of the adaptive machinery.

**Decision 5: 200ms inter-arm pause.**

Without pausing between arms, the second arm to run benefits from a warmer GPU (clock
boost from the first arm). The third arm might be slower due to thermal throttling.
200ms of host sleep + `cudaDeviceSynchronize()` allows the GPU voltage and thermals to
stabilize before each arm's T_total measurement begins.

**Decision 6: Arm order randomization with `rep % 2`.**

Even with thermal pauses, there might be subtle ordering effects. By alternating the order
of STATIC-LARGE and STATIC-MATCHED between repetitions, any remaining bias cancels out
in aggregate rather than systematically favoring one arm.

---

## 12. Known Gaps, Limitations, and Technical Debt

**Severity: High**

1. **Experiment D incomplete.** Only 9 repetitions completed before the run was
   interrupted. `D_bursty` has no `statistical_results.json` or plots. Do not draw
   conclusions from D.

2. **No occupancy measurement.** Nsight Compute is not installed. The occupancy
   starvation hypothesis (Section 10) is strongly supported by timing data but lacks
   direct confirmation from occupancy metrics.

3. **Single GPU, single OS configuration.** All results are from RTX 4070 under Windows
   WDDM mode (~0.75ms launch overhead). On Linux (TCC mode, ~microseconds overhead),
   the results would likely be entirely different. The "ADAPTIVE hurts because of launch
   overhead" conclusion may not generalize.

**Severity: Medium**

4. **No formal unit tests.** The only automated test is the scipy vs. C++ Wilcoxon
   cross-check in `analyze_v2.py`. Correctness of the workload generator, causal
   boundary, and sliding window is verified only by code review and manual inspection.

5. **`test2.cu` is dead code.** It is a 30-line CUDA allocation sanity check that was
   written during debugging. It is not compiled, not documented, and not relevant to
   the experiment. It should be deleted or moved to a `debug/` folder.

6. **README instructions contain outdated multiplier.** The "How to Run" section in
   `README.md` says `--multiplier 50`, but actual experiments require `--multiplier 4000`.
   A reader following the README verbatim would get meaningless results.

7. **`variance_adaptive_v2/` not tracked in Git.** The entire V2 experiment is untracked.
   If the `experiments/` directory were deleted, the full history of V2 design decisions
   would be unrecoverable. **Commit V2 to Git.**

**Severity: Low**

8. **Variance thresholds are hardcoded in two places.** `var_low_thresh = 5000` and
   `var_high_thresh = 30000` appear in both `main.cu` (as defaults) and
   `experiment_runner.cpp` (in `run_calibration()`). If you change one, you must change
   both. A single constant in `config.h` would be cleaner.

9. **The calibration sweep has a fixed set of initial candidates** (1, 10, 50, 100, 500)
   before switching to doubling. If the target falls between 50 and 100 or between 100
   and 500, the sweep would skip past it. The doubling growth after 500 is correct.

10. **`const_cast` in `experiment_runner.cpp`.** The phase annotation code uses
    `const_cast<vector<...>&>(sched.trace())` to modify trace data post-hoc. This is
    flagged in the comment as "ugly." A cleaner design would expose a non-const annotate
    method on the scheduler interface.

---

## 13. "If I Want to Change X, Edit Y" — Modification Map

| What you want to change | Where to edit | What to be careful about |
|------------------------|---------------|--------------------------|
| Number of repetitions | `main.cu` default arg for `--reps`, or pass `--reps 30` | Fewer than 20 triggers a WARNING |
| Variance thresholds | `config.h` (struct fields) + `main.cu` (lines ~117-118) + `experiment_runner.cpp` (lines ~432-433) | Must update all three occurrences |
| Chunk bounds (MIN/MAX) | `config.h` fields `min_chunk`, `max_chunk` | Recompile after changing |
| Add a new workload preset (E) | `workload_generator.cpp`: add `preset_experiment_E()` + register in `main.cu` (`else if (exp_name == "E")`) | Also update analysis Python scripts to recognize "E" |
| Add a new scheduler arm | Create `new_scheduler.h`, include in `experiment_runner.h`, add pass in `run_one_repetition()`, add column to `result_serializer.cpp` and its header | Also update Python analyze_v2.py comparisons list |
| Change statistical test | `statistical_analysis.cpp`: replace `wilcoxon_signed_rank()` body | Also update scipy cross-check in `analyze_v2.py` |
| Add a new output column | `result_serializer.h` method signatures + `result_serializer.cpp` CSV header + implementation | Must also update `analyze_v2.py` column references |
| Add a new plot | `analyze_v2.py`: add a `plot_xxx()` function and call it in `main()` | Follows the same pandas/matplotlib pattern as existing plots |
| Change the thermal settling pause | `config.h` field `inter_arm_sleep_ms` | Longer is safer; shorter may reintroduce bias |
| Run on a different experiment workload | Pass `--experiment A/B/C/D` at the command line | No recompilation needed |

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **Arm** | One scheduling strategy in a controlled experiment. Each repetition runs all arms on the same workload. |
| **Block (CUDA)** | A group of up to 1,024 threads that can share fast memory and synchronize with each other. A kernel launch specifies how many blocks to create. |
| **Causal** | A system that only uses past information when making decisions. Cannot use future data. |
| **Chunk** | A contiguous slice of tasks processed in a single kernel launch. |
| **CausalObserver** | A wrapper around the task cost array that mechanically blocks access to uncompleted tasks. |
| **cost_multiplier** | A scaling factor applied to each task's iteration count, controlling how much real computation each task performs. |
| **Kernel** | A CUDA function that runs in parallel on the GPU. Defined with `__global__`. |
| **Launch overhead** | The fixed time cost of submitting a kernel to the GPU, independent of the computation it performs. ~0.75ms on WDDM. |
| **Mean** | The arithmetic average of a set of values. |
| **NVCC** | NVIDIA's CUDA C++ compiler. Compiles `.cu` files. |
| **Occupancy** | The fraction of the GPU's warp schedulers that are actively executing threads. Low occupancy = wasted hardware. |
| **p-value** | The probability of observing the measured difference by chance if the null hypothesis (no difference) were true. p < 0.05 is conventionally "significant." |
| **Phase** | A contiguous range of tasks drawn from the same distribution. The workload has four phases per repetition. |
| **Repetition (rep)** | One complete pass through the workload with all four arms. 20 repetitions → 20 data points per arm. |
| **Ring buffer** | A fixed-size array where new elements overwrite the oldest ones. Used to implement the sliding window efficiently. |
| **Sample variance** | Variance computed by dividing by n−1 (Bessel's correction). Unbiased estimate of the true population variance. |
| **Scheduler** | The decision-making component that chooses the next chunk size. Implements the abstract `Scheduler` interface. |
| **Sliding window** | A buffer holding the N most recent observed values. New observations push out the oldest. |
| **T_kernel** | GPU-side execution time, measured using CUDA events. Excludes launch overhead and synchronization. |
| **T_total** | Total wall-clock time for one arm to process the full workload. The primary metric. |
| **Thread** | The smallest unit of execution in CUDA. One thread = one GPU core doing one task. |
| **Variance** | A measure of how spread out values are around their mean. High variance = unpredictable. |
| **Warp** | A group of 32 threads that execute in lockstep on the GPU. If they have different amounts of work, they all wait for the slowest one. |
| **WDDM** | Windows Display Driver Model — the standard GPU driver mode on Windows. Adds significant per-launch overhead compared to Linux TCC mode. |
| **Wilcoxon signed-rank test** | A non-parametric statistical test for whether the median difference between paired observations is zero. Robust to non-normal distributions. |

---

## 15. Ten Questions to Test Your Understanding

1. Experiment C runs with `--multiplier 4000`. The README's "How to Run" section says
   `--multiplier 50`. What would happen if you followed the README literally, and why?

2. ADAPTIVE uses a sliding window of size 20. At the moment tasks 0–9,999 have been
   processed (the first 10,000 tasks), what is the window content, and why is it smaller
   than 20?

3. Why does the INVERTED-ADAPTIVE arm finish in ~1 second on Experiment A (uniform
   workload) but ~4.7 seconds on Experiment C?

4. Explain why STATIC-MATCHED is a better baseline for testing the value of adaptiveness
   than STATIC-LARGE.

5. In `cuda_executor.cu`, the output array is declared `volatile int*`. What would
   happen if the `volatile` keyword were removed?

6. The calibration sweep found `cost_multiplier = 4000` satisfying the 10× criterion.
   What would happen if you ran the experiment with `cost_multiplier = 50` (the old
   default) instead? Which failure mode from the README would this trigger?

7. `run_one_repetition()` always runs the ADAPTIVE arm first. Why must ADAPTIVE run
   before STATIC-MATCHED? Why can INVERTED-ADAPTIVE run in any order?

8. You observe that for repetition 7 of Experiment C, the ADAPTIVE arm chose chunks of
   {10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000} for the first 10
   decisions. What does this tell you about the workload distribution in those first
   decisions? What is the sliding window seeing?

9. The `CausalObserver` uses `assert(task_index < num_completed_)`. In production C++,
   `assert` is typically disabled in release builds with `-DNDEBUG`. What would be the
   consequence of disabling the assert in this project, and is it safe to do so?

10. After Experiment C, the detection lag results show `action_lag_tasks = 12,000` at
    the Phase 0→1 boundary. The chunk size at transition was 10,000. Explain why the
    action lag must be at least 10,000 tasks, and what accounts for the additional 2,000.

---

*Document generated from direct code inspection of the `variance_gpu_project` codebase.
All code citations reference confirmed file contents. Claims labeled [INFERRED] were not
directly visible in the source.*
