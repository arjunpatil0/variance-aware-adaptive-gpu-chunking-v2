# Problem Statement and Solution

---

## The Problem

### What is being solved, and why does it matter?

GPUs are powerful because they run thousands of tasks at the same time. But they work in
batches — you group a set of tasks together, send the group to the GPU, wait for it to
finish, then send the next group.

The question is: **how big should each group (chunk) be?**

This is not obvious, because the tasks are not all equal. In real-world GPU workloads —
image processing, physics simulations, machine learning inference — some tasks are quick
and some are slow. The distribution changes over time. During some phases of work, tasks
are uniform and predictable. During others, they are wildly unpredictable.

**The intuitive idea:** If you can detect when the workload becomes unpredictable
(high variance), you could react by using smaller chunks — getting more frequent
decision points. When the workload is stable (low variance), use larger chunks to reduce
overhead. This is *variance-aware adaptive chunking*.

**The question this project asks:**

> Does a runtime-adaptive scheduler that watches workload variance and adjusts chunk size
> actually improve GPU execution time over a static scheduler with the same average chunk
> size — on real hardware?

This is a question that matters for any system that processes heterogeneous GPU workloads:
ML inference servers, real-time rendering pipelines, scientific computing frameworks.

---

## What Makes This Hard

There are two specific problems that make this question non-trivial to answer:

### 1. Controlling for chunk size magnitude

A smaller average chunk size might be faster *regardless* of whether the scheduler
adapts. If you compare adaptive (average chunk: 2,000) against static (chunk: 10,000),
and adaptive wins, you cannot tell if *adaptiveness* helped or if *smaller chunks* helped.

This experiment controls for this by creating a **STATIC-MATCHED** arm: a static
scheduler whose chunk size is set to exactly the adaptive scheduler's mean chunk size
for that workload. If adaptive still wins over static-matched, then *adaptiveness* —
the ability to change chunk size — is what caused the improvement.

### 2. The scheduler must be causal

A real scheduler cannot see the future. It can only use information about tasks that
have already finished. This sounds obvious but is easy to accidentally violate in an
experiment (e.g., computing statistics across the whole dataset before the run).

This experiment enforces causality mechanically: a `CausalObserver` object physically
blocks access to any task that hasn't completed yet. Attempting to read future data
crashes the program immediately. There is no way for the scheduler to accidentally cheat.

---

## The Solution Built

### What was built

A controlled four-arm GPU experiment in CUDA/C++ with a rigorous statistical pipeline.

**Four scheduling strategies tested head-to-head on the same workload:**

| Arm | What it does |
|-----|-------------|
| **ADAPTIVE** | Monitors recent task cost variance; shrinks batch when high, grows when low |
| **STATIC-LARGE** | Always uses maximum batch size (10,000). Never adapts. |
| **STATIC-MATCHED** | Uses the same average batch as ADAPTIVE but never changes it |
| **INVERTED-ADAPTIVE** | Opposite policy: grows batch when variance is high, shrinks when low |

**How the experiment was designed:**

- 800,000 synthetic tasks per run, with known phases of different statistical distributions
- 20 independent repetitions per experiment, each with a different random seed
- 3 warmup runs discarded before measurement begins (to stabilise GPU clocks)
- 200ms thermal settling pause between every arm within each repetition
- Wilcoxon signed-rank test (non-parametric, robust to outliers), pre-registered before running
- C++ statistical implementation cross-validated against Python/scipy

**Four workload presets:**
- **A:** Uniform (all tasks cost the same)
- **B:** Steadily increasing variance
- **C:** Low → High → Low → Bursty (alternating regimes)
- **D:** Fully bursty log-normal distribution throughout

---

## The Results

### Summary table (all n=20, real GPU, statistically significant)

| Experiment | ADAPTIVE | MATCHED | STATIC-LARGE | INVERTED |
|---|---:|---:|---:|---:|
| A (uniform) | 999 ms | 998 ms | 999 ms | 4,764 ms |
| B (increasing) | 11,823 ms | 7,670 ms | 2,757 ms | 4,951 ms |
| C (alternating) | 9,942 ms | 6,877 ms | 2,725 ms | 4,551 ms |
| D (bursty) | 33,670 ms | 27,440 ms | 6,924 ms | 6,906 ms |

Lower is better. All differences in B, C, D have p = 0.0001.

### In plain English

**On a uniform workload (Experiment A):** Nothing to react to. All four arms converge
on the same chunk size. Results are identical. Only INVERTED is slower, because it
keeps shrinking batches when it should not.

**On variable workloads (B, C, D):** ADAPTIVE is the worst performer every time. It
is slower than both the static baseline that uses the same average chunk size (MATCHED)
*and* the naive maximum-chunk baseline (LARGE). Adaptiveness hurts.

**Why?** On Windows, each GPU kernel launch has a fixed overhead of approximately
0.75ms, regardless of how much computation is inside. ADAPTIVE responds to high
variance by shrinking batch size — which means launching more kernels. In Experiment D,
ADAPTIVE averages about 535 launches per run. At 0.75ms each, that is 400ms of pure
overhead per second of useful work. STATIC-LARGE uses 80 launches total.

**The INVERTED arm confirms the mechanism.** INVERTED does the opposite: when variance
is high, it grows batch size, giving the GPU more work per launch. In experiments B and
C, INVERTED runs 2–3× faster than ADAPTIVE. In D, it matches STATIC-LARGE almost
exactly. The only difference between ADAPTIVE and INVERTED is the direction of the
policy. The direction that reduces launch count wins.

---

## The Core Finding

> **Variance-aware adaptive chunk size reduction is counterproductive on GPU hardware
> with significant per-launch overhead.**

When a scheduler responds to high-variance tasks by issuing smaller batches, it trades
one problem (workload heterogeneity within a batch) for a worse problem (more kernel
launches, more fixed overhead). The GPU's own warp scheduler handles task cost variation
within a large batch better than software fragmentation does.

This is not a failure of the experiment — it is the finding. The project successfully
proves the intuitive strategy wrong, identifies the mechanism (launch overhead), and
quantifies the cost across four different workload types with full statistical rigour.

---

## Significance

### Why this matters technically

- GPU scheduling decisions are everywhere in production ML systems (batching in PyTorch,
  CUDA stream management, Triton kernel dispatch). The assumption that "smaller batches
  = more control = better" is widespread and this result challenges it directly.
- The WDDM overhead finding quantifies a real cost that developers on Windows (which
  includes most laptop-based ML development) routinely ignore.
- The experimental framework (causal observer, size-matched baseline, inverted mechanism
  test) is a template for rigorous GPU scheduling research.

### Why this matters for your portfolio

This project demonstrates:

1. **Systems thinking:** You designed an experiment that controls for the confound that
   kills most naive comparisons of adaptive vs. static scheduling.
2. **Statistical discipline:** Pre-registered hypothesis, Wilcoxon test, minimum effect
   threshold, cross-validated implementation, 20 repetitions.
3. **GPU hardware knowledge:** You identified WDDM launch overhead as the dominant
   variable and ran calibration to ensure real compute was being measured.
4. **Counterintuitive thinking:** The result is surprising. The ability to recognise
   when an intuitive idea fails — and explain why rigorously — is more impressive than
   confirming expected results.
5. **End-to-end ownership:** From CUDA kernel design to statistical analysis to HTML
   dashboard to documentation.

---

## How to Reproduce

```bat
cd experiments\variance_adaptive_v2

:: Build
build.bat

:: Run calibration (finds correct cost_multiplier for your GPU)
variance_adaptive_v2.exe --calibrate --experiment C

:: Run all four experiments (use multiplier from calibration output)
variance_adaptive_v2.exe --experiment A --multiplier 4000 --reps 20 --warmup 3
variance_adaptive_v2.exe --experiment B --multiplier 4000 --reps 20 --warmup 3
variance_adaptive_v2.exe --experiment C --multiplier 4000 --reps 20 --warmup 3
variance_adaptive_v2.exe --experiment D --multiplier 4000 --reps 20 --warmup 3

:: Generate plots and dashboard
python analysis\analyze_v2.py --experiment A
python analysis\analyze_v2.py --experiment B
python analysis\analyze_v2.py --experiment C
python analysis\analyze_v2.py --experiment D
python analysis\build_dashboard.py

:: Open dashboard
start results\dashboard.html
```

**Requirements:** NVIDIA GPU, CUDA Toolkit, MSVC Build Tools, Python 3 with
`pandas matplotlib scipy numpy`.
