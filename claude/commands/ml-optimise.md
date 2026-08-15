---
description: Profile an ML target (flamegraph, allocations, syncs, GPU/PCIe saturation, idle gaps), then optimise the top-N bottlenecks under a parity gate, re-profiling between rounds.
argument-hint: [entry point, script, or stage to optimise]
---

Optimise: **$ARGUMENTS**

With no argument, pick the project's main training entry point and say which one you picked.

Measure first, then fix. Nothing gets "optimised" here on a hunch: every change is
justified by a number from a profile and validated against a saved baseline. The
loop is profile, fix the top few, re-profile, stop when the top item stops mattering.

## What this does and does not do

Does:

- write one re-runnable profiling script that emits a flamegraph and a machine-readable summary
- rank bottlenecks by measured cost and fix the top N, where N adapts per round
- prove parity against a frozen baseline after every change
- stop and ask when a fix would change the maths, not just the mechanics

Does not:

- rewrite the model, change the algorithm, or drop a term silently
- optimise anything that is not in the profile
- chase items under the noise floor
- add dependencies. If a tool is not installed, write the twenty lines instead

## Tracing is not free, and it can take the machine down

Profiling adds memory, disk and time to a run that may already be near its limits. A
profiler that OOMs the box has produced no numbers and cost the user their session. Assume
the first run will fall over and size it so that it does not matter when it does.

Measured on a 4090, launch-bound workload (many small ops, the worst case for tracing):
full tracing cost **1.7x wallclock** and about **450 bytes per traced event**. A training
step with a few thousand ops is therefore a couple of megabytes of trace per step, so a
hundred steps is a multi-gigabyte trace, and `json.load` on that is several times its size
again in host RAM. On a compute-bound workload the time overhead was under 5 percent, so
do not assume either extreme: measure it on the first small run.

### The observer effect is the real hazard, not just the crash

Instrumentation stacks multiplicatively, and a heavy enough stack does not just slow the
run, it **reorders the ranking**. You then optimise whatever the profiler was most
expensive to watch.

A real instance from this project: `with_stack=True` plus `profile_memory=True` plus
`record_shapes=True` over a 150 second run, plus `_record_memory_history(max_entries=100_000)`,
plus a background thread fork/exec-ing `nvidia-smi` twice every 100 ms. Three expensive
things stacked on a fourth. One stage went from 16 s to 60 s purely from being watched.
The profile was measuring the profiler.

So instrument in levels, adding one thing at a time:

| Level | Adds | Cost |
|---|---|---|
| 0 | stage timing with syncs | negligible |
| 1 | `profile()` with no options | low |
| 2 | `+ record_shapes=True` | low, cheapest of the three flags |
| 3 | `+ profile_memory=True` | moderate |
| 4 | `+ with_stack=True, verbose=True` | highest, and only needed for the flamegraph |

Never run levels 2, 3 and 4 together. `with_stack` gets its own run, at the shortest
schedule that produces a readable flamegraph. Telemetry sampling is a separate axis:
add it to at most one level at a time, never to the `with_stack` run.

**The check that catches this:** after any traced run, compare the total and the per-stage
shares against the level-0 untraced timing. If total wallclock inflated past about 1.3x, or
any stage's share moved materially, the ranking is instrumentation and not code. Drop a
level and re-measure before acting on it. Report the inflation factor alongside the ranking
so the next reader can judge it too.

Failure modes to size against:

- **Host RAM.** The event buffer lives in RAM until export, and parsing a trace back costs
  more than writing it. This is the most common way a profile kills a box.
- **Disk.** Multi-gigabyte traces fill a partition mid-run and take the training output
  with them. Check free space before, write gzipped, delete traces from rounds you have
  already read.
- **Device memory.** `profile_memory`, the memory-history ring buffer, and a
  `TorchDispatchMode` all add overhead on top of a run that may already sit near capacity.
  A run that fits normally can OOM under tracing.
- **The probes themselves.** `set_sync_debug_mode("error")` aborts the run on the first
  sync by design, so it is a verification tool, never a discovery tool.
  `TorchDispatchMode` intercepts every op and can break exotic ops and `torch.compile`.
  Never hold references to intercepted tensors: record shapes and dtypes, not tensors.

### Escalate up the ladder, do not start at scale

| Rung | Config | Tracks | Purpose |
|---|---|---|---|
| 0 | 2 to 4 random samples, 3 to 5 steps | instrumentation level 0, no tracing | prove the harness runs, get a first shape of the cost |
| 1 | small batch, 10 steps | one track per run, level 1 to 2 | first ranked list, land the obvious mechanical fixes |
| 2 | production shapes, 10 to 20 steps | one track per run, bounded schedule, level 3 or 4 | numbers that transfer, the real ranking |
| 3 | production config, longer | level 0 plus telemetry, tracing off | end-to-end before-and-after |

Land the cheap fixes from rung 0 and 1 before spending a rung-2 run. Early numbers from a
tiny sample are wrong in magnitude but usually right in order, and the mechanical wins
(a sync in a loop, a materialised matrix, a missing `non_blocking`) show up there for
almost no cost.

Rules that hold at every rung:

- **One track per run.** Never enable all tracks at once. Stack capture, memory recording,
  dispatch interception and telemetry sampling each distort the others.
- **Sampling threads are instrumentation too.** Count them in the level budget, and never
  spawn a subprocess per sample.
- **Bound what you trace.** Use the profiler schedule so only a few steps are recorded no
  matter how long the run is: `schedule(wait=5, warmup=3, active=3, repeat=1)`. Cap
  `_record_memory_history(max_entries=100_000)` and disable it immediately after the dump.
- **Flush per rung.** Write `summary.json` and the flamegraphs as soon as each track ends,
  never only at the end. A crash on track 6 must not lose tracks 1 through 5.
- **Long runs go in the background with a timeout** so a hang is recoverable and partial
  output survives.
- **Read traces with `key_averages()`, not by loading the JSON.** If a trace is large
  enough that you want to parse it in python, it is large enough to OOM you.
- **Never report a number measured under tracing as a performance number.** Wallclock,
  throughput and peak memory all get their own untraced run. Tracing tells you where the
  time goes, not how much there is.
- If the profile OOMs where the real run does not, that is profiler overhead, not a
  finding. Drop a rung, drop a track, and say so.

## Step 0. Read the prior decisions before touching anything

Perf work re-opens settled arguments. Find out what was already decided.

- Read the project's findings and journal docs (`KEY_FINDINGS.md`, `JOURNAL.md`, `TODO.md`,
  `AGENTS.md`, `CLAUDE.md`, or whatever this repo uses). Grep them for the target's
  module names, for `precision`, `float32`, `chunk`, `stream`, `workers`, `alpha`.
- Note anything that says "this was tried and rejected" or "this must stay exact".
  A rejected idea does not get re-proposed as an optimisation without saying it was
  rejected before and why the reason no longer holds.
- Note the project's own noise floor if it has one (cross-validation standard error,
  seed variance). That number becomes the parity epsilon in Step 3.
- Also note where the repo allows new files. Some repos forbid new top-level files or
  docs outright. Respect that: if there is no `scripts/` or `bench/` directory and
  the rules forbid adding one, write the profiling script to the session scratchpad
  and tell the user where it went. Ask before creating a new top-level folder.

## Step 1. Write the profiling script

One script, re-runnable, no arguments needed for the default target. Name it
`profile_<target>.py`. Its outputs go in a single flat run directory
(`runs/<timestamp>-profile/` if the repo has `runs/`, otherwise beside the script).

The script profiles a small number of steps of the target, and emits:

| Output | File |
|---|---|
| Flamegraph (CPU and device) | `flame_cpu.svg`, `flame_cuda.svg` |
| Chrome trace | `trace.json.gz` |
| Op table (top 25 by self device time, with call counts) | `ops.txt` |
| Summary of every track below, as parseable JSON | `summary.json` |

Warmup 5 to 10 iterations before measuring, `torch.cuda.synchronize()` around every
timed region, production shapes and batch size, and the same autocast context the
real code uses. First-iteration numbers are allocator and compile warmup, never report them.

### Tracks the script must cover

**1. Stage wallclock.** Per-stage times with syncs between, sorted by share of total.
Wrap stages in `record_function` so they appear in the trace and the flamegraph too.

**2. Flamegraph.** Two gotchas, both of which produce a silently empty flamegraph if missed:

- `with_stack=True` alone is not enough. Without
  `experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)` the python
  frames are dropped and `export_stacks` writes a zero-byte file.
- `export_stacks` accepts only `self_cpu_time_total` and `self_cuda_time_total` as its
  metric, even on torch versions where `key_averages().table()` wants
  `self_device_time_total`. Export both metrics.

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    profile_memory=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
) as prof:
    ...
prof.export_stacks(f"{out}/flame_cpu.folded", "self_cpu_time_total")
prof.export_stacks(f"{out}/flame_cuda.folded", "self_cuda_time_total")
```

Render folded stacks to SVG in-process, no external tooling. This renderer is tested,
including the escaping (frame names contain `<module>` and `<built-in method ...>`, which
produce invalid XML if passed through raw):

```python
import html

def flamegraph(folded, out, width=1600, row=16):
    root = {"v": 0.0, "c": {}}
    for line in open(folded):
        stack, _, val = line.rstrip().rpartition(" ")
        if not stack:
            continue
        node, v = root, float(val)
        root["v"] += v
        for frame in stack.split(";"):
            node = node["c"].setdefault(frame, {"v": 0.0, "c": {}})
            node["v"] += v
    if not root["v"]:
        raise SystemExit(f"{folded} is empty: profile needs with_stack and verbose=True")
    rects, depth_max = [], [0]

    def walk(node, depth, x):
        depth_max[0] = max(depth_max[0], depth)
        for name, ch in sorted(node["c"].items(), key=lambda kv: -kv[1]["v"]):
            w = width * ch["v"] / root["v"]
            if w >= 0.3:
                rects.append((x, depth, w, name, ch["v"]))
                walk(ch, depth + 1, x)
            x += w

    walk(root, 0, 0.0)
    h = (depth_max[0] + 2) * row
    body = []
    for x, d, w, name, v in rects:
        y = h - (d + 1) * row
        fill = f"hsl({20 + hash(name) % 40},{60 + (hash(name) >> 8) % 30}%,55%)"
        safe = html.escape(name)
        label = safe if w > 6 * len(name) else ""
        body.append(
            f'<g><title>{safe} ({v:.0f})</title>'
            f'<rect x="{x:.1f}" y="{y}" width="{max(w - 1, 0.3):.1f}" height="{row - 1}" fill="{fill}"/>'
            f'<text x="{x + 2:.1f}" y="{y + row - 4}" font-size="10" font-family="monospace">{label}</text></g>'
        )
    open(out, "w").write(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" '
        f'style="background:#000"><style>text{{fill:#fff}}</style>{"".join(body)}</svg>'
    )
```

**3. Allocations.** `profile(..., profile_memory=True)` for per-op allocated bytes, plus
`torch.cuda.max_memory_allocated()` / `max_memory_reserved()` per stage, and a snapshot:

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
...
torch.cuda.memory._dump_snapshot(f"{out}/mem.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

Report allocated versus reserved. A large gap is fragmentation, not demand. Report
allocations per step: a step that allocates thousands of blocks is churning.

**4. Device synchronisations.** Turn the implicit ones into warnings and count them:

```python
torch.cuda.set_sync_debug_mode("warn")   # "error" once you believe you removed them all
```

Every `.item()`, `.cpu()`, `.tolist()`, `nonzero()`, `bool(tensor)` and print of a tensor
in the hot path is a full pipeline stall. The warning text itself carries no call site, so
capture it and read the frame off the warning record:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    run_one_step()
syncs = [(w.filename, w.lineno) for w in caught if "synchroniz" in str(w.message)]
```

Count them per step in `summary.json`, grouped by call site, worst first. Note that the mode
is a prototype and does not catch every synchronising op, so treat the count as a floor.
Once you believe they are gone, re-run with `"error"` to prove it.

**5. Materialisation of large matrices.** Log any op whose output exceeds a threshold:

```python
from torch.utils._python_dispatch import TorchDispatchMode

class BigTensors(TorchDispatchMode):
    def __init__(self, mb=64): self.mb, self.hits = mb, {}
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in (out if isinstance(out, (list, tuple)) else [out]):
            if torch.is_tensor(t) and t.numel() * t.element_size() > self.mb << 20:
                k = (str(func), tuple(t.shape), str(t.dtype))
                self.hits[k] = self.hits.get(k, 0) + 1
        return out
```

Rank by bytes times count. Anything quadratic in batch or feature count is a candidate
for chunking, for a fused formulation, or for never being formed at all (`A @ (B @ x)`
instead of `(A @ B) @ x`, Gram matrix accumulated in blocks instead of an `N x N` materialise).

**6. Windowing and chunking.** For every large intermediate found above, record the
chunk size actually used and sweep two or three alternatives for time and peak memory.
Report the knee, not just the winner. A chunk size that is 3 percent faster and 4x the
peak memory is not the winner.

**7 and 8. GPU and PCIe saturation, from one sampler.** These are one track, not two,
because one command streams both. Spawn **a single long-lived process** and read its stdout
from a thread:

```python
smi = subprocess.Popen(["nvidia-smi", "dmon", "-s", "ut", "-d", "1"],
                       stdout=subprocess.PIPE, text=True)
# columns: gpu, sm%, mem%, enc, dec, jpg, ofa, rxpci MB/s, txpci MB/s
```

**Never fork a sampler per sample.** One `nvidia-smi` fork/exec costs about 14.5 ms
measured, and forking a python process holding a large RSS and a CUDA context stalls the
parent on top of that. Two pollers at 100 ms is roughly 29 percent of a core spent watching
the run, and it will show up as your bottleneck. One process, one stream, one second
between samples. Sub-second sampling buys nothing here: these counters are averaged over a
sampling window anyway.

Do not use `torch.cuda.utilization()` unless `pynvml` is importable; it raises otherwise.

GPU saturation is two numbers, because they disagree in a useful way:
- *Kernel occupancy of wallclock*: sum of device kernel time from the trace divided by
  measured wallclock. This is the honest "is the GPU doing anything" number.
- *Sampled `sm%`*: min, median and max from the stream above.

Low kernel occupancy with high sampled utilisation means many tiny kernels: launch-bound.
Low on both means the GPU is starved: look at track 9 and the dataloader.

PCIe saturation comes from `rxpci` and `txpci` in the same stream, cross-checked against
the host-to-device and device-to-host memcpy bytes in the trace divided by elapsed time.
Compare against the link's actual capability, read *under load*
(`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`
reports a downtrained link when idle, so a reading of gen 1 at rest means nothing).
Gen4 x16 is about 25 GB/s practical against 31.5 GB/s theoretical.
Sustained transfer above roughly 60 percent of practical is a real ceiling: the fix is
transferring less (send uint8 and cast on device, cache features on device, avoid
round trips), not transferring faster.

**9. Blocked versus active periods.** Build a device-activity timeline from the trace,
then invert it. Report:
- fraction of wallclock with at least one kernel in flight (active) versus none (blocked)
- the 10 longest idle gaps, each with its duration and the CPU-side op spanning it

This is the single most diagnostic track. A 40 percent blocked timeline with the gaps
landing on `next(iterator)` is a dataloader problem, not a model problem, and no kernel
tuning will touch it.

Run the script. If a track fails on this machine (no CUDA, no `nvidia-smi`), degrade that
track and say so in the report. Do not silently drop it.

## Step 2. Rank, and pick N adaptively

From `summary.json`, build one ranked list of candidate items across all tracks, each with
its measured cost share. Then:

- **N** is the smallest number of items whose costs sum to 60 percent of total measured cost.
- Clamp N to at most 5 per round, and drop any item below 5 percent of total.
- Order within the round: critical and cheap first. An item that is high cost and a
  mechanical fix (a sync in a loop, a materialised matrix, missing `non_blocking`)
  outranks an item that is higher cost but needs thought.
- Items needing a maths or algorithm change do not enter the round. They go to Step 4.

After each round, re-profile and recompute N. The list reorders after every fix, which is
the entire point of re-profiling. **Stop** when the top remaining item is under 5 percent of
total, or a full round yields under 10 percent end-to-end improvement, or only Step 4 items
remain. Say which stop condition fired.

## Step 3. Fix, in this order

Work the phases in order. Do not jump ahead because a later phase looks juicier: earlier
phases change the measurements that later phases depend on.

**Phase 1. Forward pass.** A few samples, random data is fine. Fastest loop to iterate in
and it isolates model cost from data cost. Target the top items: syncs, materialised
intermediates, launch-bound kernel storms, wrong operand order in chained matmuls, repeated
work that is loop-invariant.

**Phase 2. Dataloader.** Stream wherever the data allows: load sample, use sample, drop
sample. Never hold the corpus in memory or on device if it can be iterated. Where the
loader supports it: `num_workers` above zero, `persistent_workers=True`, `pin_memory=True`
paired with `non_blocking=True` on the transfer, and a prefetch depth. Where the loader is
a plain generator with no worker support, leave it sequential rather than inventing a
threading layer, and say that is what you did. Preprocess and cache anything deterministic
so it is computed once, not once per epoch. Verify with track 9: idle gaps should shrink.

**Phase 3. Visualisations and evaluation.** These belong outside the training loop.
- Ask the user to confirm moving metric and plot work to post-training, with metrics
  flushed to file at each computation so a watcher can render live. If the repo already
  mandates offline visualisation, do not ask, just enforce it and cite the rule.
- Move anything computable offline out of the loop: PCA, projections, confusion matrices,
  distribution plots, anything reading only recorded state.
- What stays in the loop writes append-only records and flushes. No figure is rendered
  during training. No metric forces a sync it does not need: accumulate on device, reduce
  once at the interval boundary, transfer once.

**Phase 4. Precision, non-critical paths only.** Reducing precision is allowed here when
the profile says precision is the cause, not merely when it is available. Candidates:
metric accumulation, visualisation feature dumps, distance and similarity matrices used
for ranking, nearest-neighbour search, anything whose output is a plot or a rank ordering.
TF32 for matmul is usually the first free win on Ampere and later. Never touch the core
model, data pipeline semantics, or the training loop's numerics here; those are Step 4.

Experimenting across variants is expected and encouraged: compare precision levels, compare
operand orders and association in chained products, compare accumulate-in-fp32 against
accumulate-in-input-dtype, compare a low-rank or blocked form against the dense one.
Report the comparison table, not just the winner.

### The parity gate, applied to every change

Before the first change of a round, freeze a baseline: fixed seed, fixed data slice, save
model outputs and the headline metrics to `parity_baseline.npz` in the run directory.
After each change, re-run and compare.

| Change class | Required parity |
|---|---|
| Pure mechanics (caching, streaming, removing syncs, moving work out of the loop) | bitwise identical, or exact-equal metrics |
| Reassociation or reordering in fp32 | max relative difference under 1e-5, metric unchanged to reported precision |
| Reduced precision on a non-critical path | metric delta strictly inside the project's own noise floor, and report the max relative difference |

The epsilon comes from the project's measured noise floor found in Step 0 (cross-validation
standard error, seed variance). If the project has no such number, measure one: run the
baseline twice with different seeds and use that spread. An epsilon you made up is not a
parity gate.

A change that fails parity gets reverted the same turn. Record what it was and what it cost
in accuracy so nobody retries it in three weeks.

## Step 4. When the fix needs the maths changed, ask

An item that is heavy but critical, where the only real win requires changing the algorithm,
the numerics of the core model, or the meaning of a term, stops and goes to the user via
AskUserQuestion. Do not decide this alone and do not quietly skip it.

Bring to that question:

- what it costs now, as a share of total, with the measurement
- what the change would be, concretely
- what it would buy, estimated from the profile
- what it would cost in exactness, and whether that is inside the noise floor
- **whether this was already decided before**, quoted from the findings or journal, and
  what has changed since

Bold ideas are welcome here, as long as they are viable and named as bold: a blocked or
streaming reformulation, a low-rank factorisation of a dominant product, a closed-form
substitution for an iterative solve, restructuring a loop so a large intermediate is never
formed. Propose them with the number that motivates them. Continue with the rest of the
round while the question is open; never block the whole task on one answer.

## Step 5. Report and record

Report:

- before and after end-to-end wallclock and peak memory, at the same config
- the ranked bottleneck table per round, with what was done to each
- the parity result for every landed change, with the epsilon used
- the comparison tables from any precision or ordering experiments, including the losers
- what was rejected, and why
- what remains, ranked, and which items are waiting on a user decision

Record: append findings to the project's findings document and the journal, following that
repo's conventions. Attach the flamegraphs and `summary.json` from the final round so the
next round starts from measurements, not memory.

Commit the profiling script along with the optimisations, in one commit, only when the
user asks for a commit.
