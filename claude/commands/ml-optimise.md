---
description: Profile an ML target (flamegraph, allocations, syncs, GPU/PCIe saturation, idle gaps), then optimise the top-N bottlenecks under a parity gate, re-profiling between rounds.
argument-hint: [entry point, script, or stage to optimise]
---

Optimise: **$ARGUMENTS**

With no argument, pick the project's main training entry point and say which one you picked.

Measure, fix the top few, re-measure, stop when the top item stops mattering. Every change
is justified by a number from a profile and validated against a saved baseline.

Not this:

- rewriting the model, changing the algorithm, or dropping a term silently
- optimising anything that is not in the profile
- chasing items under the project's noise floor
- adding dependencies. If a tool is not installed, write the twenty lines instead

## Tracing can take the machine down, and worse, it can lie

Profiling adds time, memory and disk to a run that may already sit near its limits. Assume
the first attempt falls over, and size it so that does not cost anything when it does.

Measured on a 4090: full tracing cost **1.7x wallclock** on a launch-bound workload and
under 5 percent on a compute-bound one, at roughly **450 bytes per traced event**. A step
with a few thousand ops is megabytes of trace per step, so a hundred steps is
multi-gigabyte, and loading that back costs several times its size in host RAM. Do not
assume either extreme: measure the overhead on the first small run.

What actually kills a box:

- **Host RAM.** The event buffer lives in RAM until export, and parsing a trace back costs
  more than writing it did. The most common way a profile takes the machine down.
- **Disk.** A multi-gigabyte trace fills the partition mid-run and takes the training
  output with it. Check free space first, gzip, delete traces already read.
- **Device memory.** `profile_memory`, the memory-history buffer and a `TorchDispatchMode`
  all stack onto a run that may already be near capacity. A run that normally fits can OOM
  under tracing.
- **The probes themselves.** `set_sync_debug_mode("error")` aborts on the first sync by
  design, so it is a verification tool and never a discovery tool. `TorchDispatchMode`
  intercepts every op and can break exotic ops and `torch.compile`, and must record shapes
  and dtypes, never tensor references.

### The observer effect is the worse hazard

Instrumentation stacks multiplicatively, and a heavy enough stack does not just slow the
run, it **reorders the ranking**. You then go and optimise whatever the profiler was most
expensive to watch.

Real case: `with_stack` plus `profile_memory` plus `record_shapes` over a 150 second run,
plus `_record_memory_history`, plus a background thread fork/exec-ing `nvidia-smi` twice
every 100 ms. Three expensive things stacked on a fourth. One stage went from 16 s to 60 s
purely from being watched. The profile was measuring the profiler.

So add instrumentation one level at a time, and know what each level costs:

- **Level 0**, stage timing with syncs. Negligible.
- **Level 1**, `profile()` with no options. Low.
- **Level 2**, `+ record_shapes=True`. Low, the cheapest of the three flags.
- **Level 3**, `+ profile_memory=True`. Moderate.
- **Level 4**, `+ with_stack=True` with `verbose=True`. Highest by far, and only needed for
  the flamegraph, so it gets its own short run.

Levels 2, 3 and 4 never run together. Sampler threads are instrumentation too: count them
in the budget, and never attach one to the level-4 run.

**The check that catches all of this:** after any traced run, compare the total and the
per-stage shares against level-0 untraced timing. If total wallclock inflated past about
1.3x, or any stage's share moved materially, the ranking is instrumentation and not code.
Drop a level and re-measure before acting on it, and report the inflation factor next to
the ranking so the next reader can judge it too.

### Escalate scale separately, cheap fixes first

- **Rung 0.** 2 to 4 random samples, 3 to 5 steps, level 0, no tracing. Proves the harness
  runs and gives a first shape of the cost.
- **Rung 1.** Small batch, 10 steps, level 1 to 2, one track per run. First ranked list,
  and where the obvious mechanical fixes get landed.
- **Rung 2.** Production shapes, 10 to 20 steps, level 3 or 4, bounded schedule. The
  ranking that actually transfers.
- **Rung 3.** Production config, longer, level 0 plus the sampler, tracing off. End-to-end
  before and after.

Tiny-sample numbers are wrong in magnitude but usually right in order, and the mechanical
wins (a sync inside a loop, a materialised matrix, a missing `non_blocking`) show up at
rung 0 or 1 for almost nothing. Land those before spending a rung-2 run.

Rules that hold at every rung:

- **One track per run.** Stack capture, memory recording, dispatch interception and
  sampling each distort the others.
- **Bound what you trace.** `schedule(wait=5, warmup=3, active=3, repeat=1)` records a few
  steps no matter how long the run is. Cap `_record_memory_history(max_entries=100_000)`
  and disable it immediately after the dump.
- **Flush per track.** A crash on track 6 must not lose tracks 1 through 5. Long runs go in
  the background with a timeout so a hang stays recoverable.
- **Read traces with `key_averages()`, never `json.load`.** A trace big enough to want
  parsing in python is big enough to OOM you.
- **Never report a traced number as a performance number.** Wallclock, throughput and peak
  memory each get their own untraced run. Tracing says where the time goes, not how much
  there is.
- If the profile OOMs where the real run does not, that is profiler overhead and not a
  finding. Drop a rung, drop a track, say so.

## Step 0. Read the prior decisions first

Perf work re-opens settled arguments, so find out what was already decided. Read the
project's findings and journal docs (`KEY_FINDINGS.md`, `JOURNAL.md`, `TODO.md`,
`AGENTS.md`, `CLAUDE.md`, whatever this repo uses) and grep them for the target's module
names and for `precision`, `float32`, `chunk`, `stream`, `workers`.

Note:

- anything saying "tried and rejected" or "must stay exact". A rejected idea does not get
  re-proposed as an optimisation without saying it was rejected and why the reason no
  longer holds.
- the project's own noise floor: cross-validation standard error, seed variance. That
  number becomes the parity epsilon in Step 3.
- where the repo allows new files. Some repos forbid new top-level files or docs outright.
  If there is no `scripts/` or `bench/` and the rules forbid adding one, write the
  profiling script to the session scratchpad and tell the user where it went. Ask before
  creating a new top-level folder.

## Step 1. Write the profiling script

One script, re-runnable, no arguments needed for the default target, named
`profile_<target>.py`. Outputs go in a single flat directory, `runs/<timestamp>-profile/`
if the repo has `runs/`, otherwise beside the script:

- `flame_cpu.svg` and `flame_cuda.svg`
- `trace.json.gz`
- `ops.txt`, top 25 by self device time with call counts
- `summary.json`, every track's numbers in parseable form

Warm up 5 to 10 iterations before measuring, `torch.cuda.synchronize()` around every timed
region, production shapes and batch size, and the same autocast context the real code uses.
First-iteration numbers are allocator and compile warmup: never report them.

### The tracks

**1. Stage wallclock.** Per-stage times with syncs between, sorted by share of total. Wrap
stages in `record_function` so they appear in the trace and the flamegraph too.

**2. Flamegraph.** Two gotchas, each of which silently produces an empty flamegraph:

- `with_stack=True` alone is not enough. Without
  `experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)` the python
  frames are dropped and `export_stacks` writes a zero-byte file.
- `export_stacks` accepts only `self_cpu_time_total` and `self_cuda_time_total` as its
  metric, even on torch versions where `key_averages().table()` wants
  `self_device_time_total`. Export both.

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
) as prof:
    ...
prof.export_stacks(f"{out}/flame_cpu.folded", "self_cpu_time_total")
prof.export_stacks(f"{out}/flame_cuda.folded", "self_cuda_time_total")
```

Render folded stacks to SVG in-process, no external tooling. This renderer is tested,
including the escaping: frame names contain `<module>` and `<built-in method ...>`, which
produce invalid XML raw.

```python
import html

def flamegraph(folded, out, W=1600, R=16):
    root = {"v": 0.0, "c": {}}
    for line in open(folded):
        stack, _, val = line.rstrip().rpartition(" ")
        if not stack:
            continue
        node = root
        root["v"] += float(val)
        for f in stack.split(";"):
            node = node["c"].setdefault(f, {"v": 0.0, "c": {}})
            node["v"] += float(val)
    if not root["v"]:
        raise SystemExit(f"{folded} empty: profile needs with_stack and verbose=True")
    rects = []

    def walk(node, d, x):
        for name, ch in sorted(node["c"].items(), key=lambda kv: -kv[1]["v"]):
            w = W * ch["v"] / root["v"]
            if w >= 0.3:
                rects.append((x, d, w, name, ch["v"]))
                walk(ch, d + 1, x)
            x += w

    walk(root, 0, 0.0)
    H = (max(r[1] for r in rects) + 2) * R
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" style="background:#000">']
    for x, d, w, name, v in rects:
        n, y = html.escape(name), H - (d + 1) * R
        svg.append(
            f'<g><title>{n} ({v:.0f})</title>'
            f'<rect x="{x:.1f}" y="{y}" width="{max(w - 1, 0.3):.1f}" height="{R - 1}" '
            f'fill="hsl({20 + hash(name) % 40},70%,55%)"/>'
            f'<text x="{x + 2:.1f}" y="{y + R - 4}" font-size="10" fill="#fff" '
            f'font-family="monospace">{n if w > 6 * len(name) else ""}</text></g>'
        )
    open(out, "w").write("".join(svg) + "</svg>")
```

**3. Allocations.** `profile(..., profile_memory=True)` for per-op allocated bytes, plus
`max_memory_allocated()` and `max_memory_reserved()` per stage, plus a snapshot:

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
...
torch.cuda.memory._dump_snapshot(f"{out}/mem.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

Report allocated against reserved: a large gap is fragmentation, not demand. Report
allocations per step, since a step allocating thousands of blocks is churning.

**4. Device synchronisations.** `torch.cuda.set_sync_debug_mode("warn")` turns the implicit
ones into warnings. Every `.item()`, `.cpu()`, `.tolist()`, `nonzero()`, `bool(tensor)` and
tensor print in the hot path is a full pipeline stall. The warning text carries no call
site, so read it off the record:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    run_one_step()
syncs = [(w.filename, w.lineno) for w in caught if "synchroniz" in str(w.message)]
```

Count per step in `summary.json`, grouped by call site, worst first. The mode is a prototype
and misses some synchronising ops, so treat the count as a floor. Once you believe they are
gone, re-run with `"error"` to prove it.

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

Rank by bytes times count. Anything quadratic in batch or feature count is a candidate for
chunking, for a fused formulation, or for never being formed at all: `A @ (B @ x)` instead
of `(A @ B) @ x`, a Gram matrix accumulated in blocks instead of an `N x N` materialise.

**6. Windowing and chunking.** For every large intermediate found above, record the chunk
size actually in use and sweep two or three alternatives for time and peak memory. Report
the knee, not just the winner: a chunk size 3 percent faster at 4x the peak memory is not
the winner. Note where the knee sits relative to the current setting, since that is what
tells the user whether the parameter is worth exposing.

**7 and 8. GPU and PCIe saturation.** One track, not two, because one command streams both.
Spawn a **single long-lived process** and read its stdout from a thread:

```python
smi = subprocess.Popen(["nvidia-smi", "dmon", "-s", "ut", "-d", "1"],
                       stdout=subprocess.PIPE, text=True)
# columns: gpu, sm%, mem%, enc, dec, jpg, ofa, rxpci MB/s, txpci MB/s
```

**Never fork a sampler per sample.** One `nvidia-smi` fork/exec costs about 14.5 ms
measured, and forking a python process holding a large RSS and a CUDA context stalls the
parent on top of that. Two pollers at 100 ms is roughly 29 percent of a core spent watching
the run, and it will show up as your bottleneck. One process, one stream, one second apart.
Sub-second sampling buys nothing: these counters are window-averaged anyway. Do not use
`torch.cuda.utilization()` unless `pynvml` is importable, since it raises otherwise.

GPU saturation is two numbers, because they disagree in a useful way:

- *Kernel occupancy of wallclock*: device kernel time from the trace over measured
  wallclock. The honest "is the GPU doing anything" number.
- *Sampled `sm%`*: min, median and max from the stream above.

Low occupancy with high sampled `sm%` means many tiny kernels, so it is launch-bound. Low on
both means the GPU is starved, so look at track 9 and the dataloader.

PCIe saturation comes from `rxpci` and `txpci` in the same stream, cross-checked against
host-to-device and device-to-host memcpy bytes in the trace over elapsed time. Compare
against the link's real capability read *under load*, since
`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current` reports a
downtrained link when idle and a reading of gen 1 at rest means nothing. Gen4 x16 is about
25 GB/s practical against 31.5 GB/s theoretical. Sustained transfer above roughly 60 percent
of practical is a real ceiling, and the fix is transferring less (uint8 on the wire and cast
on device, cache features on device, kill round trips), not transferring faster.

**9. Blocked versus active periods.** Build the device-activity timeline from the trace and
invert it. Report:

- the fraction of wallclock with at least one kernel in flight, against the fraction with
  none
- the 10 longest idle gaps, each with its duration and the CPU-side op spanning it

This is the single most diagnostic track. A 40 percent blocked timeline with the gaps
landing on `next(iterator)` is a dataloader problem, not a model problem, and no amount of
kernel tuning will touch it.

Run the script. If a track cannot run on this machine (no CUDA, no `nvidia-smi`), degrade
that track and say so in the report. Never drop one silently.

## Step 2. Rank, and pick N adaptively

From `summary.json`, build one ranked list of candidates across all tracks, each with its
measured cost share. Then:

- **N** is the smallest number of items whose costs sum to 60 percent of total measured
  cost.
- Cap N at 5 per round, and drop any item under 5 percent of total.
- Order within the round by critical and cheap first. A high-cost mechanical fix (a sync in
  a loop, a materialised matrix, a missing `non_blocking`) outranks a higher-cost item that
  needs thought.
- Items needing a maths or algorithm change never enter a round. They go to Step 4.

After each round, re-profile and recompute N. The list reorders after every fix, which is
the entire point of re-profiling. **Stop** when the top remaining item is under 5 percent of
total, or a full round yields under 10 percent end-to-end, or only Step 4 items remain. Say
which stop condition fired.

## Step 3. Fix, in this order

Work the phases in order. Do not jump ahead because a later phase looks juicier: earlier
phases change the measurements the later ones depend on.

**Phase 1. Forward pass.** A few samples, random data is fine. It is the fastest loop to
iterate in and it isolates model cost from data cost. Target the top items: syncs,
materialised intermediates, launch-bound kernel storms, wrong operand order in chained
matmuls, work that is loop-invariant.

**Phase 2. Dataloader.** Stream wherever the data allows: load sample, use sample, drop
sample. Never hold the corpus in host or device memory if it can be iterated. Where the
loader supports it, use `num_workers` above zero, `persistent_workers=True`,
`pin_memory=True` paired with `non_blocking=True` on the transfer, and a prefetch depth.
Where the loader is a plain generator with no worker support, leave it sequential rather
than inventing a threading layer, and say that is what you did. Preprocess and cache
anything deterministic so it is computed once, not once per epoch. Verify against track 9:
the idle gaps should shrink.

**Phase 3. Visualisations and evaluation.** These belong outside the training loop.

- Ask the user to confirm moving metric and plot work to post-training, with metrics
  flushed to file at each computation so a watcher can render live. If the repo already
  mandates offline visualisation, do not ask, just enforce it and cite the rule.
- Move anything computable offline out of the loop: PCA, projections, confusion matrices,
  distribution plots, anything that reads only recorded state.
- What stays in the loop writes append-only records and flushes. No figure is rendered
  during training. No metric forces a sync it does not need: accumulate on device, reduce
  once at the interval boundary, transfer once.

**Phase 4. Precision, non-critical paths only.** Reduce precision where the profile says
precision is the cause, not merely where it is allowed. Candidates: metric accumulation,
visualisation feature dumps, distance and similarity matrices used for ranking,
nearest-neighbour search, anything whose output is a plot or a rank ordering. TF32 matmul is
usually the first free win on Ampere and later. The core model, the data pipeline semantics
and the training loop's numerics are Step 4, never here.

Experimenting across variants is expected and encouraged: compare precision levels, compare
operand orders and association in chained products, compare accumulate-in-fp32 against
accumulate-in-input-dtype, compare a low-rank or blocked form against the dense one. Report
the comparison table, including the losers, not just the winner.

### The parity gate, applied to every change

Before the first change of a round, freeze a baseline: fixed seed, fixed data slice, model
outputs and headline metrics saved to `parity_baseline.npz` in the run directory. Re-run and
compare after every change.

- **Pure mechanics** (caching, streaming, removing syncs, moving work out of the loop):
  bitwise identical, or exact-equal metrics.
- **Reassociation or reordering in fp32**: max relative difference under 1e-5, metric
  unchanged to reported precision.
- **Reduced precision on a non-critical path**: metric delta strictly inside the project's
  noise floor, with the max relative difference reported.

The epsilon comes from the noise floor found in Step 0. If the project has no such number,
measure one: run the baseline twice under different seeds and use that spread. An epsilon
you made up is not a parity gate.

A change that fails parity gets reverted the same turn. Record what it was and what it cost
in accuracy, so nobody retries it in three weeks.

## Step 4. When the fix needs the maths changed, ask

An item that is heavy but critical, where the only real win requires changing the algorithm,
the numerics of the core model, or the meaning of a term, stops and goes to the user via
AskUserQuestion. Do not decide this alone and do not quietly skip it.

Bring to that question:

- what it costs now, as a share of total, with the measurement
- what the change would be, concretely
- what it would buy, estimated from the profile
- what it would cost in exactness, and whether that sits inside the noise floor
- **whether this was already decided before**, quoted from the findings or journal, and
  what has changed since

Bold ideas are welcome here, as long as they are viable and named as bold: a streaming or
blocked reformulation, a low-rank factorisation of a dominant product, a closed form
replacing an iterative solve, a loop restructured so the large intermediate never forms.
Propose each with the number that motivates it. Carry on with the rest of the round while
the question is open, and never block the whole task on one answer.

## Step 5. Report and record

Report:

- before and after end-to-end wallclock and peak memory, at the same config
- the ranked bottleneck list per round, with what was done to each item
- the parity result for every landed change, with the epsilon used
- the comparison tables from any precision or ordering experiments, including the losers
- what was rejected, and why
- what remains, ranked, and which items wait on a user decision
- the observer-effect inflation factor for any ranking taken from a traced run

Append findings to the project's findings document and journal, following that repo's
conventions. Attach the final flamegraphs and `summary.json` so the next round starts from
measurements rather than memory. Commit the profiling script together with the
optimisations, in one commit, only when the user asks for a commit.
