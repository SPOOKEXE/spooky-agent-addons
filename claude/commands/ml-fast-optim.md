---
description: Fast read-only sweep of a forward pass for device syncs, needless materialisation, per-call allocation, dtype churn and recompile triggers. Grep catalogue first, then line-by-line read, a cumulative per-function table, ranked findings, and only the certain fixes applied under a parity gate. No profiler.
argument-hint: [module, model class, file, or blank for the project's forward pass]
---

Sweep: **$ARGUMENTS**

With no argument, sweep the forward pass of the project's main model, and say which entry
point you picked and how you found it.

This is a **reading** task. No profiler, no traces, no long runs. Minutes, not hours. The
output is a table of every function checked plus a ranked list of findings, and only the
fixes that are certain get applied.

Not this:

- changing the algorithm, dropping a term, or rewriting a module
- proposing `torch.compile`, AMP, or a fused kernel as the fix for a code bug. Fix the bug
  first, then the compiler has something worth compiling
- touching backward, data loading, checkpointing or logging unless the sweep is aimed there
- reporting a pattern as a finding without saying how many times per step it runs
- padding the table with functions you grepped but did not read

## The handoff to /ml-optimise

`/ml-fast-optim` finds bugs that are wrong on inspection: a sync you can point at, a matrix
that did not need to exist, a constant rebuilt every call. Those need no profile, only an
identity argument and a parity check.

`/ml-optimise` settles anything about **magnitude**: which of two clean implementations is
faster, whether a kernel is launch-bound or memory-bound, whether a change was worth it in
wallclock. If the answer requires a number you do not already have, it is not a finding
here. It goes in the report as a hypothesis for `/ml-optimise`, with the exact measurement
that would settle it.

The tell: if the sentence needs the word "probably", it belongs to `/ml-optimise`.

## Step 0. Scope the forward pass and fix the rows

Start at the top-level `forward` and follow the call graph. Use the project's code index if
one exists, otherwise read the calls.

Produce the row list **before** reading anything in depth: every function reachable from
forward, in call order, with file and line range. That list is the table's rows and it does
not grow later. Anything not reachable from forward is out of scope: say so once, name it,
and skip it.

While here, note from config the concrete values of the dimensions that appear in shapes
(batch, sequence, feature, bank size, number of heads, chunk size). A materialisation
finding is worthless without the N it scales in, and the ranking in Step 3 needs the loop
trip counts.

Read the project's own notes (`KEY_FINDINGS.md`, `JOURNAL.md`, `TODO.md`, `AGENTS.md`,
`CLAUDE.md`) for anything marked "must stay exact", "tried and rejected", or a stated
numerical tolerance. A rejected idea does not come back as a finding without saying it was
rejected and why the reason no longer holds. A stated tolerance becomes the parity epsilon
in Step 5.

## Step 1. The grep pass

One mechanical sweep over the in-scope files. This does not check anything, it only points.
Every hit is a candidate that Step 2 confirms or discards by reading.

Run these as one batch, scoped to the files from Step 0.

### Class S, host-device sync

A sync drains the queued pipeline. The cost is not the call, it is everything the GPU had
not finished yet, and it is paid every time the line runs.

```
\.item\(\)|\.tolist\(\)|\.numpy\(\)|\.cpu\(\)|\.to\(['"]cpu
\b(float|int|bool|len)\(\s*\w+[\.\[]
if\s+.*\.(any|all|sum|max|min|item|numel)\(\)
assert\s+.*\.(any|all|item|allclose)
torch\.(nonzero|masked_select|unique|bincount|allclose|equal)
\[\s*\w*mask\w*\s*\]|\[\s*\w+\s*>\s*|\[\s*\w+\s*==\s*
torch\.cuda\.synchronize|set_sync_debug_mode
print\(|\.format\(|f['"].*\{.*tensor
torch\.linalg\.(cholesky|lstsq|eigh|svd|solve|matrix_rank)
while\s+
```

Notes on the non-obvious ones:

- `torch.nonzero`, `masked_select`, `unique`, `bincount` and boolean-mask indexing all have
  **data-dependent output shape**, so the host must learn the size. That is a sync even
  though nothing looks like `.item()`.
- `torch.allclose` and `torch.equal` return a python bool. Both sync. Fine in a test, never
  in forward.
- `while` with a tensor condition syncs once per iteration.
- `torch.linalg` factorisations sync when errors are checked. Prefer the `ex` variants
  (`cholesky_ex`, `linalg.inv_ex`) and keep the returned `info` on device rather than
  branching on it.
- an f-string or `print` of a tensor formats it, which means it copies it.

### Class M, needless materialisation

```
torch\.cat|torch\.stack
\.repeat\(|\.expand\(|\.tile\(
\.contiguous\(\)
torch\.(eye|zeros|ones|full|arange|linspace|empty)
\.T\s*@|\.transpose\(.*\)\s*@|@\s*\w+\.T
torch\.einsum|torch\.matmul|@
\[\s*:\s*,\s*None\s*\]|\[None|unsqueeze\(|\.view\(|\.reshape\(
one_hot|scatter_|\.float\(\)|\.double\(\)|\.to\(torch\.float
softmax|logsumexp|cdist
```

What to look for at each hit:

- **`cat` or `stack` inside a loop.** Quadratic copying. Collect into a list, concatenate
  once. If the loop is over a fixed small axis, preallocate and write slices instead.
- **`repeat` where `expand` works.** `repeat` copies, `expand` is a stride-0 view. `expand`
  is only wrong when the result is written to, or fed to an op that requires contiguity.
- **`contiguous()` that is already true, or that the next op did not need.** Matmul and
  most reductions handle strided input. A `contiguous()` on a large tensor is a full copy.
- **The explicit Gram.** `A.T @ A @ v` builds an `n x n` matrix to multiply a vector.
  `A.T @ (A @ v)` does not. Same for `(A @ B) @ C` when the middle product is the large one:
  reassociate to the cheap order and say so, because reassociation moves the last bits.
- **Broadcast-then-reduce.** `(a[:, None] - b[None]).pow(2).sum(-1)` materialises
  `n x m x d` to produce `n x m`. The expansion `|a|^2 + |b|^2 - 2ab` is one matmul. Note
  that this one trades exactness for speed: it loses precision on near-identical vectors,
  so it is a finding, not an automatic fix.
- **One-hot then matmul** instead of `index_select` or `embedding`.
- **Upcasting a large tensor** to fp32 or fp64 when only the reduction needed the width.
  Reduce in the wide dtype (`.sum(dtype=torch.float32)`) instead of casting the input.
- **A full score matrix that only feeds a softmax and a weighted sum.** That is the chunked
  attention pattern, and if the shapes are standard, SDPA already does it.

### Class A, per-call allocation and host work

Cheap individually, but they are launch-bound death by a thousand cuts, and they are the
easiest class to fix with zero numerical risk.

```
torch\.(eye|arange|linspace|tensor|zeros|ones)\(  # inside forward
\.to\(.*device|\.cuda\(\)|device=
for\s+\w+\s+in\s+range\(.*(batch|B|N|seq|len\()
import numpy|np\.
\.detach\(\)\.clone\(\)|\.clone\(\)\.detach\(\)
```

- A constant built inside `forward` should be a `register_buffer` (moves with the module,
  saves with the state dict) or a cached module attribute. `torch.arange(n, device=...)`
  per call is the classic.
- `torch.tensor(python_scalar)` inside forward is a host allocation plus an H2D copy, and
  it blocks on some paths. Use a buffer, or let the scalar stay a python float and let
  broadcasting handle it.
- `.to(device)` inside forward means something was on the wrong device to begin with. Find
  out what, and move it once at construction.
- A python loop over the batch is almost always a batched op that was not found. Say what
  the batched form is.
- numpy in the hot path forces a sync and a round trip. It is Class S as well as Class A.

### Class R, recompile and graph breaks

Only run this if the project actually uses `torch.compile` or CUDA graphs. If it does not,
say so and skip the class rather than reporting empty.

```
torch\.compile|cudagraph|mark_dynamic
\.item\(\).*shape|shape\[.*\.item|\.size\(\)\[
if\s+\w+\.(shape|size|numel)
```

Anything that turns a device value into a python value that then determines a shape is a
guard, so it recompiles on every new value. A `print` inside a compiled region is a graph
break. Variable sequence length without `mark_dynamic` recompiles per length.

### Class D, dtype and layout

```
autocast|amp\.|half\(\)|bfloat16|float16
channels_last|memory_format|\.permute\(|\.transpose\(
\.sum\(|\.mean\(|\.var\(|\.std\(|cumsum
```

Look for repeated crossings of an autocast boundary (cast down, cast up, cast down again in
the same function), a reduction accumulating in the narrow dtype where the project's notes
demand fp32, and a permute feeding a matmul in a layout that forces a copy.

## Step 2. The read pass and the rolling table

**grep does not check a function.** A function is checked when it has been read line by
line, including the lines grep never matched. Most of the good findings, an op ordered
wrongly, a mask recomputed every call, a reduction over the wrong axis, have no
distinctive token to grep for.

Read in call order. After each module, reprint the whole table, cumulative, never dropping
rows:

| # | module.function | lines | S | M | A | R | D | verdict |
|---|-----------------|-------|---|---|---|---|---|---------|
| 1 | hcr.bank.forward | 41-88 | 1 | 2 | . | . | . | bug |
| 2 | hcr.norm.apply   | 12-30 | . | . | . | . | . | clean |

- One row per function from Step 0. The count in each class column is confirmed hits after
  reading, `.` for none.
- `verdict`: `clean`, `suspect` (real but needs a number, so it goes to `/ml-optimise`), or
  `bug` (certain, and it gets a numbered finding below the table).
- Rows stay in call order so the reader can see where in the pass the cost sits.
- A function with zero findings is reported as zero. Do not invent something to fill a row.

Under the table, the findings so far, numbered, each one line: `file:line`, class,
per-step count, the fix. Detail goes in the final report, not in every reprint.

## Step 3. Rank without a profile

Rank by **per-step count times cost class**, and state both. Never rank by how bad the line
looks.

Cost classes, order of magnitude only, and say that is what they are:

- **Sync**: costs whatever the GPU had queued, typically tens of milliseconds if it lands
  mid-step and near zero if it lands where the host was going to block anyway. A sync at
  the end of the step, right before the optimiser or the loss print, is close to free. The
  same call at the top of a per-layer loop is paid once per layer.
- **Materialisation**: bytes, and they are computable. Write out `N x M x 4 bytes` with the
  real N and M from Step 0. A hidden `n x n` where n is the bank or sequence dim is the
  finding that usually matters most, and it also caps batch size, which is a second cost.
- **Per-call allocation**: microseconds each, so it only ranks when the count is high. It
  is worth fixing anyway because the risk is zero.
- **Recompile**: seconds, but once per distinct shape. Count the distinct shapes, not the
  calls.

Multiply by the trip count of every enclosing loop. A sync in a loop of length L is L
syncs, and that single multiplication reorders the list more often than anything else.

## Step 4. Fix only the certain ones

Certain means all three: the change is local, you can state the identity that makes it
behaviour-preserving in one line, and it does not change the shape of any public output.

Safe to apply:

- `.item()` used only for a log or a counter: keep the value on device and accumulate
  there, or reduce the log to every N steps
- `cat`/`stack` inside a loop: collect and combine once
- constant rebuilt per call: `register_buffer` or a cached attribute
- `repeat` to `expand` where the result is only read
- `A.T @ A @ v` to `A.T @ (A @ v)`, and any reassociation where the cheap order is obvious.
  State that the last bits move
- a device transfer hoisted out of forward to construction
- `.contiguous()` deleted where the consumer accepts strided input
- an `ex` variant swapped in for a factorisation whose `info` was being branched on

Everything else is a finding with a proposed patch, unapplied, including anything that
trades exactness for speed, anything needing a shape or algorithm change, and anything
where the win is a guess. Say plainly which ones were left and why.

Apply one class at a time and keep the model running after each. If a change cannot be made
without touching three other functions, it is not a fast fix.

## Step 5. Verify

**Parity.** Fixed seed, identical input, forward before and after. Exact bitwise equality
is required for every change that did not reassociate arithmetic, and there is no excuse
for a tolerance on a `register_buffer` hoist. For reassociations, state the tolerance and
where it came from, either the project's own noise floor or a computed bound. Do not paste
a default `allclose` and call it parity.

**Sync count.** To confirm a sync is actually gone, wrap one forward in
`torch.cuda.set_sync_debug_mode("warn")` and count the warnings before and after. Use
`"error"` only as the final gate on a single narrowed call, because it aborts on the first
sync by design. Never leave either mode on.

**Timing, as a sanity check only.** Same input, 5 warmup iterations, 20 timed, one
`torch.cuda.synchronize()` around the whole timed block and not per iteration. If the time
did not move, say it did not move. A correct fix with no measured gain is still correct,
and pretending otherwise is how a fast sweep starts producing fiction. Anything that needs
a real number goes to `/ml-optimise`.

## Report

1. The final cumulative table, every function, nothing dropped.
2. Findings, ranked, each with: `file:line`, class, per-step count, the arithmetic behind
   the ranking, the one-line fix, and applied or not.
3. What was applied, and the parity result with its actual epsilon.
4. Hypotheses for `/ml-optimise`, each with the exact measurement that would settle it.
5. Coverage: functions read, functions skipped and why.

Count, do not use adjectives. "3 syncs per step in a loop of 12 layers, so 36" is a
finding. "several unnecessary syncs" is not.

## Known false positives, do not report these

- `.item()` or `.cpu()` once per epoch, in eval, or in a metrics path already off the hot
  loop
- a sync at the end of a step where the host was about to block anyway
- `nonzero`, `unique` or a mask built once in `__init__` or setup
- `repeat` where the result is written to, or fed to an op requiring contiguity
- `contiguous()` immediately before a `view` that genuinely needs it
- `expand` already in use, flagged only because the regex matched
- a python loop over a fixed axis of size 2 or 3
- a materialised intermediate that is genuinely reused several times downstream
- fp32 accumulation the project's notes explicitly require

If a candidate is on this list, drop it silently. It does not need a line in the report
explaining that it was considered and rejected.
