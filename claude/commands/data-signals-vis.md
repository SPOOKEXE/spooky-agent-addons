---
description: Build a signal and visualisation suite for a training run. Ask the user which categories they want, record everything during training asynchronously and flush per record, then render every figure offline from the emitted files once the run ends or dies. Covers dataset signals, optimisation and gradient health, representation and output signals, and deriving diagnostics for training methods that have no backward pass.
argument-hint: [model, script, or entry point to instrument, and optionally which signals you care about]
---

Instrument and visualise: **$ARGUMENTS**

With no argument, find the project's main training entry point or its demo scripts, list what you
found, and say which one you picked.

Two rules hold over everything below, and both are load-bearing.

**Record during, render after.** The training loop writes append-only records and nothing else.
No figure is rendered, no metric is reduced across steps, no image is written while training is
running. Every plot in this suite is produced by a separate offline script that reads the emitted
files and never imports the model.

**A signal that costs more than about 1% of step time gets switched off during a crunch, and
then it is not a signal.** Budget accordingly, and measure the budget rather than assuming it.

Not this:

- plotting inside the training loop, or importing matplotlib into the training process
- `.item()` on a tensor every step to log it
- a dashboard that renders one number as a gauge. A gauge is a table with worse density
- recording only means. The mean is where problems hide
- inventing signals for a method whose maths you have not read
- 200 separate scalar series presented as a dashboard

## Step 0. Ask the user what they actually want

Do this **before** writing any code, and do it with AskUserQuestion. The suite is worth building
only if it answers questions someone has. Present categories, not individual metrics, and make
clear that everything picked is recorded during training and rendered afterwards.

Offer these as multi-select, with the must-have set already identified as the default:

- **Must-have, always built**: loss against tokens with a reference-run delta, pre-clip grad norm
  with z-score and clip fraction, update-to-weight RMS ratio per tensor, throughput and step
  time, and the run manifest.
- **Optimisation and stability**: LR schedule against realised update size, optimiser second
  moment distribution, loss spikes with the batch indices that caused them, weight-norm growth.
- **Gradients**: per-layer grad norm family, per-group ratios, gradient conflict between data
  domains, gradient noise scale.
- **Representations**: per-layer activation RMS, dead and saturated units, activation histograms,
  attention entropy where applicable, residual-stream norm growth with depth.
- **Outputs and calibration**: per-position entropy, top-k accuracy, confidence and reliability,
  logit magnitude, per-domain held-out BPB.
- **Data**: sequence-length and padding waste, per-domain tokens consumed against configured,
  duplicate and near-duplicate batch rate, loss by domain and by position, contamination check.
- **Efficiency**: MFU and HFU with the recompute tax between them, roofline placement, per-rank
  step-time skew. For a deeper pass, hand this to `/ml-optimise` instead.
- **Method-specific**: RL panels, MoE routing, LoRA adapter scale, or the derived signals in
  Step 6 for anything without a backward pass.

Ask three things beyond the category list, because they change the design rather than the
content:

1. **Is there a known-good reference run** to diff against? Absolute loss means nothing; the
   delta against a reference at equal token count means a great deal. If there is none, say so
   in the report and treat the first run as the future reference.
2. **What decision is each chosen plot supposed to drive**, and at what magnitude of deviation
   would they stop a run? Decide that in advance. Dashboards otherwise encourage per-step
   reaction to noise.
3. **What is the step-time budget** for instrumentation, and is this run already tight on memory?

If the user does not answer, build the must-have set only, say that is what happened, and list
what was left out so it can be added without re-instrumenting.

## Step 1. The recording layer

Measured per record, one JSON line, local NVMe:

| operation | cost per record | verdict |
| --- | --- | --- |
| `json.dumps` + write, buffered | 1.35 us | |
| **+ `flush()`** | **1.79 us** | the default. Flush costs 0.44 us |
| + `os.fsync` | **4845 us** | 2700x. Never per record |
| `queue.put_nowait` onto a writer thread | 0.56 us | saves 1.2 us, costs a thread |
| `logging` with QueueHandler | 5.15 us | slower than writing the file yourself |
| sqlite WAL, `synchronous=NORMAL` | 5.44 us p50 | when concurrent readers matter |
| sqlite WAL, `synchronous=FULL` | 5374 us | 1000x. Always set NORMAL |

At 10 records per second, flushing every record costs 0.0004% of wall time. **The worry about
flush-per-record is unfounded, so do it.** `fsync` is the one to avoid: 4.8 ms per record is 4.8%
of wall time at the same cadence, and all it buys is power-loss durability, which is not the
failure being defended against.

So the default recorder has no thread and no queue:

```python
import json, os, atexit

f = open(run_dir / "metrics.jsonl", "a", buffering=1)     # line buffered
def log(**rec):
    f.write(json.dumps(rec) + "\n")
    f.flush()                                             # 0.44 us over buffered
atexit.register(lambda: (f.flush(), os.fsync(f.fileno()), f.close()))
```

This is non-blocking in the only sense that matters: 1.79 us against a training step measured in
milliseconds, and the record is on disk before the next line of training code runs. It therefore
cannot lose anything to a crash.

**Add the writer thread only in these two cases:**

- the cadence is above roughly 10k records per second, where 1.2 us each starts to add up
- the run directory is on a network or shared filesystem, where a single write can block for
  milliseconds. **This is the real argument for async: tail latency, not mean cost.**

Then use a bounded `queue.Queue` with `put_nowait` (0.56 us) and a daemon thread that writes and
flushes. State the trade-off in the report: **records still sitting in the queue are lost if the
process is SIGKILLed**, which the direct version cannot suffer. Count the drops when the queue
fills, because dropping records is correct and blocking the training step is not.

```python
import json, os, queue, threading, atexit, sys

class Recorder:
    """Only for the two cases above. The training thread only ever does put_nowait;
    a daemon thread writes and flushes, so completed records survive a SIGKILL."""
    def __init__(self, path):
        self.q = queue.Queue(maxsize=100_000)
        self.f = open(path, "a", buffering=1)
        self.stop = threading.Event()
        self.dropped = 0
        self.t = threading.Thread(target=self._drain, daemon=True)
        self.t.start()
        atexit.register(self.close)

    def log(self, **rec):
        try: self.q.put_nowait(rec)
        except queue.Full: self.dropped += 1

    def _drain(self):
        while not (self.stop.is_set() and self.q.empty()):
            try: rec = self.q.get(timeout=0.1)
            except queue.Empty: continue
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()

    def close(self):
        if self.stop.is_set(): return
        self.stop.set(); self.t.join(timeout=5.0)
        self.f.flush(); os.fsync(self.f.fileno()); self.f.close()
        if self.dropped: print(f"WARNING dropped {self.dropped} records", file=sys.stderr)
```

**Why flush per record, whichever version is used.** Verified across four endings, 200 planned
records with the process ending at step 99:

| ending | records on disk | malformed lines |
| --- | --- | --- |
| clean exit | 200 | 0 |
| exception | 100 | 0 |
| SIGINT | 100 | 0 |
| SIGKILL | 100 | 0 |

SIGKILL runs no handler at all, so the atexit path never fires. Every completed record still
survives, because flush had already handed the line to the OS, and line buffering means a partial
line is never left behind. That is the property that makes "render after it stops" work when a
run dies unattended at 3am.

**Format.** JSONL unless concurrent readers are needed, then sqlite in WAL mode with
`synchronous=NORMAL`. Verified unusable for this job: **parquet** (reading while the writer is
open raises `ArrowInvalid` because the footer is unwritten, and one row group per record costs
425 bytes a row) and **zarr** (669 us per record, 370x JSONL, because each resize rewrites
metadata). Arrow IPC streaming works but costs 208 bytes a row at one batch per record. h5py SWMR
works with `libver="latest"` and `maxshape=(None,)`, at 28.8 us.

On the read side, `polars.read_ndjson` does 100k records in 0.009 s against pandas at 0.078 and
stdlib json at 0.093. If write cadence ever justifies it, `orjson.dumps` is 7.5x faster than
`json.dumps` (0.16 us against 1.21).

Rules for the recorder:

- **One stream per cadence, not one file.** `metrics.jsonl` per step, `tensors.jsonl` every 50 to
  100 steps, `eval.jsonl` per eval, `data.jsonl` per batch sample, `manifest.json` once. Mixing
  cadences into one file makes every reader filter.
- **Rank 0 writes** unless per-rank series were requested. Per-rank step-time skew is one of the
  few things that genuinely needs all ranks.
- **Every record carries `step`, `tokens_seen`, and wall-clock.** Plot against **tokens**, not
  steps, because steps are not comparable across batch sizes.
- **The manifest is not optional**: config hash, code commit, effective batch size, seed, dtype,
  data-loader position, and the reference-run id. Without it two runs cannot be compared and the
  whole suite is decorative.
- Restart-safe: append, never truncate. Evals on a **fixed step cadence, not wall-clock**, so a
  restart does not shift the sampling grid.

### Reading a tensor costs whatever the GPU still owes

- **Every scalar read syncs.** `.item()`, `float(t)`, `.cpu()`, `.tolist()` and
  `cuda.synchronize()` cost 4.8 to 7.8 us on an idle stream but **340 to 460 us** behind a single
  queued 2048-cubed matmul, because they block on the outstanding work. `.detach()` does not sync
  (0.3 us idle, 9.3 us behind that matmul).
- **Batch the scalars.** 50 separate `.item()` calls cost 232 us; `torch.stack(scalars).cpu()`
  once costs 20 us, **11.6x** cheaper. The win starts at roughly 20 scalars, so below that do not
  bother.
- **Grad norms via `torch._foreach_norm`.** 50 `p.norm().item()` calls cost 2198 us;
  `torch.linalg.vector_norm(torch.stack(torch._foreach_norm(params)))` costs 330 us, **6.7x**
  cheaper. This is the single highest-value fix in a real training loop.
- **Histogram on device, transfer the counts.** `torch.histc(w, bins=64, min=-4, max=4)` on 4M
  elements costs 4.87 us. Moving that 16 MB tensor to host first costs 2883 us, and doing it in
  numpy costs 15703 us: **77x worse for identical numbers.**
- `torch.cuda.Event(enable_timing=True)` is the only sync-free way to time a step.

## Step 2. What to record, with the thresholds that make it readable

Log **distributions, not just means**: p50, p99, max, and a histogram where it is cheap. The mean
is where problems hide.

**Gradient norm** (pre-clip, always; post-clip alone logs the threshold, not the gradient):

```
g_t = ||grad||_2 pre-clip
mu_t  = b*mu + (1-b)*g_t                    b ~ 0.97 to 0.99
var_t = b*var + (1-b)*(g_t - mu_t)^2
z_t   = (g_t - mu_t) / sqrt(var_t + eps)
```
Healthy `z_t` inside +/-2 with occasional 3. Concerning sustained above 3, any above 10. Clip
rate healthy under 5%, concerning above 20% or rising. Per-group norms within one order of
magnitude of each other. Spikes of 1000x typical are documented (2501.06842).

**Update-to-weight RMS**, the scale-free one to trust when grad norm is not comparable:

```
ratio_W = rms(lr * applied_update) / (rms(W) + eps)      rms(x) = sqrt(mean(x^2))
```
Use the **applied** update, after weight decay, clipping and optimiser scaling, **per matrix**,
every 50 to 100 steps, computed on the accumulated update rather than per-microbatch. Median
around **1e-3**, spread inside one order. Climbing at constant LR means weight decay is shrinking
norms toward weight-norm criticality and spikes (2607.21005). A jump at restart means optimiser
state is not round-tripping. Log median, min, max and **the identity of the argmin and argmax
matrices**: two hundred scalar series is not a dashboard.

**Held-out loss as bits per byte**, the only cross-tokenizer-comparable unit:

```
BPB = (L_nats / ln 2) * (N_tokens / N_bytes)
```
Anchor 0.6 to 0.9 for English web text, lower for code, higher for multilingual. Healthy is
monotone decreasing and roughly linear against log compute, with stable domain rank ordering.
Train and held-out matching to many decimals is a leak, not a triumph. Freeze and hash held-out
sets before the run, decontaminate first, fix the token budget per domain, and log byte counts
next to BPB.

**Policy entropy** for RL: `H_t = -sum_v pi(v|s_t) log pi(v|s_t)`, reported as `H/H_0`. Healthy
0.5 to 0.9 and falling slowly; watch below 0.4; collapsed below 0.2 and flat with shrinking grad
norms; diverging above `H_0`.

**Train/inference log-prob mismatch**, for anything where rollouts and training use different
kernels: `delta_t = log pi_train - log pi_rollout`. Healthy `mean|delta_t|` under 1e-3 and
`max|delta_t|` under 0.1; concerning at 1e-2 and approaching 1.0. Log the argmax-flip rate too.
The divergence bound scales with `(1-p)`, so error concentrates in the tail and the mean hides it
(2512.23087).

**MoE**: expert-load Gini, roughly 0.035 balanced against 0.70 collapsed (2506.21328). Load
falling **late** in training is healthy relaxation, not breakage (2604.04230). Routing collapse
is a bifurcation and therefore abrupt, so it wants an alert rather than a chart (2605.29121).

**Efficiency**: MFU, HFU, and the gap between them as the recompute tax. Per-rank step time.

**Enablers**, cheap and worth more than most metrics: batch indices at every spike step so the
offending documents can be pulled later, and loss on the same batch immediately before checkpoint
save and immediately after restore, which must match to numerical noise. That single assertion is
the highest-value test in a training codebase.

## Step 3. Dataset signals

Recorded from the loader, during the run, at whatever cadence is affordable:

- **Sequence-length histogram and padding waste.** Length-distribution shift is a named cause of
  throughput drops that looks like a hardware problem.
- **Per-domain tokens consumed against configured.** These differ more often than anyone expects.
- **Duplicate and near-duplicate batch rate.** A loader bug detector and a leading indicator of
  memorisation. Log the data-loader position and a sample hash: replayed data is a quiet source
  of memorisation.
- **Loss by domain and by token position.** Position-wise loss catches packing and mask bugs that
  the aggregate curve descends straight through.
- **Per-domain gradient conflict**, cosine similarity between domain gradients. Negative means the
  mixture is fighting itself. Expensive, so sample it.
- Decontaminate by exact and near-duplicate match **before** the run, not after.
- Offline-only dataset work, needing no training run at all: token-frequency and entropy
  distributions, document length, vocabulary coverage, near-duplicate clusters, label noise, and
  a mixture table over format and topic.

## Step 4. Render offline, and make each figure earn its place

A separate script, `render_report.py`, that takes the run directory, imports no model code, and
rebuilds every figure from the JSONL files. It must be rerunnable against a run that died.

Axes and shapes that matter:

- **Loss against tokens, log x.** Healthy is a smooth power-law descent. A step change is data, a
  spike is numerics, a plateau is the LR or the optimiser. **Always plot the delta against the
  reference run at equal token count.** Absolute loss means nothing.
- **Grad norm with clip fraction on the same panel.** Both rising together is the classic
  pre-divergence signature.
- **Update ratio per tensor** as a family of lines in one band. Two orders below the bundle means
  that tensor is not learning; two above means it is about to blow up. As a heatmap over (layer,
  step) for depth trends.
- **Per-layer activation RMS and grad norm** read as a family: look for the one line leaving the
  bundle, not the absolute values.
- **Heatmaps that earn their place**: token loss by position, per-domain BPB delta across
  checkpoints, per-position entropy over (step, position), optimiser second-moment histogram over
  steps, per-rank step-time skew.
- **The RL four-panel**: reward, entropy, length and KL together. Any three can look fine during
  reward hacking; all four do not. Watch for monotone entropy collapse, cyclical eruption in
  agent RL (2605.27954), and length inflation into truncation collapse into repetition
  (2604.08527).
- **Roofline**: arithmetic intensity against achieved FLOP/s, both log. Read the **horizontal**
  distance to the roof, not the vertical one.
- **Multi-checkpoint, offline by nature**: filter-normalised loss-landscape slices (1712.09913),
  PCA of the weight trajectory, mode-connectivity interpolation, eval curves **with noise bands**,
  isoFLOP and scaling-law log-log fits.

Do not build: saliency or grad-times-input maps, max-activating-neuron dashboards without an SAE,
weight histograms of a finished model, or anything that renders one number as a gauge.

### The rendering stack, chosen by measurement

- **matplotlib is the default.** `plt.style.use("dark_background")` already gives a `#000`
  facecolor. 1k points to PNG is 0.04 s and 29 KB.
- **Never emit a scatter as SVG.** 50k points is a **4.4 MB** file. PNG data-URI for anything
  scattered, SVG only for line plots, where the same figure is 33 KB.
- **Over roughly 100k points, use datashader.** 10M points render in 0.05 s and 50M in 0.13 s, to
  a PNG of about 160 KB regardless of count. First call pays a 0.6 s numba JIT, and it rejects
  3-digit hex, so write `#000000` not `#000`.
- **Do not ship the interactive libraries' default chrome.** A plotly `to_html` with
  `include_plotlyjs=True` is **4756 KB per file**, bokeh inline is 1256 KB, altair is 913 KB, and
  all three ship spinner animations in their loading chrome, which breaks the no-repainting rule
  outright. If plotly is genuinely wanted, use `include_plotlyjs=False, full_html=False` at 18 KB
  a chart and inline the library once.
- **Assemble one hand-written HTML shell** with embedded PNG data-URIs and no JavaScript. Zero
  repainting, true `#000` trivially, and it opens from a file path with no server.

**Visual style**, non-negotiable: true black `#000` background, white primary text,
information-dense, no decorative card or pill chrome, no light-grey subtitle lines above
sections, minimal copy. No continuously repainting animation of any kind: it pegs the GPU on a
high-refresh display and this is a report, not a screensaver.

### Trackers and diagnostic libraries

If the project already uses a tracker, read from it rather than replacing it. None of them
replace the JSONL, because the report has to be rebuildable by anything, without their reader.

- **dvclive** is the lightest, writing plain TSV and JSON at 1102 bytes per 50 points.
- **tensorboard** or **tensorboardX** write append-only protobuf that is tail-safe and re-readable
  via `EventAccumulator`. tensorboardX avoids the torch dependency.
- **wandb** offline works but spawns a `wandb-core` sidecar and needs a local unix socket, and its
  `.wandb` file is binary that only wandb reads.
- **mlflow's filesystem backend is now rejected outright** as being in maintenance mode; it needs
  `sqlite:///mlflow.db` to work offline. **aim** resolves only on Python 3.11 and should be
  avoided.

For diagnostics, prefer these over reimplementing:

- **weightwatcher** (maintained, 2026) for per-layer heavy-tail alpha from the weight spectrum:
  `ww.WeightWatcher(model=m).analyze()`. It pulls in tensorflow, so weigh that.
- **pyhessian** still runs but has been unmaintained since 2021. Vendor its ~300 lines rather than
  taking the dependency.
- **torchlens** (very active) for layer-by-layer activation capture.
- **openTSNE** fits 28x faster than umap-learn (0.3 s against 8.4 s including JIT), and plain PCA
  is faster still. Remember that no distance, gap or cluster size on any of these means anything.
- **netcal** `ECE(15).measure(p, y)` or `sklearn.calibration_curve` for reliability.

## Step 5. Signals that lie, and how the suite avoids being one of them

Build these caveats into the report text itself, not just your own reasoning:

- **A loss curve is a liveness check, not a correctness check.** A broken attention mask or a
  truncated context still descends smoothly.
- A smooth curve can hide an already-destabilised run. Mechanism monitors fire thousands of steps
  before loss or grad norm move (2606.28116).
- **KL estimators lie.** `K1 = -log r` and `K3 = (r-1) - log r` sit near baseline through the
  first 700 steps of a documented degradation, with reward falling 0.87 to 0.40 by step 650 and
  collapse near 1665.
- **Partial float absorption produces no visible spike** while inflating parameter norms
  (2605.06152). Watch LM-head logit magnitude, not the loss curve.
- Raw grad norm is scale-dependent and not comparable across widths or loss normalisations. The
  update ratio is.
- Per-sequence gradient statistics carry a length bias needing an explicit `sqrt(T)` correction
  (2605.09920).
- Attention maps do not show attribution: real and hallucinated objects attract equally strong
  attention (2608.07302). UMAP and t-SNE distances, gaps and cluster sizes are hyperparameter
  artefacts, so never measure anything off them.
- MFU is not utilisation. A profiled step is not an average step. The roofline is a ceiling, not
  a prediction, and the datasheet peak is usually the sparse lowest-precision number.
- Smoothing windows, axis limits and colour scales change what a curve appears to say. State the
  smoothing on the figure.

## Step 6. Methods with no backward pass

Everything above assumes a gradient, an optimiser state and a backward pass. When the method has
none of those, do not port the backprop dashboard across and do not invent signals. Derive them
from the method's own mathematics, in this order.

1. **Write the update as `dtheta = -eta * g`, and ask what `g` estimates.** Three cases, three
   different first plots:
   - it estimates a **true gradient** (ES, forward gradient, MeZO): plot **estimator quality**,
     variance and cosine against a finite-difference reference on a small proxy model, and how
     both scale with width.
   - it estimates one **approximately** (feedback alignment, equilibrium propagation, predictive
     coding): plot **the approximation error itself**, per-layer angle or RelMSE against the exact
     gradient, on a model small enough that backprop is affordable as a reference.
   - there is **no global objective** (Forward-Forward, Hebbian, greedy layerwise): skip gradient
     comparison entirely, go to 2.
2. **Name the quantity each unit is actually minimising, and log it per layer.** Layer-local loss,
   free energy, goodness margin, energy difference. The depth profile is the diagnostic; the
   global average hides the failure.
3. **Instrument the inner loop.** Anything with a relaxation, settling or equilibrium phase has an
   inner iteration and a step budget. Log the residual at the budget and the trajectory.
   Under-convergence biases every outer update and is invisible from outside.
4. **Sweep the hyperparameter the method's own theory says trades bias against variance**, and
   plot the target metric against it on a log axis: `beta` in equilibrium propagation, `epsilon`
   in MeZO, `T` in predictive coding, `sigma` in ES, surrogate `beta` in spiking nets, `theta` in
   Forward-Forward. **Look for a flat basin. No basin means the method is not in its valid
   regime**, and no amount of training fixes that.
5. **Check what is conserved or bounded.** Oja and SoftHebb weight norms converge to 1, CMA-ES
   `||p_sigma||` should track `E||N(0,I)||`, Forward-Forward goodness should sit either side of
   `theta`. Drift in a supposedly conserved quantity is the earliest warning available.
6. **Count the units that can silently stop learning, every epoch.** Dead spiking neurons,
   never-winning competitive units, layers whose local loss has flatlined, coordinates with a
   collapsed CMA-ES std. Zero-gradient states are usually absorbing: they never recover.
7. **Ask what would still look fine if this were broken.** Almost always one of: a biased but
   still-descending update, a collapsed representation that still separates, or a deep suffix
   learning nothing while a shallow prefix carries the task. All three are invisible in the task
   loss and visible in a per-layer profile.

**Make every primary metric a per-layer array rather than a scalar, and put depth on the x-axis
at least once.** That one habit catches most of the pathologies below.

| method | primary signal | the plot | what the task loss hides |
| --- | --- | --- | --- |
| Forward-Forward (2212.13345) | `G_l = sum_j y_j^2`, margin `mean(G_pos) - mean(G_neg)`, measured before layer-norm | goodness density per layer, two histograms, vertical line at `theta` | goodness collapse: every layer separates by uniformly inflating or shrinking activity, so the margin looks fine while the representation carries no class information |
| Predictive coding (2006.04182, 2407.01163) | `F = 0.5*sum_l norm(e_l)^2`, residual `max_abs(dF/da)` at t=T | `F` against inference step t, log y, one curve per sampled iteration | under-relaxation: still falling at t=T means every weight update came from a non-fixed-point state and is a biased estimate |
| Equilibrium propagation (1602.05179) | `F = E + beta*C`, RelMSE between `d^EP` and `-grad^BPTT` per layer | the GDU plot: EP update overlaid on the BPTT gradient across second-phase time | a bad `beta`: too large biases the gradient, too small amplifies settling noise by `1/beta`. Both look like "training, just worse" |
| Feedback alignment, DFA (1411.0247, 1609.01596) | angle between `d_FA` and `d_BP` per layer, degrees | angle against iterations, one line per layer, reference at 90 degrees | pinned at 90 degrees means no alignment ever formed and only the last layer is learning. Note alignment 0.92 still gave 79.9 against backprop's 87.2 (2006.12878), so plot alignment and accuracy together |
| Greedy layerwise (1812.11446, 2101.10832) | per-block linear-probe accuracy, `I(h,x)` by block | probe error against block index, greedy against end-to-end | early blocks greedily discard what later blocks need. Every local loss falls happily; only the depth profile shows the plateau |
| Hebbian, Oja, BCM, SoftHebb (1806.10181, 2209.11883) | per-unit `norm(w_i)` trajectory, win count per unit | win count against unit index sorted, plus the receptive-field grid | often there is no task loss at all. Bimodal weights at the bounds mean a binary, information-poor code while probe accuracy decays slowly |
| ES and CMA-ES (1604.00772, 1703.03864) | `sigma`, axis ratio `sqrt(cond(C))`, population fitness spread, `norm(p_sigma)` | pycma's panel: best f, `sigma` and axis ratio on one log plot against evaluations | premature convergence: fitness looks converged while `sigma` has collapsed somewhere that is not the optimum. Log the stopping criteria as booleans every generation, not just at the stop |
| Spiking, surrogate gradients (1901.09948) | per-layer firing rate, zero-spike fraction, membrane mass inside `abs(U - U_thr) < 1/beta` | spikes per neuron per trial, log count; and accuracy against mean rate | the silent-neuron ratchet: dead neurons get exactly zero surrogate gradient and can never recover, while loss falls on the surviving subnetwork |
| Forward gradient (2202.08587) | `cos(g_hat, grad)` and `Var[g_hat]` against width | cosine against parameter count, both log | the estimator is unbiased so loss still descends, just at a rate that collapses as `1/sqrt(d)`. No loss curve distinguishes that from "needs more steps" |
| MeZO, zeroth order (2305.17333) | the scalar projected gradient, per step; Hessian effective rank `tr(H)/spectral_norm(H)` | projected-gradient magnitude against step, log y, one line per `epsilon` | drifting to zero means `epsilon` is below numerical resolution and cancellation has eaten the signal; blow-up means it exceeds the local-linearity radius |

Two things to say out loud in the report when instrumenting any of these:

- Surrogate **shape** is robust in spiking networks and surrogate **scale** is not, so normalise
  the surrogate peak to 1 and sweep learning rate against `beta` as a grid.
- The honest baseline: a 2026 comparison of backprop, checkpointed backprop, forward-mode AD and
  zeroth-order (2506.21833) found checkpointed backprop ahead by up to 31.1% accuracy at
  comparable memory. If the goal is memory rather than biological plausibility or a hardware
  constraint, that is the number the alternative has to beat, and the suite should measure it.

## Step 7. Deliverables

- `record.py` or the patch that adds the recorder, with the measured step-time overhead stated as
  a percentage, measured rather than assumed.
- The run directory: one JSONL per cadence, plus `manifest.json`.
- `render_report.py`, rerunnable offline against a finished or a dead run, producing a
  self-contained HTML report with embedded figures.
- A short table in the report: each figure, the question it answers, the threshold that would
  change a decision, and the signal's cost in step-time percent.
- What was recorded but not plotted, what was requested but not built, and the drop count if it
  is not zero.
