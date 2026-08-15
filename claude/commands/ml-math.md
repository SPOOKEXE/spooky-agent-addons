---
description: The field layer for machine-learning mathematics on top of /smart-math. Conventions that must be decided before deriving, the counting formulas with their known-wrong variants, roofline and arithmetic intensity, precision limits, information-theory units, muP, RL objectives, LoRA scaling, and how to check every closed form against the running model.
argument-hint: [the ML maths: a derivation, a loss, a budget, a scaling estimate, an RL objective]
---

Work through: **$ARGUMENTS**

`/smart-math` is the engine and all of it applies here: parse once and hold handles, verify by
tier, fuzz the domain and step outside it, 3 seeds by default and 5 when load-bearing, name the
transform, and report proved / disproved+point / unknown+budget. This file is only the ML layer.
It is a checklist, not an essay.

**Get the ground truth before deriving, in this order.**

1. **The tools this session actually has.** Check what is connected rather than assuming: paper
   search, docs lookup, code search, any MCP server. One call often settles what an hour of
   derivation would only guess at.
2. **Web research**, for anything external: the paper, the spec, the release notes. An arXiv ID
   or a real URL for every external claim, never an invented one, and a status that is honest
   about what was verified against what was recalled.
3. **The repository's own documents**: README, `docs/`, `CLAUDE.md`, `AGENTS.md`, findings and
   journal files, prior commits touching the same maths.
4. **The code.** Failing all of the above, derive it from what is actually running. A model
   object is a better authority on its own parameter count than any formula, and reading the
   forward pass settles arguments about the forward pass.

## Decide these before the first line of algebra

Almost every ML factor error is one of these, chosen implicitly and then changed halfway.
Write the choice at the top of the file, in words, once.

| ambiguity | the two readings | how it bites |
| --- | --- | --- |
| `B` | sequences, or tokens (`B*s`) | every per-token quantity off by `s` |
| loss units | nats (`ln`, the ML default) or bits (`log2`) | factor `ln 2 = 0.693` |
| `H(p, q)` | which argument is reality | KL direction silently flips |
| KL direction | `D(pi_theta \|\| pi_ref)` mode-seeking, or the reverse | different objective, same symbol |
| per-token vs per-byte | loss and PPL are tokenizer artifacts | not comparable across vocabularies |
| FLOPs | model FLOPs, or hardware FLOPs (recompute included) | MFU vs HFU, ratio `1 + s/(6h)` |
| FLOPs, again | forward only, or forward+backward | factor 3 |
| params | with or without embeddings | the Kaplan/Chinchilla exponent gap, 2406.12907 |
| grad norm | pre-clip or post-clip | post-clip logs the threshold, not the gradient |
| update | raw gradient, or applied update after decay, clip, optimiser scaling | 10x errors |
| ratio | per-token or per-sequence | the unit mismatch GSPO exists to fix |
| `n_kv_heads` | equals `n_heads` only for MHA | KV size wrong by the GQA factor |
| seq len | current, or allocated maximum | KV budget wrong by the ratio |

Layout convention, for anything with a matrix derivative: **numerator or denominator, declared
in writing before deriving.** Layout confusion produces results that are transposes of correct,
which look correct. It ruins more derivations than every other cause combined.

## Tier 1 here is shapes, and it is nearly free

- Name every axis once: `b s h d v a` and what each means. `einops.rearrange` over `.view()` in
  anything being derived, because it fails loudly with the actual numbers. `einops.einsum`,
  `pack`/`unpack` and `parse_shape` cover the rest.
- **`@jaxtyped(typechecker=beartype)` on the function, not `@beartype` alone.** Bare `@beartype`
  does not bind the symbolic dimensions, so `Float[Tensor, "b h n d"]` on two arguments with
  incompatible `d` passes silently. The `@jaxtyped` wrapper is what makes the dims mean
  anything. Avoid `from __future__ import annotations` in those modules.
- `jax.eval_shape(f, *args)` does the shape and dtype algebra for zero FLOPs, which is the
  cheapest possible check on a large model. `chex.assert_shape` if you prefer assertions to
  annotations.
- **torch named tensors are gone** as of 2.13: no `.names`, no `rename`/`refine_names`/`align_to`,
  and `torch.zeros(2, 3, names=(...))` raises. Do not write named-tensor code.
- `opt_einsum.contract_path` gives FLOPs and peak intermediate before running anything:
  `info.opt_cost`, `info.largest_intermediate`. For `bqhd,bkhd->bhqk` at B=8 S=1024 H=16 D=64
  that is 1.718e10 FLOPs and a 1.342e8-element score matrix.
- A two-operand contraction has no reordering to win, so optimised equals naive. The only lever
  left is not forming the intermediate at all. Report FLOPs and peak memory together, since
  fewer FLOPs at 4x peak memory is not a win.
- Integer constraints go to z3 and are answered exactly: `d_model % n_heads`, `n_kv_heads`
  dividing `n_heads`, sequence length against block size, vocab against tensor-parallel degree.

## Gradients

- A closed-form gradient is a **claim** until something independent disagrees or fails to.
- Custom `autograd.Function` backward: `torch.autograd.gradcheck(MyFn.apply, (x64,))`, plus
  `gradgradcheck` if the second derivative is used. **float64 is mandatory**, verified: the same
  correct function in float32 raises `Jacobian mismatch` and warns that the input is not double.
- Closed form against autograd: `torch.func.grad` / `jacrev` / `jacfwd` / `hessian`, compared
  with `torch.testing.assert_close`. Per-sample gradients via
  `vmap(grad(f), in_dims=(None, 0, 0))`, which must sum to the full-batch gradient.
- JAX `custom_vjp`: `jax.test_util.check_grads(f, (x,), order=2, modes=("fwd", "rev"))`. Set
  `jax.config.update("jax_enable_x64", True)` **before creating any array**, or it fails on
  correct code.
- Numpy-only: `scipy.optimize.check_grad(f, g, x0)`. **It returns a scalar norm and never
  raises**, so it does nothing unless you threshold the result yourself.
- No analytic reference at all: `numdifftools.Gradient`, roughly 1e6 times tighter than
  `scipy.approx_fprime` (8e-14 against 3.3e-8) via Richardson extrapolation. Slow, so use it for
  a one-off audit rather than in a loop.
- Check the awkward points, not the easy ones: zero, the non-differentiable kink, near-duplicate
  rows, saturated softmax, a masked-out position, the excluded point of the assumption.

**Sympy matrix calculus is partial, and it fails silently.** It gets `trace(X.T*A*X).diff(X)` =
`A*X + A.T*X` and `((X*a-b).T*(X*a-b)).diff(X)` = `2*(X*a-b)*a.T` right, but:

- `log(det(X)).diff(X)` returns **`0`**. The answer is `X^-T`.
- `exp(X).diff(X)` returns **`0`**. `det(X).diff(X)` stays an unevaluated `Derivative`.
- **Never trust a sympy MatrixSymbol derivative that came back `0` or unevaluated.** Treat both
  as "the CAS declined", not as a result.
- When it stalls, build an explicit `Matrix(n, m, lambda i, j: Symbol(...))` and differentiate
  entrywise at n=3. That proves the identity for the concrete case, which beats a wrong symbol.
- Sympy is **numerator layout**: `y.jacobian(x)` is `(m, n)` = `dy_i/dx_j`, and so is
  `torch.func.jacrev`. Denominator layout is the transpose. Fix the convention once and assert
  the shape, since the two differ by exactly the mistake that is hardest to see.

## The counting formulas

`l` layers, `h` d_model, `s` seq len, `V` vocab, `a` heads, `p` bytes per element.

```
params  ~ 12*l*h^2                      only if d_ff = 4h, MHA, embeddings EXCLUDED
        exact: l*(12h^2 + 13h) + V*h*(1 + untied) + n_ctx*h + 2h
        GQA attention block is 2h^2 + 2*h*n_kv*d_head, not 4h^2
        SwiGLU has THREE h x d_ff matrices, so d_ff = 8h/3 preserves 12*l*h^2

model FLOPs    = 72*B*s*l*h^2 * (1 + s/(6h) + V/(16*l*h))     fwd+bwd, no recompute
hardware FLOPs = 96*B*s*l*h^2 * (same)                        one recompute included
HFU/MFU        ~ 1 + s/(6h)

activations/layer = s*b*h*(34 + 5*a*s/h)        Korthikanti 2205.05198 Eq 1, fp16, no TP
  tensor parallel t:        s*b*h*(10 + 24/t + 5*a*s/(h*t))
  TP + sequence parallel:   (s*b*h/t)*(34 + 5*a*s/h)
  selective recompute:      34*s*b*h/t          full recompute: 2*s*b*h

adam mixed precision = 16 bytes/param    2 bf16 W + 2 bf16 G + 4 fp32 master + 4 m + 4 v
                                         20 if gradients are kept in fp32
                                         8-bit Adam drops the state to 2 bytes/param

KV bytes = 2 * b * s * l * n_kv * d_head * p         the 2 is K and V
```

Cross-checked against real configs: Llama-3.1-8B (l=32, n_kv=8, d_head=128, bf16) gives exactly
131072 B/token = 128 KiB/token. Adam accounting sums to 16.

**The known-wrong variants, which circulate more than the right ones:**

- **`6ND` is wrong in three ways**: it drops `s/(6h)`, drops `V/(16lh)`, and ignores recompute.
  Compute the correction, then decide, rather than assuming attention is negligible: it is
  **1.14x** at `s=2048, h=4096`, **3.67x** at `h=8192, s=128k`, and **6.39x** at `h=4096,
  s=128k`. Chinchilla's own Table A4 puts the full count at 1.03 to 1.10x of 6ND for their
  configurations, which is where the "negligible" folklore comes from.
- **The attention correction is `s/(6h)`, not `s/(12h)`.** Kaplan publishes the ratio as
  `n_ctx/(12*d)` because that expression counts one attention matmul; there are two, `QK^T` and
  `AV`, at `2*b*s^2*h` each. Contraction-counted directly at b=2, s=512, h=256, a=8 the measured
  ratio is 0.33333, which is `s/(6h)` exactly. Using Kaplan's form halves the attention term.
- **Quoting MFU against wall-clock that included recompute** is the most common accounting error
  in the field. Model FLOPs over a hardware-FLOPs denominator flatters by `1 + s/(6h)`.
- **`6ND` for inference.** Decode is forward only, so `2N` per token.
- **`12*l*h^2` for a model with GQA, SwiGLU, or untied embeddings.** Check which. Counting
  embeddings while quoting `12*l*h^2` is exactly the conflation behind the Kaplan versus
  Chinchilla exponent discrepancy (2406.12907).
- **The `5*a*s/h` activation term applied to a FlashAttention model.** That term *is* the
  materialised `s x s` score matrix, which FlashAttention never writes. The constant 34 also
  assumes fp16, `d_ff = 4h` and a GELU MLP.
- **"Fewer trainable parameters means less memory"** is false. Activations and the frozen base
  dominate; trainable-parameter count is not a memory estimate.

## Roofline, and the only two numbers that matter

```
I*  = peak FLOP/s / peak bytes/s        the ridge, hardware only
I   = FLOPs / bytes moved               the kernel's arithmetic intensity
I < I*  memory bound        I > I*  compute bound
```

- A100 SXM: `312e12 / 2.04e12` = **153 FLOP/byte**. H100 SXM: `989e12 / 3.35e12` = **295**.
  Compute the ridge for the actual device rather than reusing a remembered number.
- **Two different decode intensities, do not conflate them.** Streaming the weights gives a
  whole-step intensity of roughly **the batch size**, which is why B=1 decode sits ~295x below
  the ridge and the GPU acts as a memory controller. The attention-over-KV part is separately
  about `(n_heads/n_kv)/p`, roughly **1 FLOP/byte at MHA fp16**, independent of `s`. GQA
  multiplies that by `n_heads/n_kv`.
- Llama-3.1-8B at S=4096 admits B_max around 102 before KV exhausts memory, still far under the
  295 ridge, so decode stays memory bound at every batch size that fits.
- FlashAttention (2205.14135) changes the **bytes** term, HBM traffic from `O(N*d + N^2)` to
  `O(N^2*d^2/M)` for SRAM size M, and *adds* FLOPs through recompute. Any claim that it makes
  attention cheaper in FLOPs, or asymptotically linear in compute, is wrong.
- Prefill is compute bound, decode is memory bound. A single number for "attention cost" that
  does not say which phase is meaningless. Batch-1 intensity does not describe a batched server.

## Precision

Get limits from the library, never from memory. `np.finfo` for fp32/fp16, `ml_dtypes.finfo` for
bf16 and fp8 (`np.finfo` raises on those), `torch.finfo` when torch is present. Verified:

| dtype | max | eps |
| --- | --- | --- |
| float32 | 3.40e38 | 1.19e-7 |
| float16 | 65504 | 9.77e-4 |
| bfloat16 | 3.39e38 | 7.81e-3 |
| float8_e4m3fn | 448 | 0.125 |
| float8_e5m2 | 57344 | 0.25 |

- bf16 has fp32's range with a **65x coarser epsilon**, so its failures are accumulation and
  cancellation, not overflow, and a float64 check never sees them.
- fp8 e4m3 saturates at 448, so the scaling factor is part of the maths, not an implementation
  detail. Per-tensor scaling is fragile; Hopper fp8 tensor cores accumulate at roughly 14 bits
  and need CUDA-core promotion every 128 elements.
- **Low-precision bugs are a function of token count, not model size.** A clean 100B-token fp8
  ablation proves nothing about a 15T run. SwiGLU outlier amplification only turns fatal past
  about 1T tokens.
- Deterministic round-to-nearest at 4 bits biases updates toward zero. Stochastic rounding is
  required, not optional.
- **Catastrophic cancellation is not theoretical.** `(x*x).mean() - x.mean()**2` on fp32 data
  centred at 1e6 returns **-131072.0**, a negative variance, where `np.var` on the same array
  returns 0.9989. Any near-equal subtraction of large numbers is suspect: recompute in fp64 and
  compare relative error.
- **Summation.** On `[1.0] + [1e-16]*100000`, naive and `np.sum` both lose the tail;
  `math.fsum` and Kahan are exact. Torch has no `fsum`, so use `t.sum(dtype=torch.float64)`,
  which recovers it where an fp32 accumulate returns a flat `1.0`.
- `np.log(1 + 1e-30)` is `0.0`; `np.log1p(1e-30)` is `1e-30`. Same for `expm1`. Use `mpmath` as
  ground truth when fp64 itself is the suspect.
- **Conditioning predicts the loss**: `log10(cond)` digits. Verified on Hilbert-8, `cond=1.5e10`
  gave a solve error of 1.3e-7, which is 1e-16 times 1e10 as advertised.
- **`torch.testing.assert_close` instead of a guessed `atol`.** Its dtype-aware defaults are
  fp64 (1e-7, 1e-7), fp32 (1.3e-6, 1e-5), fp16 (1e-3, 1e-5), bf16 (0.016, 1e-5). A hand-picked
  tolerance is usually wrong in one direction or the other.
- Softmax: subtract the max. `softmax(x)_i = exp(x_i - m)/sum_j exp(x_j - m)`,
  `lse(x) = m + log sum exp(x_j - m)`. The shifted forms are as accurate as the unshifted ones,
  and the division-free variant `exp(x_i - lse(x))` is measurably worse (Blanchard, Higham and
  Higham 2021). Naive softmax at logits of 800 returns `[nan, nan, 0]` while the shifted form is
  correct, and the two are symbolically identical. Check numerics separately from algebra.
- bf16 for training, loss scaling generally unnecessary. Adapters stay bf16, never quantised.

## Information theory, where the factors hide

```
PPL = exp(L_nats) = 2^(L_bits)
BPB = (L_nats / ln 2) * (N_tokens / N_bytes)
bits = nats / ln 2         nats = bits * 0.6931
```

- Per-token loss and perplexity are **tokenizer artifacts** and not comparable across
  vocabularies. BPB is the comparable unit. Any cross-model loss comparison that skips this is
  comparing tokenizers.
- nats vs bits is a constant factor, irrelevant to training and fatal to a written claim.
  `2^loss` on a nats loss is the classic version of this.
- `sympy.stats` does closed-form `E`, `variance`, `moment_generating_function`, `cdf`, `density`,
  with symbolic parameters (`E(Beta("B", a, b))` gives `a/(a+b)`).
- **There is no KL primitive, but the integral works**: `integrate(p1*log(p1/p2), (x, -oo, oo),
  conds="none")` returns the exact Gaussian KL. **Pass `conds="none"`** or it stalls forever on
  assumption branches. Same trick gives differential entropy.
- **What it cannot do**: anything with `Max`, `Piecewise` or an indicator. `E(Max(N, 0))` comes
  back as an unevaluated `Integral(Piecewise(...))`. Split the integral by hand, or use
  `scipy.stats`: `dist.expect(lambda v: max(v, 0))` gives `0.39894228040143276` = `1/sqrt(2*pi)`
  to 15 digits.
- `torch.distributions.kl_divergence` is the numeric cross-check, with 87 registered pairs
  including cross-family ones. **An unregistered pair raises `NotImplementedError` rather than
  falling back.** The Monte Carlo fallback `(p.log_prob(z) - q.log_prob(z)).mean()` is worth
  only about 3 digits at 2M samples, so it confirms a formula but never validates one.

## Scaling laws

```
L(N, D) = E + A/N^alpha + B/D^beta        entropy floor, capacity waste, evidence waste
N_opt ~ C^a,  D_opt ~ C^b,  a = beta/(alpha+beta),  b = alpha/(alpha+beta)
```

- Chinchilla (2203.15556) fitted `E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28`, three
  methods giving `(a,b)` of (0.50,0.50), (0.49,0.51), (0.46,0.54), and `D/N ~ 20` under `C=6ND`.
- Besiroglu et al. (2404.10102) refit approach 3 to `E=1.82, A=482.0, B=2085.4, alpha=0.3478,
  beta=0.3658, a=0.5126`, found the published `beta=0.28` was rounded from 0.2849 (about 13%
  prediction bias) and the confidence intervals implausibly tight. Roughly 20 tokens per
  parameter survives, but the interval spans about 4 to 40 at 1e26 FLOP. Quote the interval.
- **20:1 is not compute-invariant** (`a != b`, so it drifts with C) and is **not**
  inference-optimal. Kaplan's compute-optimal prescription is superseded. Nobody trains
  Chinchilla-optimal anyway, because inference cost dominates the lifetime budget.
**Fitting one of these is where the real errors are.** Measured end to end on synthetic data with
a true `alpha = 0.34`, lognormal noise and 9% outliers:

- **Naive `curve_fit` in linear space recovered `alpha = 0.577`, off by 70%.** Do not do this.
  The noise is multiplicative, so the fit must be too.
- Fitting in log space gets to 0.407. The working recipe is
  `least_squares(resid_in_log, loss="huber", f_scale=0.03, bounds=...)`, which recovered
  **0.3579 +/- 0.0329**. This is how Chinchilla-style fits are done.
- **`least_squares` does not return `pcov`.** Build it: `pcov = inv(J.T @ J) * (2*res.cost/dof)`
  from `res.jac`.
- **`curve_fit(..., absolute_sigma=)` changes the standard error by 7.3x** on identical data and
  identical `sigma` (0.0751 against 0.0103). The default `False` rescales by residual variance.
  Only pass `True` when the sigmas are genuine absolute error bars.
- Report uncertainty three ways, because they disagree: the t-interval from `pcov` (t, not
  normal, at small n), a bootstrap (400 resamples gave a visibly non-symmetric interval, trust
  it over the asymptotic one), and **the correlation matrix**. `corr(A, alpha) = 0.997` and
  `cond(J.T @ J) = 2.3e10`, so `E`, `A` and `alpha` are near-degenerate and a marginal interval
  on `alpha` alone badly understates the real uncertainty. Publish the `E`-versus-`alpha`
  tradeoff, not just the exponent.
- Extrapolation outside the fitted range is a new claim, labelled as one.

## muP, where the whole point is that nothing drifts with width

Tensor Programs V (2203.03466) Table 3, per fan_in:

| | init variance | Adam LR | SGD LR |
| --- | --- | --- | --- |
| input weights and all biases | `1/fan_in` | `O(1)` | `fan_out` |
| hidden weights | `1/fan_in` | `1/fan_in` | `O(1)` |
| output weights | `1/fan_in^2` | `1/fan_in` | `1/fan_in` |

- Attention logits use `q.k / d`, **not** `q.k / sqrt(d)` (Def 4.1).
- **The coordinate check is the test, and it is cheap.** Train widths 256 to 8192 for about 4
  Adam steps and plot the mean absolute coordinate size of every activation and logit against
  width. muP gives flat lines. Standard parametrisation grows or decays with width. An
  implementation that has not passed a coordinate check is unverified, whatever the code says.
- Common breakages: shrinking the embedding LR with width (it must stay `O(1)` under Adam),
  dropping the `1/fan_in` readout multiplier, dropping the `1/d` attention scaling.

## RL objectives, written out or not written at all

- GRPO: `A_i = (r_i - mean(r)) / std(r)`, k3 KL estimator `ratio - log ratio - 1`.
  - `std(r)` is an **unrequested difficulty reweighting**. Say so when it is in the objective.
  - `1/|o_i|` is a length bias. DAPO's token-level denominator `sum_i |o_i|` is the fix.
  - Zero-variance groups can be 40% of a batch. Count them before trusting a gradient estimate.
- GSPO: `s_i = (pi/pi_old)^(1/|y_i|)`, clip bounds **3e-4 / 4e-4**. Reusing GRPO's `eps=0.2`
  means **the clip never fires**, because a geometric mean concentrates near 1.
- DAPO: `eps_low=0.2`, `eps_high=0.28`, soft length penalty at `L_max=16384`, `L_cache=4096`.
- GSPO exists because GRPO pairs a per-token ratio with a per-sequence advantage. That is a unit
  mismatch, and it is the same class of error as adding tokens to sequences.
- K1 and K3 KL estimators stay flat through 700 steps of degradation, so a flat KL is not
  evidence of a healthy policy.

## LoRA and adapters

```
h = W0*x + gamma_r * B*A*x,   A ~ N(0, sigma^2),  B = 0
params per adapted d x k matrix = r*(d + k)
gamma_r = alpha / sqrt(r)      rsLoRA 2312.03732, the rank-stabilising choice
DoRA: W' = m * (W0 + BA) / ||W0 + BA||_c
```

- **`alpha/r` collapses adapter updates as `r` grows**, so raising `r` at fixed `alpha` shrinks
  the update as `1/r`. The "r=8 is enough" folklore is a scaling bug reported as a finding, and
  `alpha=2r` throws rank scaling away entirely.
- `alpha` behaves roughly as a learning-rate multiplier **under Adam only**, because of Adam's
  gradient-scale invariance. Under SGD the effect is quadratic. Do not quote it as an exact LR
  multiplier for an arbitrary optimiser.
- **LoRA learning rate is roughly 10x full fine-tuning.** Single most common configuration
  mistake.
- Settled: all-linear targeting, rsLoRA scaling, 10x LR. Attention-only and "rank 4 to 16" are
  dead. PiSSA/MiLoRA SVD init wins under SFT and collapses under RLVR.

## Diagnostics that are maths, not vibes

```
ratio_W  = rms(lr * update) / rms(W)             healthy ~1e-3, use the APPLIED update
delta_t  = log pi_train - log pi_rollout         healthy mean <1e-3, max <0.1
H / H_0                                          healthy band 0.5 to 0.9
```

- Comparing optimisers at equal nominal LR without matching update RMS compares two step sizes
  and then credits the algorithm.
- Grad-norm z-score against an EMA, and log **pre-clip**.

## The model is the oracle

Closed forms are cheap to write and easy to get wrong. Every one has a two-line empirical check,
and the check is the point:

```python
assert closed_form_params(cfg) == sum(p.numel() for p in model.parameters())

from torch.utils.flop_counter import FlopCounterMode        # measured, not estimated
with FlopCounterMode(display=False) as m:
    model(batch).sum().backward()
assert abs(m.get_total_flops() - closed_form_flops(cfg)) / m.get_total_flops() < 0.05

torch.cuda.reset_peak_memory_stats(); step()
assert torch.cuda.max_memory_allocated() <= closed_form_memory(cfg) * 1.1
```

- Disagreement means the **formula** is wrong until proven otherwise, not the measurement.
- `FlopCounterMode` counts matmuls and attention, not elementwise work, so it is the right
  referee for a `6ND`-style estimate and the wrong one for total device work. It is exact where
  it counts: a `64x128 @ 128x256` matmul reports `4194304` = `2*64*128*256`, and fwd+bwd over
  fwd comes out at exactly 3.0. `mods=` is deprecated; use `get_flop_counts()` for the breakdown.
- **The counter that silently reports zero.** A bare `F.scaled_dot_product_attention` on CPU
  counts **0 FLOPs** in `FlopCounterMode`, fvcore and ptflops alike, because CPU lowers to
  `_scaled_dot_product_flash_attention_for_cpu`, which is not in the 23-entry dispatch registry.
  Forcing `with torch.nn.attention.sdpa_kernel(SDPBackend.MATH):` gives the correct
  `2*(2*b*a*s*s*d)`. So: profile on CUDA, or force the MATH backend, or add the attention term
  analytically. A CPU profile that "confirms" 6ND has confirmed nothing.
- Pick the counter deliberately: `torch.utils.flop_counter` is the reference, `calflops` matched
  it to 1.004 (but hard-depends on `transformers`), `ptflops` needs `backend="aten"` (its
  `"pytorch"` backend is 8% off), `fvcore` has been unreleased since Dec 2022 and misses op
  types, and **`torchinfo` is not a FLOP counter at all**: its `total_mult_adds` came out 170x
  low because it omits the batch and sequence dimensions. Use torchinfo for params and shapes.
- Checkpoint round-trip equality on the same batch is the highest-value test in a training
  codebase. Cheap, and it catches what no derivation will.

## Settled, do not re-litigate

- Most claimed emergence is a metric artifact (2304.15004). 8 to 18% of HumanEval is leaked
  (2311.04850). Attention maps do not demonstrate attribution.
- Loss aggregation, normalisation and curriculum choices mostly move compute efficiency, not the
  asymptotic ceiling.
- BF16 replaced FP16 for training. Adapters stay bf16.

## Report

Everything `/smart-math` asks for, plus:

- the conventions table as actually chosen, at the top
- every counting formula with the empirical number it was checked against, and the percent gap
- for FLOPs claims: model or hardware, forward or forward+backward, and the correction factor
- for precision claims: the dtype it was checked in, never float64
- an arXiv ID or URL for every external claim, and which of the four sources it came from:
  a tool, the web, a repository document, or derived from the code
