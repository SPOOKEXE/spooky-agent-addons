---
description: Do maths with a CAS, a solver and a numeric prober instead of from memory, verify every step that survives into the answer, and hold multi-formula work in a symbol/relation context that can be resolved, impacted and audited.
argument-hint: [equation, derivation, paper section, or the model maths being iterated]
---

Work through: **$ARGUMENTS**

With no argument, find the maths currently under discussion in the repo or the conversation, and
say which you picked.

You cannot certify your own algebra. A wrong simplification and a right one are the same
number of tokens and read identically. So derive however you like, but **every claim that
survives into the answer has been through a tool that would have failed had the claim been
wrong**, and anything that could not be checked ships labelled unverified.

Not this:

- doing the algebra in your head, then presenting the result as if a tool produced it
- building a maths MCP, a CAS, or a symbolic engine. The packages exist and are better
- `simplify()` as the answer to every question
- saying "verified" when only one tier of the ladder ran
- `pip install` or `uv install` into the project's environment
- a registry of variables and formulas for a problem that has two of each

## Step 0. An environment, without touching the project's

Bare system python usually has none of this, and the project's venv is not yours to modify.
Use an ephemeral one, which costs nothing after the first resolve:

```bash
uv run --with sympy --with mpmath --with numpy python math_check.py
```

If the project already has the packages in its own venv and the maths is about the project's
code, use the project's interpreter instead, so what you check is what runs. Adding a package
to the project is a question for the user, never a side effect.

The ladder, add only what the task needs:

- **always**: `sympy`, `mpmath`, `numpy`
- **inequalities, constraints, "does this hold for all x"**: `z3-solver`
- **shapes and contraction cost**: `opt_einsum`, `einops`, `jaxtyping` + `beartype`
- **gradients against autograd**: the project's `torch` or `jax`, not a fresh one
- **LaTeX in**: `lark`, and call `parse_latex(s, backend="lark")`. The default ANTLR backend
  demands `antlr4-python3-runtime` at exactly 4.11 and raises an ImportError otherwise
- **fitting, quadrature, structural analysis**: `scipy`, `networkx`

Print `sympy.__version__` and `mpmath.__version__` in the header of every script and in the
report. Symbolic output is version-dependent, and a result nobody can reproduce is not a
result. Verified on sympy 1.14.0 and mpmath 1.3.0.

## Step 1. Parse once, hold handles, never retype an expression

Every retype of an expression is a new chance to drop a factor of two. Write **one scratch
module** that constructs each symbol and expression exactly once, and import it from every
check. `math_check.py` beside the work, or in the session scratchpad if the repo forbids new
files. It is rerunnable, it takes no arguments, and it prints its verdicts.

```python
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
import sympy as sp

n = sp.Symbol("n", integer=True, positive=True)   # assumptions at construction
sigma = sp.Symbol("sigma", positive=True)
local = {"n": n, "sigma": sigma}

expr = parse_expr(user_string, local_dict=local,
                  transformations=standard_transformations)
```

`parse_expr` with an explicit `local_dict`, never `sympify` or `eval` on a string, and never a
bare `Symbol("x")` where the real x is positive.

**Assumptions are most of the correctness.** `sqrt(x**2)` simplifies to `x` when x was declared
positive and stays `sqrt(x**2)` when it was not, and that is sympy being right both times. The
overwhelming majority of wrong symbolic answers are a missing assumption rather than bad
algebra. Before the first transform, write down for each symbol: real or complex, sign, integer
or not, and the range it is meant to live in. If the source does not say, that is a question
for the user or an assumption stated out loud in the report.

## Step 2. The verification ladder

Cheapest tier first, stop at the first tier that gives a definite answer, and **report which
tier gave it**. Never return a bare boolean.

The verdict vocabulary is three words, and there are only three:

- **proved**, with the method that proved it
- **disproved**, with the counterexample point
- **unknown**, with the methods tried and the budget they were given

**Tier 1, dimensions and shapes.** Cheap and catches an embarrassing fraction. Use
`SI._collect_factor_and_dimension`, which raises on an inconsistent sum, and not
`SI.get_dimensional_expr`, which returns `length` for `meter + second` without complaint:

```python
from sympy.physics.units.systems.si import SI
SI._collect_factor_and_dimension(meter + second)
# ValueError: Dimension of "second" is Dimension(time, T), but it should be Dimension(length, L)
```

For unitless ML quantities the same tier is the shape check in Step 9. Do not skip it because
the quantities are dimensionless: `n_tokens` and `n_sequences` are both counts and adding them
is still wrong.

**Tier 2, structural.** `sp.srepr(a) == sp.srepr(b)`, or `a == b` for structural equality.
Instant, and settles the boring cases where the two expressions really are the same tree.

**Tier 3, symbolic zero test.** `sp.simplify(a - b) == 0`, and when that stalls try the
targeted ones: `trigsimp`, `radsimp`, `ratsimp`, `cancel`, `factor`, `logcombine`, `powsimp`.
Say which one closed it. A zero here under the declared assumptions is a proof.

**Tier 4, randomised numeric probe.** This is where wrong answers get caught, and where a
careless prober invents right ones. **Sample the boundary, not the comfortable interior.**
Drawing `w` from `U(0.1, 3)` reports `sqrt(w**2) == w` as holding, every time, forever. This
one is measured, not hypothetical:

```python
import random, sympy as sp

def find_counterexample(e1, e2, syms, n=200, tol=1e-8, seed=0):
    """First point where e1 and e2 disagree, or None. Edge points before random ones."""
    rng = random.Random(seed)
    edges = [-1.0, 0.0, 1.0, -1e-6, 1e-6, 1e3, -1e3]
    for i in range(n):
        edge = i < len(edges) * len(syms)
        sub = {s: sp.Float(rng.choice(edges) if edge else rng.uniform(-5, 5))
               for s in syms}
        try:
            d = complex(sp.N(e1.subs(sub) - e2.subs(sub), 30))
        except (TypeError, ValueError):
            continue
        if abs(d) > tol or d != d:                     # d != d catches NaN
            return {str(k): str(v) for k, v in sub.items()}, abs(d)
    return None
```

Fixed seed, always. Restrict the draw to the declared domain, then deliberately push at its
edge: zero, the sign flip, tiny, huge, and the poles of anything in the expression. A probe
that finds nothing is worth stating as "no counterexample in 200 draws over D", which is not
the same sentence as "proved".

Then check the counterexample before believing it. Pushing at the edge means landing on
singularities, and a NaN at a removable one is a domain artefact rather than a refutation: this
prober reports `z = -1` with a NaN for `1/(z-1)` against `(z+1)/(z**2-1)`, which are the same
function everywhere the second one is defined. A counterexample is only a disproof once it has
been evaluated at nearby points and found to be a genuine disagreement rather than a hole.

**Tier 5, the solver, for anything quantified or conditional.** When the claim is an
inequality, or holds only under constraints, z3 decides it properly. Negate the claim and ask
for a model: `unsat` is a proof, `sat` hands back the counterexample, `unknown` is honest.
All four of these are verified runs:

```python
from z3 import Reals, Ints, Solver, Not, Implies, And, sat

x, y = Reals("x y")
s = Solver()
s.add(Not(Implies(And(x > 0, y > 0), x*x + y*y >= 2*x*y)))
s.check()          # unsat, so the inequality is proved

s2 = Solver(); s2.add(Not(Implies(x > 0, x*x >= x)))
s2.check(), s2.model()      # sat, [x = 1/2], claim is false at x = 1/2

d, h = Ints("d_model n_heads")
s4 = Solver(); s4.add(d == 768, h == 10, d % h == 0)
s4.check()         # unsat, that head count does not divide that width
```

Nonlinear real arithmetic is not complete in z3 and it will return `unknown` or run forever, so
always `s.set("timeout", 2000)`. `unknown` after a timeout is a reportable outcome, not a
failure to hide.

## Step 3. Check the derivation, not just the endpoint

An endpoint that matches by luck hides a broken middle, and the broken middle is what gets
copied into the next derivation. Lay the steps out as a list and check each adjacent pair, then
report **the index of the first failure**, not a global verdict:

```python
steps = [e0, e1, e2, e3]                      # each an expression, or an Eq
for i, (a, b) in enumerate(zip(steps, steps[1:])):
    verdict = check_equivalence(a, b)         # the ladder from Step 2
    if verdict.status != "proved":
        report(f"step {i} -> {i+1}: {verdict}")
        break
```

Everything after the first bad step is meaningless, so stop there. When a step is a rewrite
rather than an identity, name the rule that licenses it (chain rule, linearity of expectation,
Cauchy-Schwarz, iid factorisation) and check the rule's own precondition holds at that point.
A step that is only valid in a limit or a regime carries that condition forward to every later
step, which is what Step 7's closure query is for.

## Step 4. Name the transform, do not reach for `simplify`

`simplify` is ill-defined, irreproducible across versions, and expensive. It is a fine
exploratory poke and a bad thing to put in a report. State the intent instead:

`factor`, `expand`, `collect(var)`, `cancel`, `together`, `apart`, `trigsimp`, `logcombine`,
`powsimp`, `radsimp`, `refine(expr, Q.positive(x))`.

And the calculus surface, each of which takes a domain or an assumption you should be passing:
`diff`, `integrate`, `limit(..., dir=)`, `series`, `summation`, `solveset(..., domain=S.Reals)`,
`dsolve`, `rsolve`, `linsolve`, `nonlinsolve`.

Expect the output to be correct and ugly. The closed-form KL between two normals comes back
with `log(s2**(2*s2**2)/s1**(2*s2**2))` sitting in it, which is right but unreadable; a
`logcombine` or `powsimp` under positivity assumptions is what makes it presentable. Presenting
raw CAS output as the final formula is half the job.

Render at the end, from the expression, never by hand: `sp.latex(expr)` for the paper form,
`sp.pretty(expr)` for the terminal, and `sp.cse([...])` before `sp.lambdify` when the expression
is going into code, so the shared subexpressions are computed once. Retyping a formula into
LaTeX by hand puts an unchecked expression in the report, which is exactly what Step 1 exists
to prevent.

**Going the other way, when a number is what you have.** `mpmath.identify` and `nsimplify`
invert a decimal back to a closed form, which is the fastest way to recognise what an
experiment just produced:

```python
mpmath.identify(mpmath.mpf("1.2020569031595942854"), ["zeta(3)"])   # (1*zeta(3))
sp.nsimplify(0.61803398874989484820, [sp.sqrt(5)])                  # -1/2 + sqrt(5)/2
mpmath.pslq([mpmath.pi, mpmath.atan(1)], maxcoeff=100)              # [1, -4]
```

`identify` needs the constants guessed for it, so feed it the ones the problem is made of.

## Step 5. Budget every call

`integrate` will hang forever given the chance, and so will `solve`, `dsolve` and nonlinear z3.
Bound them, and treat the bound as data:

```python
import signal

def with_timeout(fn, secs=5):
    def handler(signum, frame): raise TimeoutError("budget exceeded")
    old = signal.signal(signal.SIGALRM, handler); signal.alarm(secs)
    try:    return fn()
    finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)
```

Main thread and POSIX only; from a worker thread, spawn a subprocess with a timeout instead.
When the budget fires, the answer is "unknown, integrate timed out at 5s", which is a fact
about the tool. It is never "this has no closed form", which is a claim about mathematics that
the timeout did not establish.

## Step 6. Build the context only when the maths is bigger than one equation

Everything above works on one expression at a time, and cannot see the errors that live
*between* formulas: B meaning sequences in one equation and tokens in another, a factor of two
that only appears where two definitions meet, a result still standing on an iid assumption that
was dropped three iterations ago.

Build the context when **three or more formulas interact**, or when the same maths is being
iterated across turns. For two equations it is bookkeeping, and bookkeeping that costs more
than it catches is abandoned in week two.

It is a **file**, not a service. One `context.py` or `context.json` beside the work, holding
two lists, loaded by the same scratch module from Step 1.

A **symbol** is not a name:

- `name`, and `meaning` in one line of prose
- `domain`: the sympy assumptions, in the same words used at construction
- `shape` for tensors, or `units` for physical quantities
- `role`: free, measured, derived, or hyperparameter
- `aliases`: theta in the paper, `W` in the notes, `weight` in the code
- `source`: paper section, file and line, or "chosen by us on <date>"

A **relation** is not an equation string:

- `kind`: definition, constraint, assumption, derived, approximation, or empirical-fit
- `expr`, stored **undirected**. Baking in `y = f(x)` at definition time throws away exactly
  the flexibility that makes the graph worth having; direction is chosen at resolve time
- `valid_when` and `error_term`, mandatory for approximations
- `provenance` and `verified`: which tier of the ladder passed, and when

**The registry rots if it is tedious.** Unknown symbols encountered in a new relation get
auto-registered as provisional with their type inferred from use, and get flagged in the audit
rather than raising. Demand an explicit declaration only where the ambiguity actually bites,
which in practice is units, shapes, and anything counted.

## Step 7. The queries that pay for the context

Six verbs, parameterised by noun. Not twenty near-duplicate tools.

**`resolve(target, given)`** is a real algorithm and not a lookup. Build the incidence matrix of
relations against unknowns, then:

```python
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import structural_rank
import networkx as nx

inc = csr_matrix(incidence)                  # relations x unknowns, 1 where used
rank = structural_rank(inc)                  # < n_unknowns means under-determined
                                             # < n_relations means over-determined
blocks = list(nx.strongly_connected_components(dep_graph))   # simultaneous blocks
order  = list(nx.topological_sort(nx.condensation(dep_graph)))  # solve order
```

`structural_rank` tells you the system is singular before any solve is attempted, the
condensation gives the block lower-triangular solve order, and each strongly connected
component is a block that has to be solved simultaneously. **Return the path, not just the
value**: which relations were used, in what order, under which assumptions. This is what
acausal modelling tools do and it is worth stealing outright.

Flag any resolve path that chains two or more `approximation` relations. Compounding
approximation error silently is a standard way to produce a confidently wrong number.

**`impact(symbol)`** is `nx.descendants(graph, symbol)`, paired with staleness: change a
relation and everything downstream of it flips to unverified until its check is rerun. This is
`make` for mathematics, and it is what makes iterating the maths safe instead of frightening.

**`assumption_closure(result)`** walks back to every assumption the result rests on. Then invert
it: "if iid is relaxed, what dies?" That single query justifies building the graph.

**`audit`** replaces one vague "orphaned" list with six questions that have six different fixes:

- **undefined**: used in a relation, never declared
- **unused**: declared, never referenced
- **unreachable**: cannot be computed from any known set
- **unverified**: derived, but the derivation never passed a tier
- **shadowed**: one name, two meanings, across scopes or across paper-versus-code
- **dangling**: references a symbol that was deleted

**`check_consistency`** runs the Step 2 ladder pairwise across relations that share symbols, and
**returns witnesses**. "Eq 3 and eq 7 conflict" is nearly useless. "At n=10, d=64, eq 3 gives
0.5 and eq 7 gives 0.7" is actionable. Never report "consistent". Report "no contradiction found
by tiers 1 to 4 at 200 samples".

**`fork` and `diff`** for variants: the paper's formulation against ours, v1 against v2, dense
against MoE. Fork the context file, change one thing, diff the resolved values. When
reconciling a paper against an implementation, match on semantics and shape to propose the
symbol mapping, then make the user confirm it.

## Step 8. Where a generic CAS stops being enough: gradients and layout

**Declare the layout convention once, in writing, at the top of the file.** Numerator layout or
denominator layout. Layout confusion silently ruins more matrix derivations than every other
cause combined, and it produces results that are transposes of correct, which look correct.

Sympy will do some of it directly, for example `sp.diff(sp.Trace(X.T*X), X).doit()` gives `2*X`,
and `sp.derive_by_array` plus `sp.hessian` cover the elementwise cases. But a closed-form
gradient is a **claim** until autograd disagrees or fails to:

```python
import torch
x = torch.randn(7, 3, dtype=torch.double, requires_grad=True)   # float64, not float32
torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-5)          # analytic vs finite diff
torch.testing.assert_close(my_closed_form(x), torch.func.jacrev(f)(x))
```

float64 is not optional here; in float32 the finite-difference reference is noisier than the
error being looked for. Test at the awkward inputs too: at zero, at the non-differentiable
kink, at near-duplicate rows, at the point the assumption says is excluded.

## Step 9. Shapes, cost, and precision

**Shapes.** Name the dimensions once and resolve the ambiguity explicitly. `B` is tokens or
sequences, never both, and writing down which one it is here is worth more than any later
check. `jaxtyping` annotations plus `beartype` enforce it at runtime; `einops` patterns fail
loudly and with the actual numbers, for example refusing to split an axis of length 6 into
`h=4`. Prefer a named-axis rearrange over a bare `.view()` in anything being derived.

**Contraction cost.** `opt_einsum.contract_path` gives FLOPs and peak intermediate size without
running anything:

```python
path, info = oe.contract_path("bqhd,bkhd->bhqk", q, k)
info.opt_cost               # 1.718e10 FLOPs at B=8, S=1024, H=16, D=64
info.largest_intermediate   # 1.342e8 elements, the score matrix
```

Note when the optimised path equals the naive one, as it does for a two-operand contraction:
there is no reordering to win, and the only remaining lever is not forming the intermediate at
all. Report both numbers, since "fewer FLOPs at four times peak memory" is not a win.

**Architecture maths.** Parameters, FLOPs per token, activation memory and KV-cache size go into
the context as relations, and then `resolve(peak_memory, given={...})` shows its work.
Then cross-check the closed form against reality, which is two lines and catches a wrong
formula immediately:

```python
assert closed_form_params(cfg) == sum(p.numel() for p in model.parameters())
```

**Precision.** Probe the expression at the representable limits of the dtype actually in use,
not in float64. Check where it overflows, where subtracting nearly equal numbers destroys the
result, and whether the softmax is the max-subtracted form. Integer and divisibility
constraints (`d_model % n_heads`, sequence length against block size, vocab against tensor
parallel degree) go to z3, which answers them exactly.

**Scaling laws and fits.** An empirical fit is a relation of kind `empirical-fit` and carries
its fitted range. Extrapolating outside it is a new claim, and gets said out loud.

## Step 10. Report

Every claim in the report carries how it was checked:

- the verdict word, proved or disproved or unknown, and the tier that produced it
- for disproved, the counterexample point and the size of the disagreement
- for unknown, the methods tried and the budget each was given
- the transforms used, by name, never "simplified"
- package versions, and the seed
- the assumptions the result rests on, from the closure query
- what was left unchecked, and what would catch it

Attach the script. It reruns from nothing, prints the same verdicts, and lives beside the work
or in the scratchpad. Someone rerunning it in a month is the entire point.

Where an unverifiable claim is load-bearing, say so plainly. A CAS is ordinary software with
ordinary bugs and no formal guarantee, so an important result that survived only tier 3 is worth
a second, structurally different check rather than a confident sentence. Escalating to a proof
assistant is real but expensive, and is a decision for the user, not a thing to start unasked.

Commit the check script together with whatever it verifies, in one commit, only when the user
asks for a commit.
