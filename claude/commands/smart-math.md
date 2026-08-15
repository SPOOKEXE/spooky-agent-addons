---
description: Field-agnostic maths engine. Do the maths with a CAS, a solver, a fuzzer and a numeric prober instead of from memory, verify every step that survives into the answer, and hold multi-formula work in a symbol/relation context that can be resolved, impacted and audited. Hands off to a domain command where one exists, /ml-math for machine learning.
argument-hint: [equation, derivation, paper section, or the maths being iterated]
---

Work through: **$ARGUMENTS**

With no argument, find the maths currently under discussion in the repo or the conversation, and
say which you picked.

This is the engine, and it is deliberately field-agnostic: parse, verify, fuzz, track, report.
The field layer sits on top of it. If the maths is machine learning, run `/ml-math` instead,
which is this ladder plus the conventions, formulas and known-wrong variants of that field. For
a field with no dedicated command, Step 10 is how to bolt one together.

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
- one seed, then a verdict

## Deterministic first, and one seed is not a result

A check that gives a different answer on the second run has not checked anything. Seed
everything at the top of the script and print the seed with the verdict: `random.Random(seed)`,
`np.random.default_rng(seed)`, `torch.manual_seed(seed)`, `hypothesis` under
`@settings(derandomize=True)`, z3 under a fixed `timeout` rather than a wall-clock race. Where
a tool is nondeterministic by nature, pin what can be pinned and **name what could not**, since
that is where a flapping verdict will come from later.

Deterministic is not the same as sufficient. One seed samples one path through the domain, so
a single "no counterexample found" is the weakest possible evidence dressed as a result.

- **Default to 3 seeds.** Use **5** when the claim is load-bearing: it goes in a paper, it sets
  a hyperparameter, it gates a training run, or another derivation is about to build on it.
- More than 5 only when each run is cheap and the verdict sits near the boundary. If 5 seeds do
  not settle it, more seeds are not the missing ingredient, a different tier is.
- **The seeds must agree.** All agree, report the verdict with the seed count. Any disagreement
  and the verdict is **unknown**, reported with both outcomes and the seeds that produced them.
  A proof is never a majority vote, and the seed that agreed with you is never the one to quote.
- For numbers rather than verdicts (fits, condition numbers, timings, flip rates), report the
  median and the spread across seeds. If the spread crosses the decision boundary, the decision
  is not supported by the measurement, whatever the median says.

Report the seed list itself, not the word "seeded".

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
- **fuzzing the domain, and shrinking counterexamples**: `hypothesis`
- **inequalities, constraints, "does this hold for all x"**: `z3-solver`
- **fitting, quadrature, finite-difference gradient checks, structural analysis**: `scipy`,
  `networkx`
- **units and dimensions**: `sympy.physics.units`, or `pint` for measurement-heavy work
- **LaTeX in**: `lark`, and call `parse_latex(s, backend="lark")`. The default ANTLR backend
  demands `antlr4-python3-runtime` at exactly 4.11 and raises an ImportError otherwise
- **the field's own packages**: from the domain command, or Step 10 if there is not one

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

Where the quantities carry no units, this tier is whatever the field's cheap conservation law is
(Step 10): shapes and named axes, a density integrating to 1, a mass balance, a parity argument.
Do not skip the tier because the quantities are dimensionless. Two counts of different things
are still not addable, and `n_tokens + n_sequences` is a bug that no amount of algebra finds.

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

Restrict the draw to the declared domain, then deliberately push at its edge: zero, the sign
flip, tiny, huge, and the poles of anything in the expression. Run it under the seed set from
the section above, 3 by default, and a probe that finds nothing across all of them is worth
stating as "no counterexample in 3 x 200 draws over D", which is not the same sentence as
"proved". A hand-rolled loop like this one is the floor; Step 6 fuzzes properly.

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

n, k = Ints("n_items n_groups")
s4 = Solver(); s4.add(n == 768, k == 10, n % k == 0)
s4.check()         # unsat, that split does not divide evenly
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
step, which is what Step 8's closure query is for.

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

## Step 6. Fuzz the domain, then step outside it

The edge points in Step 2 catch what you thought to list. Fuzzing catches what you did not, and
out-of-bounds testing tells you whether the conditions attached to the claim are the real ones.

**Fuzz, and let it shrink.** The reason to use `hypothesis` over a sampling loop is not coverage,
it is **shrinking**: the loop reports the first failing point it happened to draw, and hypothesis
reduces it to the smallest one. On the same false identity the loop returned `w = -1000` and
hypothesis returned `w = -1.0`, which is the difference between a data point and an explanation.

```python
from hypothesis import given, strategies as st, settings, example

@settings(derandomize=True, max_examples=300)     # deterministic, no flapping CI
@example(0.0)                                     # pin every counterexample ever found,
@example(-1.0)                                    # one per line: @a @b is a matmul, not two
@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_identity(w):
    assert math.isclose(math.sqrt(w*w), w, rel_tol=1e-9, abs_tol=1e-12)
```

`derandomize=True` makes it repeatable, and `@example` turns every counterexample the fuzzer
has ever found into a permanent case, so a fixed bug stays fixed. Build strategies that respect
the declared domain (`st.integers(min_value=1)` for a count, `st.floats(0, 1)` for a
probability) rather than filtering afterwards, since a heavily filtered strategy just times out.

**Out of bounds is a test, not an accident.** Evaluate deliberately outside the stated domain
and read the result as three different findings:

- it **breaks immediately** outside: the condition is real and load-bearing. Record the failure
  mode next to it, because that is what makes the condition survive being edited later.
- it **still holds well outside**: the stated condition is over-tight, or copied from a source
  that needed it for a different reason. An assumption nobody needs is an assumption someone
  will drop silently, so chase it down rather than leaving it decorative.
- it **degrades**: find where. That boundary is the real validity condition, and it goes back
  into the relation's `valid_when` in Step 7 so `resolve` refuses to route through it out of
  range.

The standing out-of-bounds set: 0, plus and minus 1, plus and minus machine epsilon, plus and
minus 1/epsilon, the dtype's max and tiny, denormals, `n = 0` and `n = 1` and `n = 2`, an empty
input, exactly 0 and exactly 1 for anything that is a probability, and the excluded point of
every assumption in play.

**Collapse and rerank: the failures a scalar comparison cannot see.** A result can be numerically
close and still useless, because what was actually being used was the ordering or the structure.

- **Degenerate inputs**, which is where these live: all-equal scores, duplicate or near-duplicate
  rows, the zero vector, a single dominant element, exact ties, a rank-deficient matrix.
- **Reranking.** When the output orders things (top-k, argmax, selection, routing, a leaderboard),
  check the **permutation**, not the value. Perturb the input by the noise floor and count how
  often the top-k membership changes. Measured on 64 random scores under 1e-3 noise: **0.0 flip rate
  when well separated, 0.995 when near-tied.** The values moved by 1e-3 in both cases. A ranking
  with a 0.995 flip rate is noise wearing a result's clothing, and no amount of extra precision
  in the formula fixes it.

```python
def topk_flip_rate(scores, eps, k=5, trials=200, seed=0):
    r = np.random.default_rng(seed)
    base = set(np.argsort(-scores)[:k])
    return sum(set(np.argsort(-(scores + r.normal(0, eps, scores.shape)))[:k]) != base
               for _ in range(trials)) / trials
```

- **Conditioning, and why rank is the wrong detector.** `np.linalg.matrix_rank` applies a
  tolerance and happily reports full rank on a matrix that is numerically dead: an 8x8 with one
  column duplicated to within 1e-12 still comes back rank 8, while `np.linalg.cond` moves from
  54 to 2.06e13. Report `cond` at the sampled points. A formula that is fine at cond 1e3 and
  garbage at cond 1e12 has a validity condition nobody wrote down.
- **Saturation**, which is the same failure in the other direction: softmax at large inputs,
  sigmoid tails, `log` of near-zero, division by near-zero.

Every genuine failure point found here is worth more as a permanent case than as a sentence in
a report. Pin it with `@example`, and write it into the relation's validity condition.

## Step 7. Build the context only when the maths is bigger than one equation

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

## Step 8. The queries that pay for the context

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

## Step 9. A derivative is a claim until something independent agrees

Closed-form derivatives are where careful people make silent errors, and the error is usually a
sign, a factor, or a transpose, all of which look correct. Sympy will do the simple cases
(`sp.diff`, `sp.derive_by_array`, `sp.hessian`, and `sp.diff(sp.Trace(X.T*X), X).doit()` gives
`2*X`), but the check is a second, structurally different computation:

```python
from scipy.optimize import check_grad
check_grad(f, my_closed_form_grad, x0)          # finite differences, any callable

import torch                                     # when the maths is about torch code
x = torch.randn(7, 3, dtype=torch.double, requires_grad=True)   # float64, not float32
torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-5)
```

float64 is not optional; in float32 the finite-difference reference is noisier than the error
being hunted. Check at the awkward inputs as well as the easy ones: zero, the non-differentiable
kink, near-duplicate rows, and the point the assumption excludes.

For matrix and tensor derivatives, **write the layout convention down before deriving**,
numerator or denominator. Layout confusion produces results that are transposes of correct.
`/ml-math` carries the rest.

## Step 10. Specialising to a field

Every field has two things this engine needs, and they are worth finding **before** deriving
rather than after.

- **A cheap conservation law**, which becomes Tier 1 and catches an embarrassing fraction of
  errors for almost no cost.
- **An independent oracle**, a slow and obviously-correct computation to check the fast clever
  one against. Brute force at small n is a proof for the case it covers.

| field | tier 1, cheap | independent oracle |
| --- | --- | --- |
| physics, engineering | units and dimensions | limiting cases, known solutions |
| machine learning | shapes and named axes | autograd, and a counting loop |
| statistics | density integrates to 1, support | Monte Carlo simulation |
| combinatorics | parity, small-n sanity | exhaustive enumeration |
| finance | currency and time units, no-arbitrage | replication, path simulation |
| signal processing | energy, Parseval | FFT round trip |
| chemistry | mass and charge balance | stoichiometric solve |
| geometry | invariants, degrees of freedom | numeric construction |

Then find the field's own three things, which no general tool knows:

- **The notation ambiguity that bites**, resolved once and written down. Every field has one.
- **The known-wrong variant** of each canonical formula, since the wrong one circulates too and
  is usually the one that is easier to remember.
- **The reference values**, real numbers from real published cases, so a formula can be
  evaluated against something that is known to be right.

When a dedicated command exists for the field, use it rather than improvising: `/ml-math` for
machine learning. When one does not, build the three things above into the Step 7 context as
relations with provenance, and say in the report that the field layer was improvised.

## Step 11. Report

Every claim in the report carries how it was checked:

- the verdict word, proved or disproved or unknown, and the tier that produced it
- for disproved, the counterexample point, shrunk, and the size of the disagreement
- for unknown, the methods tried and the budget each was given
- the transforms used, by name, never "simplified"
- package versions, and the seed list, with any seed that disagreed called out
- the fuzz budget actually spent, as "no counterexample in 3 x 300 examples over D"
- the out-of-bounds points tried and what each did: held, degraded, or collapsed
- for anything ordered, the flip rate under noise-floor perturbation, not just the value
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
