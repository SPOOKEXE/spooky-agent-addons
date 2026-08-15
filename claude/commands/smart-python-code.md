---
description: The Python specialisation of /smart-simple-code. Slotted dataclasses and dict rows, module-level containers resolved at call time, the four-way guard ladder where assert is not a guard, monotonic deadlines, asyncio and the free-threading era where GIL-safe code stops being safe, dict dispatch over match, and pydantic at the boundary only.
argument-hint: [the module, feature, or system to write, extend, or refactor]
---

Build: **$ARGUMENTS**

`/smart-simple-code` holds the reasoning. This file is the Python answer, measured on CPython 3.12
with free-threaded 3.13t, 3.14t and 3.15t checked where it matters.

## Module shape

```python
@dataclass(slots=True)
class Row:
    eid: int
    hp: int
    ready_at: float

_rows: dict[int, Row] = {}

def row(eid: int) -> Row:
    r = _rows.get(eid)
    if r is None:
        r = Row(eid, 100, 0.0)
        _rows[eid] = r
    return r
```

Measured, nanoseconds per operation and bytes per instance:

| shape | construct | read | write | bytes |
| --- | --- | --- | --- | --- |
| tuple | 3.4 | 5.5 | | 104 |
| dict literal | 33.9 | 7.4 | 10.5 | 224 |
| **dataclass, `slots=True`** | **62.1** | **5.5** | **5.5** | **96** |
| plain class | 64.1 | 5.5 | 5.5 | 128 |
| dataclass | 66.0 | 5.4 | 5.5 | 128 |
| NamedTuple | 91.8 | 10.2 | | 112 |
| dataclass, `frozen=True` | 196.7 | 5.4 | raises | 128 |

- **`slots=True` is a memory win, not a speed win**: 128 to 96 bytes, 25% off, with attribute access
  identical. Worth it past ten thousand instances, noise below that.
- **`frozen=True` costs 3x on construction** (66 to 197 ns) because every field goes through
  `object.__setattr__`. Frozen is for value types you build rarely. Never for a per-frame row.
- **NamedTuple attribute access is 2x a dataclass** (10.2 against 5.4 ns) because each name is a
  property over a tuple index. It is not the cheap option it looks like.
- **`TypedDict` is a plain `dict` at runtime.** It accepted `{"hp": "x"}` without complaint. It is a
  static-checking annotation and nothing more.
- For genuinely hot rows, a plain dict is the fastest to build and the most expensive to read. Pick
  by which you do more of.

## Wiring

```python
_c: Container | None = None

def wire(c: Container) -> None:          # phase 2: capture only
    global _c
    _c = c

def act(x) -> tuple[bool, str]:
    if _c is None: return False, "not wired"
    _c.bus.publish(x)                    # phase 3: resolve at call time
    return True, ""
```

- Resolving through the module global costs **6.1 ns**. An `lru_cache` accessor costs 15.8 ns, about
  2.7x, and buys nothing here.
- **`functools.lru_cache` is not a thread-safe singleton.** Verified: eight threads ran the factory
  body eight times and produced eight distinct objects. It does not hold a lock across your function.
  Use the module global with an `if None` check, or an explicit `threading.Lock`.
- **Capturing a member at construction time captures whatever was there**, which during wiring is
  usually `None`, permanently. Verified: `self.bus = c.bus` in `__init__` froze `None`; a `@property`
  reading `self._c.bus` saw the real value later. Store the container, not its members.
- **Circular imports fail in both directions** with `cannot import name ... from partially
  initialized module`. Two fixes work: import inside the function, or **`import module_b` and call
  `module_b.func()`**, since module-object attribute lookup is deferred to call time. Prefer the
  second, which costs nothing at runtime and reads normally.
- **A module-level `BUS = Bus()` runs at import time**, which quietly makes import order into
  construction order. That is the Python version of the static initialisation order problem.

## Guards

```python
if row is None: return                                  # bad input, silent
if now() < row.ready_at: return False, "cooling down"   # policy refusal
if cfg.get("rate") is None: log.warning("missing rate, using 3")
if kind not in TABLE: raise ValueError(kind)            # programmer error
```

- Returning `None` and returning `(ok, reason)` cost the same, 16.9 against 16.7 ns. There is no
  performance argument for the less informative one.
- **Raising and catching costs 117.9 ns, about 7x a return.** Exceptions are for the exceptional rung
  of the ladder, and cheap enough that this is a design argument rather than a performance one.
- **`assert` is stripped under `-O`.** Verified: the same code under `python -O` ran an invalid input
  straight through with no `AssertionError` and `__debug__` false. Never validate anything with
  `assert` that must still be checked in production. Internal invariants only.
- **`warnings.warn` deduplicates by default.** Verified: called five times, shown once, because the
  default filter is once per location. **Missing-config warnings must use `logging.warning`**, or the
  second entity with the same problem is silent. It also costs 161.9 ns even when filtered out,
  against 46.3 ns for a disabled `log.warning`.

## Time

```python
from time import monotonic as now
if now() < row.ready_at: return
row.ready_at = now() + COOLDOWN
```

- **`time.monotonic()`**, 34.0 ns. `perf_counter()` is 33.5 ns and on Linux is the same
  `CLOCK_MONOTONIC`, though not on every platform, so do not treat them as interchangeable and never
  subtract one from the other: their epochs are unrelated and arbitrary.
- The `_ns` variants are about 10% cheaper and return integers, which removes the float question
  entirely.
- **`time.time()` is the wrong clock twice over.** It reports `adjustable=True, monotonic=False`, so
  NTP can move it backwards, and its epoch offset eats float precision:
  **`math.ulp(time.time())` is 238 ns against 0.015 ns for `monotonic()`**.
- **The decremented-timer drift is real, and it is not floating point.** A thousand ticks decrementing
  by the *nominal* delta finished at 1099.3 ms against a real 1000 ms, nearly 10% late, because the
  loop's idea of elapsed time was never the real one. The deadline comparison finished at 1000.0 ms.
  A counter tracks your loop; a deadline tracks the clock.
- `loop.time()` is monotonic, so an asyncio deadline can use it directly.

## Concurrency

```python
_tasks: set[asyncio.Task] = set()
t = asyncio.create_task(coro())
_tasks.add(t); t.add_done_callback(_tasks.discard)     # or use asyncio.TaskGroup
```

- **asyncio keeps only a weak reference to tasks**, so an unreferenced `create_task` can be garbage
  collected mid-flight. Hold them in a set, or use `asyncio.TaskGroup` (3.11+), which holds strong
  references and joins on exit. Prefer `asyncio.timeout()` over `wait_for`.
- Costs: `loop.call_soon` to schedule 1150 ns, `create_task` plus await 7806 ns, `asyncio.to_thread`
  warm 50.8 us, and a `threading.Thread` create-start-join 1114 us on 3.12.
- **`Task.cancel()` raises at the next await and cannot preempt synchronous code.** Verified: a task
  with no awaits ran all three million iterations after being cancelled. Re-raise `CancelledError`
  after cleanup rather than swallowing it, and keep the identity re-check for the rest.
- `loop.call_soon_threadsafe` is the only legal door from another thread into the loop.

**The GIL, and the fact that this is now a live question.** Four CPU-bound threads, measured:

| build | speedup against serial |
| --- | --- |
| 3.12 | 0.95x |
| 3.13t with the GIL on | 0.74x, **slower than not threading at all** |
| 3.13t with the GIL off | 3.05x |
| 3.15 free-threaded | 3.81x |

IO-bound threads are fine either way, at 3.89x on four sleeping threads.

- **Free-threaded builds ship, and 3.14t and 3.15t have the GIL off by default** where 3.13t needed
  opting in. Detect with `sysconfig.get_config_var("Py_GIL_DISABLED")`, **not** `sys._is_gil_enabled`,
  which returns `True` on a stock 3.13 as well and tells you nothing.
- **The trap that will define the next few years**: four threads doing `c.n = c.n + 1` two hundred
  thousand times each produced the correct 800000 under the GIL and **445297** on 3.14t. Code that
  was accidentally safe because of the GIL now silently corrupts. If your code will ever run on a
  free-threaded build, every shared mutable counter needs a lock or needs to stop being shared.
- Only plain data crosses a boundary. `multiprocessing` enforces this for you: lambdas, nested
  functions and `threading.Lock` all fail to pickle.

## Control flow as data

```python
TRANSITIONS = {(S.IDLE, "see"): S.WALK, (S.WALK, "die"): S.DEAD}

def send(row, ev) -> tuple[bool, str]:        # the only writer of row["state"]
    nxt = TRANSITIONS.get((row["state"], ev))
    if nxt is None: return False, "no transition"
    row["state"] = nxt
    return True, ""
```

Dispatch cost per call, measured:

| form | ns |
| --- | --- |
| `dict` keyed by `str` | **16.6** |
| `match` on `str` | 35.0 |
| if/elif chain | 54.5 |
| `dict` keyed by `Enum` | 52.5 |
| `match` on `Enum` | 64.1 |

- **A dict keyed by strings is the fastest and stays data.** Everything else is either slower or
  code.
- **Enum keys cost 3x string keys** because `__hash__` and `__eq__` are Python-level, and
  `SomeEnum(value)` reverse lookup costs 150 ns. Use `StrEnum` for the type-checker benefit and index
  hot dictionaries with the raw string.
- **`match` on an Enum is the slowest option of all**, because it compiles to sequential `is`
  comparisons rather than a jump table. It reads beautifully and it is not a dispatch mechanism.
- A hand-rolled eight-node tree ticks in 0.30 us. **`py_trees` takes 8.75 us for five nodes**, about
  29x, and pulls in `pydot`. Take it for its visualisation and blackboard tooling, not to avoid
  writing forty lines.

## Config

| path | ns per record |
| --- | --- |
| plain dict | 6.5 |
| `dataclass(**raw)` | 254.9 |
| pydantic `TypeAdapter` | 601.2 |
| pydantic `model_validate` | 672.7 |
| pydantic `model_construct` | 1118.6 |

- **Pydantic at the boundary only**, where untrusted JSON or config arrives, then hand dataclasses or
  dicts inward. Attribute reads on a pydantic model cost 21.2 ns against 5.4 for a dataclass, so it is
  a validation tool and not a data structure. Note `model_construct` is *slower* than validating,
  despite being the "skip validation" path.
- **Pydantic coerces by default**: `"30"` became the integer 30. Pass `strict=True` for config, which
  correctly rejected it.
- **Dataclasses do zero runtime validation.** `Spec(name=1, hp="abc", speed=None)` constructs happily.
  If the data is untrusted, a dataclass is not a check.

## Tooling and validation

Run a linter, one type checker, and tests. The overlap between them is smaller than people assume.
Measured on one 65-line file with deliberate bugs:

- **ruff only** (0.03 s): unused imports and locals, bare `except`, and **`B006` mutable default
  argument**, which no type checker flags.
- **type checker only** (mypy 0.56 s, pyright 0.59 s): wrong return type, dereferencing an `Optional`,
  writing a `str` into a `dict[str, int]`, and assigning outside `__slots__`.
- **Neither, tests only**: an `IndexError`, an `a - b` where `a + b` was meant, and the observable
  effect of that mutable default.
- **mypy and pyright agreed on five of seven findings**, so running both is low value. Pick one.
  `--warn-unreachable` is **not** implied by `--strict`, so enable it explicitly.
- Detect the project's answer from `pyproject.toml` before imposing any of this. Newer checkers
  install cleanly and are worth watching, but do not introduce a pre-1.0 one into someone's project.
