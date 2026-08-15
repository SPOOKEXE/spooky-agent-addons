---
description: The C++ specialisation of /smart-simple-code. Aggregates and free functions, entity stores chosen by measurement, the static initialisation order fiasco and its fix, guard ladders with std::expected, steady_clock deadlines and why high_resolution_clock is a trap, jthread with stop_token, and the sanitizers that catch what tests do not.
argument-hint: [the module, feature, or system to write, extend, or refactor]
---

Build: **$ARGUMENTS**

`/smart-simple-code` holds the reasoning. This file is the C++ answer, measured on g++ 13.3 at `-O2`.
Prefer `-std=c++23`, which is fully supported there; note that **`std::expected` does not exist under
`-std=c++20`**, so check the standard before reaching for it.

## Module shape

```cpp
enum class PlayerId : std::uint32_t {};              // strong id, no implicit conversions

struct CombatRow { Health hp{}; float cooldown = 0; bool in_combat = false; };  // rule of zero
struct Combat    { std::unordered_map<PlayerId, CombatRow> rows; };

CombatRow& combat_row(Combat& m, PlayerId id) {      // the single lazy-create door
    return m.rows.try_emplace(id).first->second;
}
```

- **An aggregate plus free functions.** Public data, no constructors, no methods. `try_emplace` is the
  lazy row: it constructs only if absent and returns the existing one otherwise.
- **A class earns its place when there is a real invariant to hold.** `Health` keeping
  `0 <= current <= max` is a class with a private constructor and a
  `static std::optional<Health> make(...)` that rejects the impossible.
- **Trap: adding any user-declared constructor silently kills aggregate initialisation** and with it
  designated initialisers. A config struct that gained one constructor for convenience breaks every
  `Config{.window = 0.18f}` call site, and the error message will not mention the constructor.
- Strong ids as empty enum classes cost nothing and stop you passing a player id where an item id
  belongs.

## The entity store, chosen by measurement

100k entities, 20-byte rows, ids sparse over a million:

| shape | iterate | random access | memory |
| --- | --- | --- | --- |
| `unordered_map<Id, Row>` | 1.93 ns/entity | 4.82 ns | ~4.6 MiB |
| dense vector plus hash index | 0.37 ns/entity | 5.23 ns | 1.9 MiB plus index |
| sparse set | **0.37 ns/entity** | **0.74 ns** | 1.9 MiB plus 3.8 MiB sparse array |

- **Default to `unordered_map` and switch when a profile says iteration dominates.** The map's
  iteration penalty is 5.2x because it chases nodes and touches a cache line per element, but its
  random access is only about 6.5x worse than a sparse set and both are cheap in absolute terms.
- A sparse set wins both columns and costs a `4 * id_space` byte array, so it is only viable with
  bounded dense ids. That constraint, not the speed, is what decides it.

**Struct-of-arrays, measured honestly.** 1M entities, sweeping one field:

| | AoS, 64-byte row | AoS, 16-byte row | SoA |
| --- | --- | --- | --- |
| read one field | 0.95 ms | 0.371 ms | 0.372 ms |
| speedup against SoA | **2.55x** | **none** | |

**Quote 2.5x, not an order of magnitude.** AoS pulls 16x the bytes but saturates DRAM at ~63 GiB/s
while SoA becomes latency-bound on the dependent float chain, so the ratio is nothing like the
bandwidth ratio. At a 16-byte struct, SoA wins nothing at all. The 41x figure for a write sweep is
real but flattered by vectorisation, and quoting it will get you caught.

## Wiring, and the initialisation order fiasco

```cpp
struct Services { Logger* log; Telemetry* telemetry; CombatConfig cfg; };  // telemetry optional
// load(): pure data.  wire(): store the Services*.  start(): connect, spawn.
```

**The fiasco is not theoretical.** Two translation units, one global reading another's, gave `0x0`
with one link order and the correct value with the other. Link order alone flipped it, silently,
with no diagnostic.

- **The fix is a function-local static**, and it is correct under both link orders:
  `Registry& registry() { static Registry r; return r; }`. Thread-safe since C++11: concurrent
  callers block until initialisation completes.
- **The fix creates a new hazard**: destruction order. A function-local static destroyed before its
  last user is a use-after-free at shutdown. If something must outlive everything, leak it
  deliberately (`static Registry* r = new Registry;`) and say so in a comment.

## Guards

```cpp
[[nodiscard]] std::expected<float, Refusal> apply_hit(CombatRow& r, float raw) {
    assert(raw >= 0.f && "caller bug");                                  // programmer error
    if (raw <= 0.f) return 0.f;                                          // bad input, silent
    if (r.cooldown > 0.f) return std::unexpected(Refusal{"on cooldown"}); // policy refusal
    ...
}
```

- **`std::expected` for policy refusals** under C++23. Under C++20 use `std::optional` plus an
  out-parameter reason, or the codebase's existing `Result` type. Do not introduce a second error
  vocabulary beside one that already exists.
- **`[[nodiscard]]` on anything returning a result**, and on the result type itself. It fires under
  `-Wunused-result` and it is the cheapest guard in the language.
- **`assert` compiles to nothing under `NDEBUG`.** Verified: `assert((side = 1) == 1)` left `side`
  at 0 in a release build. **Never put an effect inside an assert**, and never use one to validate
  input that comes from outside the program.
- Missing configuration warns once and uses a default. It does not become an `expected`.

## Time

- **`steady_clock` is the only correct deadline clock.**
- **`high_resolution_clock` is `system_clock` on this toolchain**, verified with `is_same_v`, so it
  is **not steady** and jumps with NTP. It reads like the best clock in the standard library and it
  is the wrong one. This is the single most likely mistake in this section.
- Cost is not the tiebreaker: `steady_clock::now()` measured 16.7 ns and `system_clock::now()`
  16.8 ns, both through the vDSO. Still read it once per frame rather than per entity.

```cpp
struct Cooldown { std::chrono::steady_clock::time_point ready_at; };
if (now < cd.ready_at) return;                       // one comparison, no write
cd.ready_at = now + cfg.cooldown;                    // cfg.cooldown is std::chrono::milliseconds
```

## Concurrency

- **`std::jthread`** (available in C++20) joins in its destructor and carries a `stop_token`, which
  removes the two most common thread bugs: forgetting to join, and having no way to ask it to stop.
- **Cancel by identity as well as by token**, because a token only helps a thread that checks it:

```cpp
std::uint32_t launched_at = fsm.version;             // captured by value
std::jthread w([=, &fsm](std::stop_token st) {
    if (st.stop_requested()) return;
    if (fsm.version != launched_at) return;          // stale, the state moved on
    apply();
});
```

- **A data race is two unsynchronised accesses to the same location where at least one writes**, and
  it does not need to look wrong to be wrong. Verified: an unsynchronised `int` counter printed the
  **correct** total at `-O2` and was still a race. Tests passing tells you nothing here.
- **`-fsanitize=thread` caught it** (two warnings on the plain int, zero on the atomic). Two
  operational notes: TSan is **incompatible with ASan**, so it needs its own build, and it aborted
  with `unexpected memory mapping` on this kernel until run under `setarch -R` to disable ASLR.
- Only trivially copyable data crosses a thread boundary. A reference into a container that another
  thread can rehash is a race with extra steps.

## Control flow as data

Measured, and it does not say what this pattern is usually sold on:

| scenario | flat array of nodes | virtual `tick()` | speedup |
| --- | --- | --- | --- |
| 1 tree, 17 nodes, 20k agents | 19.7 ns/agent | 19.5 ns/agent | **0.97x, none** |
| 4000 distinct trees, 121 nodes | 0.728 ms | 0.93 ms | 1.28x |

**A small hot tree fits in L1 and the branch predictor learns the vtable targets, so the flat layout
wins nothing.** Choose flat for the reasons that actually hold: it serialises, it hot-reloads, and it
is one allocation instead of N. Claim a speed win only above roughly 100 nodes across thousands of
instances, and only with your own measurement.

For a state machine, one `fsm_goto()` door, a `static constexpr bool allowed[N][N]` transition
table, and a `version` counter bumped on every transition. That counter is what the identity re-check
above compares against.

## Config

A plain struct of values and `std::chrono` durations, zero methods, zero pointers, filled with
designated initialisers at the call site. Keep it aggregate-initialisable, which means keeping it
constructor-free.

## Tooling and validation

- `-Wall -Wextra -Wpedantic`, and treat the useful ones as errors. They caught `-Wuninitialized`,
  `-Wnarrowing`, `-Wmisleading-indentation` and `-Wunused-result` in ordinary code during this work.
- **`-fsanitize=address`** catches heap overflow, use-after-free, and leaks (LeakSanitizer is bundled
  and on by default). **`-fsanitize=undefined`** catches signed overflow and null dereference. Build
  with `-O1 -g -fno-omit-frame-pointer` so the traces are readable. All verified firing.
- **`-fsanitize=thread` is a separate build.** It cannot be combined with ASan.
- clang-tidy is worth enabling `bugprone-*`, `performance-*` and `cppcoreguidelines-*`, though that
  was not verified here.
- Say which of these actually ran. "It compiles" is not validation, and in a language where a real
  race prints the right answer, neither is a passing test.
