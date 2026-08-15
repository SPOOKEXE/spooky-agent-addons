---
description: Write and extend code in a struct-first style. Plain-data modules with free functions instead of class hierarchies, one container injected once and resolved lazily, guard ladders at the top of every function, deadline timestamps compared once instead of ticking timers, work that never blocks the loop, and control flow shaped as data so it can be edited without rewriting. Language-agnostic.
argument-hint: [the module, feature, or system to write, extend, or refactor]
---

Build: **$ARGUMENTS**

With no argument, ask what is being built rather than guessing.

The through-line: **the cheapest code to change is code with no hidden state and no hidden control
flow.** Every rule below is a way of keeping state printable and control flow visible. None of them
is about cleverness, and several trade a small amount of purity for a large amount of legibility.

Not this:

- a class where a table of data and three free functions would do
- a constructor taking eight dependencies so that a test can pass fakes for all eight
- three levels of nesting where a guard clause at the top would do
- a timer decremented every frame when a deadline compared once would do
- blocking the loop, the frame, or the request to wait for something
- a second registry standing parallel to one that already exists
- wrapping a module when replacing it would be simpler and safer
- a visual scripting layer, a DSL, or a plugin system nobody asked for

**Language files.** This file is the reasoning. The language-specific answers, each verified
against that language's toolchain, live in `/smart-luau-code` (and `/smart-roblox-code` on top of
it), `/smart-python-code`, `/smart-javascript-code` (and `/smart-typescript-code` on top of it),
`/smart-cplusplus-code`, and `/smart-java-code`. Use the language file when there is one, and this
file when there is not.

## Step 0. The codebase's answer beats this file

Before writing anything, find how this codebase already does the six things below, and match it.
These rules are defaults for new code and for a codebase with no answer of its own. They are never
a reason to convert code that works.

Look for: the module skeleton, how a new system gets registered, how systems reach each other, how
failures are reported, how time is read, and how per-frame or background work is scheduled.

Where the codebase is inconsistent, **follow the newest code and say so in the report**. Large
codebases usually contain two or three generations of style, and the honest move is to name which
one you matched rather than to average them.

## Step 1. Modules are data plus free functions

- A module is a **namespace of functions plus module-level state**, returned as one value. Not a
  class, not an instance, not a hierarchy.
- **Classes earn their place only for genuine value types**: a signal, a cleanup handle, a spatial
  index, a queue. Something you make many of and pass around. In a 98k-line codebase built this
  way, 11 files out of roughly 600 use a class, and all of them are value types.
- **State lives in a keyed table owned by the module**, not in `self`. One row per entity, keyed by
  the entity, created lazily on first touch. There is no registration step, so there is no
  registration bug and no lifecycle to keep in sync.
- Where the language allows it, make that table **weak-keyed** so a row dies with its owner. Where
  it does not, delete the row in the same place you destroy the owner, and nowhere else.
- **Types at the boundary**: export the shapes that cross a module boundary, keep row and internal
  shapes private. The exported type list is the module's real API.
- A function that takes the entity and returns its row (`getRow(entity)`) is the whole accessor
  layer. Do not build more than that.

The test of the shape: **you can print the module's entire state with one loop**, and a reader can
tell what a system knows without running it.

Where this fits the language and where it fights it:

- **Already idiomatic**: Go (struct plus methods, no inheritance), Rust (`struct` plus `impl`), C,
  Zig. You are writing normal code.
- **Idiomatic with care**: TypeScript (an interface plus free functions tree-shakes; a class is
  needed for `instanceof`, decorators, and most DI frameworks), Python (`@dataclass` plus module
  functions for records, though it fights back once you want protocols or operator overloads).
- **Needs translating, but not fighting**: Java and C# have no free functions, yet both have the two
  constructs the style actually needs. A record declares a value type in one line, and a sealed
  hierarchy with an exhaustive switch declares a closed set the compiler checks. A final class of
  static methods plus `import static` restores unqualified calls at the call site. The genuine
  friction is that records are immutable, so a mutable per-entity row has to be an ordinary small
  class. Verified on Java 21; see `/smart-java-code`.
- **Splitting a record into parallel arrays** (struct-of-arrays) pays only when you sweep one field
  across many entities. When a single entity's fields are read together, which is the normal case,
  splitting them is a pessimisation. Do not reach for it before you have a measurement.

## Step 2. Inject one container, resolve late, guard for absence

The pattern that survives contact with a large codebase is neither constructor injection per
dependency nor a global service locator. It is:

1. **Discovery, not registration.** A loader requires every module in a folder and caches it.
   Adding a system is adding a file. There is no central list to edit and therefore no merge
   conflict on it, and duplicate names fail loudly at boot.
2. **One container injected once**, at wire time: each module receives its siblings as a single
   namespace and stores it. Not eight parameters. One.
3. **Members resolved at call time**, through that container, and **optional ones nil-guarded** at
   the point of use.
4. **Dependencies from outside the container are required once at the top of the file**, never
   inside a function body.

Why this rather than the textbook answer, stated honestly:

- Per-dependency constructor injection makes every signature churn when wiring changes, and in
  languages with eager module loading it walks straight into circular-import deadlock.
- A global locator has the same late-binding benefit but no scope, so anything can reach anything,
  and the dependency graph stops existing.
- The container is the middle: late-bound like a locator, but **scoped to siblings**, so the reach
  of any module is visible from where it sits in the tree.
- The cost you accept, and it is a real one: a test must build a container rather than pass three
  fakes, and a missing dependency becomes a nil at call time rather than an error at build time.
- **The cost nobody mentions: late binding hides circular dependencies.** With eager imports, a
  cycle fails loudly at load, in the composition root, where it is obvious. Resolve lazily instead
  and the cycle stops erroring and becomes an invisible runtime coupling that surfaces as a nil at
  3am. So: keep the guard, and when two systems reach for each other, treat it as a design smell to
  fix by extracting a third leaf module rather than as a problem the container solved for you.

**Three phases, and the rule that makes them work:**

| phase | what happens | what is forbidden |
| --- | --- | --- |
| load | module body runs, config captured, channels acquired | any cross-system access |
| wire | receive the container, store it | any cross-system *call* |
| start | connect handlers, spawn loops, do first work | nothing |

**Wiring order is almost always nondeterministic** (a loader iterating a hash map, a plugin scan, a
directory walk). So a call to another system during wiring is a coin flip that lands right in
development and wrong in production. Everything ordered goes in start. In a hot function, re-bind
the container member to a local at the top rather than resolving it inside a loop.

## Step 3. Guards, and the four-way ladder for failure

The top of a function is the list of reasons not to run it. Early return, one line each, no `else`.

| situation | response | why |
| --- | --- | --- |
| bad input, missing object, not-ready state | return silently, nil or false | it is a normal state, not an event |
| policy refusal the caller may want to explain | return `(false, reason)` | the caller decides whether to surface it |
| missing content or configuration | warn, keep running | designers iterate by reading warnings |
| programmer or authoring error | raise, and prefer at boot | it can never be correct, so fail loudly |

- **Predicates return `(ok, reason)` pairs and stay pure.** That is what lets both sides of a trust
  boundary call the same function: the client gates its UI on it, the server re-checks with it, and
  there is exactly one definition of what is allowed.
- **Count refusals, do not print them.** A per-reason counter on the row (`row.denials[reason] += 1`)
  gives you a distribution you can read at any time, where a log line gives you noise in production
  and nothing at all in aggregate. Print only what a human will act on.
- The failure mode of this style, worth stating so it is guarded against: **a silent return that
  should have been an error** hides a bug for months. The caller sees "nothing happened" and cannot
  distinguish it from "nothing needed to happen".
- Two guards that look the same but are not: "this cannot happen" is a raise, "this happens
  constantly and is fine" is a return.
- **Guards do not reduce cyclomatic complexity.** The decision count is unchanged, so anyone
  arguing for guards on McCabe grounds is arguing wrongly and will be caught out. What drops is
  nesting depth, which is what *cognitive* complexity measures and what actually predicts whether a
  human can hold the function in their head. Cite that one.
- Watch multi-exit functions that own cleanup: an early return that skips teardown leaks. Use the
  language's scope-exit tool rather than remembering.
- Language support worth using where it exists: Swift `guard ... else`, which the compiler checks
  actually exits; Rust `let ... else` and `?`; Go's `if err != nil { return }`. In Python, raise for
  a caller bug and **never use `assert` to guard untrusted input**, because assertions are stripped
  under `-O` and the guard silently disappears in production.

## Step 4. Time is a deadline you compare once

Store the **absolute time the thing becomes allowed**, not the remaining time. The check is one
comparison and no write:

```
if now() < row.readyAt then return end          -- the whole cooldown system
row.readyAt = now() + config.COOLDOWN
```

Measured over 1M entities per frame, comparing three shapes of the same logic:

| shape | cost per frame | why |
| --- | --- | --- |
| decrement a remaining-time counter, then test | 139 us | a read and a **write** per entity per frame |
| compare against a stored deadline | 65 us | read only, 2.1x cheaper |
| deadline, kept sorted, binary search the due prefix | 0.9 us | **155x**: you never touch what is not due |

The third row is the real argument. You cannot skip work you have to decrement, but you can skip
everything whose deadline has not arrived.

There is a fourth shape, scheduling a callback per cooldown, and it is worse than it looks: one
scheduler entry and one closure allocated per cooldown, so 10k entities re-arming is 10k
allocations a second, and cancellation becomes your problem. Reserve it for a small number of long
one-shot events.

**Be precise about drift, because two different things get called that.** Floating-point
accumulation is **not** an argument: adding `1/60` ten thousand times drifts 2.5e-11 s in double
precision, and 3.4 ms, a fifth of one frame, in single. But decrementing by a **nominal** delta while
real time passes is a real argument, and larger than people expect: a thousand ticks decrementing by
the nominal step finished **9.9% late** against the wall clock, where the deadline comparison
finished exactly on time. A counter tracks your loop's idea of time. A deadline tracks the clock.

Conventions worth copying:

- **Name the field for what it is**: `readyAt`, `expiresAt`, `cooldownUntil` for deadlines;
  `lastFiredAt` for the other direction. A name ending in `Until` or `At` is a timestamp, and a name
  ending in `Remaining` is a bug waiting for a pause to happen.
- **Read the clock once per frame or tick**, pass it down. Every system reading its own clock is how
  two systems disagree about when now is.
- **Use a monotonic clock for durations.** A wall clock steps backwards on clock sync, DST, a leap
  second, or a user setting the time. Backwards and the deadline **never expires**; forwards and
  **every cooldown fires at once**. Cloudflare's 2016 leap-second outage is the canonical incident.

| language | monotonic, use this | wall clock, can jump |
| --- | --- | --- |
| Python | `time.monotonic()`, `time.perf_counter()` | `time.time()`, `datetime.now()` |
| JS/TS | `performance.now()` | `Date.now()` |
| Go | `time.Since(start)`, `time.Until(deadline)` | any `Time` after `Round(0)`, `AddDate`, `UTC()`, which strip the monotonic reading |
| Rust | `Instant` | `SystemTime` |
| C++ | `std::chrono::steady_clock` | `std::chrono::system_clock` |
| Java | `System.nanoTime()` | `System.currentTimeMillis()` |
| C# | `Stopwatch`, `Environment.TickCount64` | `DateTime.Now` |
| Luau/Roblox | `os.clock()`, and `workspace:GetServerTimeNow()` when client and server must agree | `os.time()`, and `tick()`, which is not monotonic |

Note the Go row: a `Time` carries a monotonic reading that several ordinary-looking operations
silently strip, which turns a correct deadline into a wall-clock one without a warning.
- **Where two processes must agree**, both read the same shared, synchronised value and store the
  deadline where both can see it. Then neither has to trust the other's clock.
- Accumulators are for **fixed-step simulation**, where the point is determinism rather than timing:
  add the delta, run the step while the accumulator exceeds the step size, keep the remainder.

## Step 5. Work that does not block

Three verbs, and they are not interchangeable:

- **spawn**: fan out work now, without waiting for it. Use it to do something per entity or per
  player from inside a loop that must not stall. **It does not make anything asynchronous.** In
  most schedulers it resumes immediately and reentrantly, so it runs on the same frame and merely
  hides the stack, and your caller's invariants must already hold at the call.
- **defer**: run after the current unit of work finishes. Use it for anything that must not observe
  a half-wired system, which is why start is usually deferred.
- **delay**: run later. Always paired with a **cancellation check on identity**, because the world
  will have changed by the time it fires:

```
local queued = row.bufferedInput
delay(remaining, function()
    if row.bufferedInput ~= queued then return end   -- superseded, do nothing
    row.bufferedInput = nil
    act(queued)
end)
```

That identity re-check is the whole trick. It replaces a cancellation handle, a timer registry, and
the bookkeeping that goes with both, with one comparison that cannot get out of sync.

- **Per-frame work is a subscription owned by a cleanup handle**, destroyed with the thing that
  created it. An unowned per-frame connection is a leak that survives the object it was watching.
- **Only plain data crosses a thread or task boundary.** No handles, no live objects, no closures
  over mutable state. If a value needs a lock to be read safely, it should not be crossing.
- Prefer one long-lived worker draining a queue over spawning per item, once the rate is high enough
  that spawn cost is visible. Below that, spawning per item is simpler and simpler wins.

**Know which kind of concurrency you have**, because "do not block" means different things:

| runtime | scheduling | what blocks everything |
| --- | --- | --- |
| Go goroutines | preemptive since 1.14 | almost nothing; a tight loop no longer stalls the scheduler |
| OS threads (C++, C#, Java platform) | preemptive | nothing, but you own the data races |
| Rust `tokio::spawn` | cooperative | a task that never awaits holds its worker |
| Python `asyncio` | cooperative | any sync call on the loop thread |
| Java virtual threads | unmounted at blocking points only | a CPU-bound virtual thread is not unmounted |
| JS event loop | cooperative | a long task, and an unbounded `queueMicrotask` chain starves rendering |
| Game/frame schedulers | cooperative | anything over the frame budget, 16.6 ms at 60fps |

The frame, the event loop, and the request thread are the same object under three names. Never
occupy one for longer than its budget: chunk the work with a yield, or move it to a worker, a
thread pool, or the runtime's blocking pool.

## Step 6. Control flow shaped as data

When behaviour needs to be edited often, make it data that a small fixed interpreter walks. A
behaviour tree is the usual shape:

- **A node is a plain record**: a type tag, its arguments, and its children. Built by a factory, not
  by subclassing.
- **Every node function has the same signature** and returns success or failure. Uniformity is what
  makes nodes composable and what makes a new node cheap.
- **Compose as nested constructor calls**, condition first, so the shape of the tree is the shape of
  the code and reads top-down as "if alive, if not stunned, if in range, attack".
- **Shared node functions live in one place** and get re-exported per entity type, so a fix lands
  once.

When a tree is the wrong tool, and this matters more than the tree itself:

- **Strict priority with interruption** wants a state machine, not a tree. Give it explicit states, a
  priority per state, an expiry per state, and **a single mutation door** that every transition goes
  through:
  `if row.stateExpiresAt > now and incomingPriority < row.statePriority then return false end`
  One function that can change state is worth more than any diagram.
- **Two or three states** want an `if`. A tree here is ceremony.
- **Long-horizon planning** wants a planner (GOAP, HTN), which buys a wider problem domain at
  matching computational and authoring cost. You probably do not need one.

Known weaknesses, so the tree does not quietly become the problem:

- **Re-ticking the whole tree from the root every frame is wasted work.** Tick on an event or at a
  fixed lower rate once the tree is more than small.
- **A deeply nested tree is as unreadable as the state machine you fled.** Depth is the thing to
  watch, not node count.
- **Shared state ends up in an untyped blackboard.** Give it a declared shape like any other module
  state, or it becomes a global with extra steps.
- Worth using rather than writing: BehaviorTree.CPP (actively maintained, the ROS2 standard),
  py_trees for Python, and for Luau the maintained fork of BehaviorTree3 rather than the original,
  which is dormant. Check maintenance before adopting any of them.

## Step 7. Data and logic, and the test that proves the split worked

- Configuration is **pure data with no logic**: tuning tables, definition registries, content. One
  comment per key explaining what it does.
- **Behaviour reaches into config; config never reaches back.** No requires, no callbacks, no
  imports in a data file.
- **Read with a fallback** (`config.WINDOW or 0.18`) so a missing key degrades instead of crashing,
  and so adding a key does not require touching every existing entry.
- **One registry per concept.** A special case gets a field in the existing registry, never a second
  parallel registry beside it. Two registries for one concept is the most reliable way to get two
  behaviours that were supposed to be one.
- **The test of the split**: adding a variant is adding a data entry and writing zero new code. If
  adding the fourth of something still requires new functions, the split failed and the fourth one
  is telling you where.
- **The failure mode of data-driven design is data growing a programming language inside it.** The
  moment configuration needs conditionals, loops, or references to other entries, it has become
  code in a worse notation. Move it back into code and keep the data flat.

Two neighbouring patterns and the honest reason to skip them most of the time:

- **An event bus** decouples a genuine one-to-many fan-out, and costs you the call graph: nothing
  tells you who handles an event, ordering is implicit, and one throwing handler can strand the
  rest. Use it for real fan-out, use a direct call for one-to-one.
- **An entity-component system** is the productionised form of struct-of-arrays and a whole-
  architecture commitment. For fifty entities it buys query overhead and a debugger that no longer
  shows you an object.

## Step 8. Build the lever with the feature

The fastest iteration loop in a codebase is usually the one that already exists and gets used least.
Find it before building anything new.

- **A command or script that exercises the feature directly**, without playing through to it, is
  worth writing at the same time as the feature and not afterwards. In the codebase this style comes
  from, that surface is 148 command files, and adding one is the single cheapest action available.
- **A boot-time audit** that walks every registry and warns about authored-but-missing entries. It
  costs one function and catches an entire class of content bug before anything runs.
- **Flags to re-trigger one-shot flows**, with a reset, so a first-time path can be tested more than
  once per account.
- **A live view of state** you can inspect while running beats adding print statements to find the
  same thing twice.

## Step 9. Validating where there is no test suite

Most codebases in this style have thin or no automated tests, and pretending otherwise produces a
report that claims a safety that was never there. Say what you actually did.

- Find out what the real loop is: sync or build, run, exercise the feature through the dev lever,
  read the warnings. If that is the loop, use it and describe it in those words.
- **Type checking is usually the cheapest real signal available.** Check whether it is on for the
  files you touched, because in a large codebase strict mode is often enabled on a small minority of
  files and none of them are the interesting ones.
- Run the linter that is configured, not the one you would have configured.
- Where a change is genuinely risky and there is no way to test it, say that plainly, name the risk,
  and suggest the smallest lever that would catch it next time.
- **Never report a change as verified because it type-checks.** Say which of these happened: it ran,
  it was exercised, it was read.

## Step 10. Report

- which generation of the codebase's style you matched, where the codebase disagreed with itself
- what you added against what you reused, and any registry, channel or config key you introduced
- every guard you added, and which rung of the ladder it sits on
- what you did not do, and why: the wrap you declined in favour of a replacement, the abstraction
  you left unbuilt, the second registry you did not create
- how it was validated, in the words of what actually ran

Two rules worth ending on, both from a codebase that has survived this style at scale:

- **If replacing a module is simpler and safer than wrapping it, replace it.** A wrapper around
  something you do not trust leaves two things to understand instead of one.
- **One runtime per concept.** Not a player version, an NPC version, and a boss version of the same
  system, which is three places for the same bug to be fixed twice and missed once.
- **Prefer the version whose failure mode is a loud error over the version whose failure mode is
  silence.** That single test decides most of the choices above, and it is the one to fall back on
  when this file does not cover the case in front of you.
