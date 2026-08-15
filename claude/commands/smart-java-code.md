---
description: The Java specialisation of /smart-simple-code. Records as the value type, sealed interfaces with exhaustive switch as control flow you can check at compile time, computeIfAbsent for lazy rows and its nondeterministic trap, nanoTime deadlines, virtual threads and what still pins them, and matching an existing DI framework instead of hand-rolling a second one.
argument-hint: [the module, feature, or system to write, extend, or refactor]
---

Build: **$ARGUMENTS**

`/smart-simple-code` holds the reasoning. This file is the Java answer, verified on JDK 21.

**On the claim that this style fights Java: it does not, on 21.** Java has no free functions, but it
has the two things the style actually needs. `record` declares a value type in one line, and
`sealed` plus `switch` declares a closed set the compiler checks exhaustively. A `final class` of
static methods over records is not a degenerate utility bag, it is the same construct with no
instance state, and `import static` restores unqualified call syntax at the call site. The one real
friction is genuine: **records are immutable, so a mutable per-entity row must be a small ordinary
class**, not a record.

## Module shape

```java
public record Health(int current, int max) {
    public Health {                                     // compact ctor validates and normalises
        if (max <= 0) throw new IllegalArgumentException("max");
        current = Math.clamp(current, 0, max);
    }
}

public final class Combat {                             // free functions, called via import static
    private Combat() {}
    public static boolean alive(CombatRow r) { ... }
    public static Outcome hit(CombatRow r, int raw) { ... }
}
```

- A record gives you the canonical constructor, accessors, value `equals`/`hashCode`/`toString`, and
  is implicitly final. Two equal records deduplicate in a `HashSet`.
- **The compact constructor validates and normalises before assignment**, so `new Health(150, 100)`
  becomes `Health[current=100, max=100]` rather than an invalid object that exists for a while.
- Records forbid instance fields (`field declaration must be static`) and extension. That is the
  point, not a limitation.
- **Trap: record immutability is shallow.** `record Bag(List<String> items)` allows
  `bag.items().add(...)` to succeed. Defensive-copy collections in the compact constructor or hand
  out an unmodifiable view.
- **The mutable row is an ordinary small class**, package-private, with public fields if the module
  owns it. Do not contort a record into holding mutable cells.

## Control flow as data, checked at compile time

```java
sealed interface Node permits Sequence, Selector, Invert, Leaf {}
record Sequence(List<Node> children) implements Node {}
record Invert(Node child)            implements Node {}
record Leaf(String name, Predicate<Blackboard> act) implements Node {}

static boolean tick(Node n, Blackboard bb) {            // the single door
    return switch (n) {
        case Sequence s -> s.children().stream().allMatch(c -> tick(c, bb));
        case Invert i   -> !tick(i.child(), bb);
        case Leaf(String name, var act) -> act.test(bb);   // record deconstruction
    };
}
```

- **Exhaustiveness is enforced.** Deleting a case gives `the switch expression does not cover all
  possible input values` at compile time. This is the strongest version of "control flow as data"
  available in any of these languages.
- **Never write `default` in a switch over a sealed type.** It silently disables exhaustiveness
  forever, so adding a new node type stops being a compile error and starts being a runtime surprise.
  That compile error is the entire value of the pattern.
- `permits` is compile-time only, so adding a subtype breaks every switch over it. That is working as
  intended.

## The entity store

- **`ConcurrentHashMap.computeIfAbsent` is the lazy row.** Verified: 200 threads over a 16-wide pool
  called the factory exactly once and produced one value.
- A factory returning `null` stores no entry, which is a quiet way to get a repeated recompute.
- **The documented trap is worse than documented.** Touching the same map inside its own mapping
  function throws `IllegalStateException: Recursive update` when the keys land in the same bin, and
  **silently succeeds when they land in different bins**. So it is hash-dependent and
  nondeterministic: it will pass every test and fail in production. Never touch the map inside its
  own mapping function. On a plain `HashMap` the same shape throws `ConcurrentModificationException`.
- `WeakHashMap` is the weak-keyed store, verified collecting an unheld key after `System.gc()`.
  **Trap: if the value strongly references the key, the entry never clears** and you have built a
  leak with extra steps.

## Wiring

- **If the project already uses Spring, Guice or Dagger, that framework is the container.** Match it:
  `@Configuration` is the wire phase, `@PostConstruct` or an `ApplicationRunner` is the start phase.
  Do not hand-roll a second container beside it.
- Spring detects a constructor-injection cycle at `refresh()` with `UnsatisfiedDependencyException`
  rooted in `BeanCurrentlyInCreationException`. **That is startup-time, not compile-time**, so a cycle
  in a rarely-loaded profile can still reach production.
- `@Lazy` at the injection point starts the context and injects a CGLIB proxy, constructing the real
  bean on first call. That is the framework's own version of "resolve late". **Trap: it cannot proxy a
  record or any final class**, since it needs to subclass.
- In a plain `main` project, the container is an explicit `record Deps(...)` passed once.

## Guards

- The ladder maps to: silent `return`; a returned `sealed interface Outcome permits Ok, Refused`; a
  logged warning plus a default; and `throw` or `Objects.requireNonNull(x, "x")`.
- **`assert` is disabled by default** (`desiredAssertionStatus()` is false; it fires only under
  `-ea`), so it is unusable as a production guard. Use `requireNonNull` or an explicit throw.
- Helpful NullPointerException messages are on by default and name the expression that was null, so a
  plain NPE is more informative than most hand-written checks.
- **`Optional` is a return type only.** Verified mechanically: it is not `Serializable`, and an
  `Optional` parameter can itself be null, so it buys nothing at a boundary it was supposed to
  protect.

## Time

- **`System.nanoTime()` for every deadline.** It is monotonic within a JVM.
  `System.currentTimeMillis()`, `Instant.now()` and `Clock.systemUTC()` are wall clock and can jump.
- **There is no monotonic `Clock` in the JDK.** For injectable time use a `LongSupplier now =
  System::nanoTime` rather than reaching for `Clock` and getting wall time by accident.
- **Compare by subtraction, never by ordering**: `while (System.nanoTime() - deadline < 0)`. The
  subtraction is wraparound-safe and the direct comparison is not.
- Measured cost per call: `nanoTime()` 16.4 to 16.9 ns, `currentTimeMillis()` 17.1 to 17.3 ns,
  `Instant.now()` 22.4 to 23.5 ns. Cheap, but not free enough for a per-entity inner loop. Read once
  per frame and pass it down.

## Concurrency

- **Virtual threads are for blocking work, not for compute.** Verified with a single carrier: a
  CPU-bound virtual thread is **not** unmounted, and a second task did not start until 1 ms after the
  first finished 1.5 s later. There is no preemption.
- Blocking does unmount properly: 10,000 virtual threads each sleeping a second finished in 1110 ms
  on one carrier.
- **Pinning on 21 is real and `synchronized` is the cause.** Four virtual threads blocking inside a
  `synchronized` block took 2003 ms, serialised; the same code with a `ReentrantLock` took 501 ms.
  Diagnose with `-Djdk.tracePinnedThreads=full`, which prints `reason:MONITOR` and the stack. JEP 491
  removes monitor pinning in JDK 24 and later, so on 21 you must replace `synchronized` around
  blocking calls with `ReentrantLock`.
- Cost: 100k virtual tasks in 127 ms, about 1.27 us each, against 10k platform threads at 56.7 us
  each, roughly 45x.
- **Structured concurrency is still preview on 21** and needs `--enable-preview` at compile and run.
  Do not ship it on 21. Use `Executors.newVirtualThreadPerTaskExecutor()`, which is `AutoCloseable`
  and joins on close, `ScheduledExecutorService` for delay, and `CompletableFuture` for chaining.
- Only immutable data crosses the boundary. Pass records, not mutable rows.

## Config

Nested records plus a `static Cfg load(Properties)`. Validation lives in the compact constructors, so
a bad value fails at load with a real message rather than at first use: a malformed number threw
`NumberFormatException: For input string: "five"` during `load`, not three screens later. A
properties file plus record parsing beats a configuration library until you genuinely need profiles
or hot reload.

## Tooling and validation

- **`javac -Xlint:all -Werror`** is the free baseline and catches `rawtypes`, `unchecked`,
  `fallthrough` and more across 34 lint keys. **It has no null analysis and no unused-local check**,
  which is exactly the gap the next tool fills.
- **NullAway** for null analysis, **Error Prone** as the host for it, **SpotBugs** for the bytecode
  pass they miss.
- **JUnit Jupiter is on version 6.** Saying "JUnit 5" as the current answer dates you and may pick
  the wrong artifact coordinates.
- **jqwik** for property tests, which pay off precisely on the record-plus-pure-function layer this
  style produces.
- Say which ran. A compile is not a test, and on 21 a `synchronized` block that quietly serialises
  your virtual threads will pass every test you have.
