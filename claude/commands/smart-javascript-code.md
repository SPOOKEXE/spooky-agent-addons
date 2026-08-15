---
description: The JavaScript specialisation of /smart-simple-code. ESM named exports that actually tree-shake, Map versus object versus WeakMap for entity stores, ESM cycles and the temporal dead zone, the nullish-versus-falsy guard trap, monotonic deadlines and clock clamping, the real event-loop ordering on Node 24, and workers and transferables.
argument-hint: [the module, feature, or system to write, extend, or refactor]
---

Build: **$ARGUMENTS**

`/smart-simple-code` holds the reasoning. This file is the JavaScript answer, measured on Node 24.15
with V8 13.6. For TypeScript, use `/smart-typescript-code`, which is this file plus the type layer.

## Module shape

**ESM named exports of free functions, plus a frozen data object.** The reason is tree-shaking, and
it is sharper than expected:

- `export function unusedA() {}` is **dropped** by a minifying bundler.
- The identical function as a member of `export const api = { ... }`, or as a class method, is
  **kept**. Grouping functions into an exported object defeats dead-code elimination.
- `import * as ns` still shakes while access is static. **One dynamic `ns[key]()` and every dead
  function comes back.**

**`Object.freeze` for config, and it is free.** Reads measured 0.55 ns frozen against 0.69 ns
unfrozen, so the frozen object is if anything faster. It is **shallow**, so a nested write succeeds
straight through it; a three-line recursive `deepFreeze` fixes that, and assignment then throws a
`TypeError` because ESM is always strict.

**Classes are genuinely right for identity plus lifecycle, and they are also the fastest dispatch**:
4.45 ns per call for a class method, against 16.08 ns for a closure-built object and 20.80 ns for a
module-level `Map` lookup. So the choice is not performance, it is tree-shaking and testability
against dispatch cost, and for most modules the exported functions win.

- **Trap: a detached method loses `this`.** `const g = c.get; g()` throws a `TypeError`. Free
  functions never have this problem, which is half of why they are the default here.
- `#private` fields give real encapsulation, and `#field in obj` is a working brand check.

## The entity store

```js
const store = new Map();                      // keyed by the entity object
function row(entity) {
    let r = store.get(entity);
    if (r === undefined) { r = { readyAt: 0, hp: 100 }; store.set(entity, r); }
    return r;
}
```

- **Use `Map`, even though a plain object reads faster.** At 100k string keys, `obj[k]` measured
  8.0 to 9.3 ns against `Map.get` at 13.4 to 14.7 ns, about 1.7x. But `Map.set` inserts about 2x
  faster (43 ns against 90 ns), and the correctness argument settles it: **plain objects coerce keys
  to strings** (`{}[1]` becomes `'1'`), inherit `toString` and `__proto__`, and destroy any non-string
  key you hand them.
- **`WeakMap` keys**, verified: objects, arrays and functions work. A **unique symbol works**, and so
  does `Symbol.iterator`. **`Symbol.for('s')` throws** `Invalid value used as weak map key`, because
  registered symbols are explicitly excluded, which is the one that will surprise you. Strings,
  numbers and null throw.
- **The lifetime trade-off, measured.** A strong `Map` of 50k entities retained 16.0 MB after garbage
  collection; the same data in a `WeakMap` retained 2.1 MB. But **a `WeakMap` is not iterable and has
  no `.size`**, so you cannot tick it. The rule that falls out: `WeakMap` for derived data you only
  ever look up, `Map` plus an explicit `delete` on despawn for anything you iterate.

## Imports and cycles

- **ESM cycles half-work, which is the dangerous kind.** Verified: `function` declarations hoist and
  are callable across the cycle, while `const` and `let` hit the temporal dead zone with a real
  `ReferenceError: Cannot access 'aValue' before initialization`. Reading the same binding after
  evaluation completes works fine.
- **CommonJS is worse, because it never throws.** The same cycle hands back a partial `exports` where
  `typeof a.aFn === 'undefined'`, plus a warning about accessing a non-existent property inside a
  circular dependency. It fails later and further away.
- **The fix that also serves as late binding**: `await import('./other.mjs')` inside the function that
  needs it. The cycle disappears because the import no longer happens at evaluation time.

## Guards

```js
if (!entity) return null;                                  // bad input, silent
if (now < r.readyAt) return { ok: false, reason: 'cooldown' };  // policy refusal
if (cfg.rate == null) console.warn('missing rate, using 3');    // missing config
if (!(kind in TABLE)) throw new TypeError(`unknown kind ${kind}`); // programmer error
```

- **`||` versus `??` is the trap that eats a zero.** Verified: `cfg.retries || 3` returns **3** when
  `retries` is `0`, where `cfg.retries ?? 3` returns `0`. Same for `''` and `false`. `||=` clobbers a
  zero and `??=` keeps it. **Use `??` for every default that could legitimately be falsy.**
- `a || b ?? c` is a **SyntaxError**. The parentheses are mandatory, which is the language stopping
  you from writing something ambiguous.
- `({}).nope?.()` returns `undefined` rather than throwing, so optional call is a guard in itself.
- `class ConfigError extends Error` works with `instanceof` at both levels, and
  `new Error(msg, { cause })` preserves the original, so wrap without losing the trail.
- **`node:assert/strict` is never stripped in production.** There is no `NDEBUG` here, so an assert
  is a permanent runtime cost and a permanent crash. Assert in tests; in library code
  `throw new TypeError(...)` so the message is deliberate.

## Time

```js
const deadline = performance.now() + ms;      // computed once
if (performance.now() < deadline) return;     // compared per iteration
```

- **`performance.now()` is monotonic**, verified non-decreasing over 500k samples, and is the deadline
  clock. `Date.now()` is wall clock and jumps backwards on NTP correction, so it belongs in logs.
- Resolution in Node: `performance.now()` about 30 us minimum non-zero delta and unclamped,
  `Date.now()` 1 ms, `process.hrtime.bigint()` 30 ns.
- Per-call cost: `performance.now()` 31.6 ns, `Date.now()` 30.5 ns, `hrtime.bigint()` 35.3 ns,
  `new Date().getTime()` 43.1 ns. **None is cheap enough for an inner loop.** Sample once and pass it
  down rather than polling.

**In a browser the clock is deliberately coarsened, and not uniformly:**

| engine | not cross-origin isolated | isolated | jitter |
| --- | --- | --- | --- |
| Chrome 91+ | 100 us | 5 us | yes, randomised |
| Firefox | 1 ms (16.667 ms under resistFingerprinting) | 20 us | yes, deterministic |
| Safari | 1 ms | 20 us | no |
| Node 24 | unclamped, ~30 us observed | | no |

MDN quotes Chrome's numbers as if they were universal. They are not: **Firefox's unisolated floor is
1 ms, 33x coarser**. So in a browser, measure a span across many iterations and never trust a single
call.

## The event loop

Verified by printing, and it is not the ordering most people recite:

```
inside any callback, and in CJS:  sync -> nextTick -> promise.then -> queueMicrotask -> setImmediate -> setTimeout(0)
at ESM top level:                 sync -> promise.then -> queueMicrotask -> nextTick -> setImmediate -> setTimeout(0)
```

**`process.nextTick` does not win at ESM top level**, because module evaluation is already a job.
Bun orders differently again. Treat all of this as a diagnostic for reading a trace, never as an API
to depend on.

- **Microtasks starve timers, measured.** A 500k-deep `queueMicrotask` chain ran to completion before
  a `setTimeout(..., 0)` fired at all: the timer callback finally saw 500,000 ticks after 16 ms of
  blocking. The same work through `setImmediate` let the timer fire at 0.4 ms, after 156 iterations.
  **Chunk long work with `setImmediate`, never with a recursive microtask.**
- **`AbortController` is the cancellation primitive**, with `AbortSignal.timeout(ms)` and
  `AbortSignal.any([...])` both available, and `signal.reason` carrying your own error. Keep the
  identity re-check as well (`const my = ++gen; ... if (my !== gen) return;`), since it covers the
  synchronous case a signal cannot.

## Workers and the boundary

- **Only structured-cloneable data crosses.** Functions throw a `DOMException`; `Map`, `Set` and typed
  arrays clone fine.
- **Spawning a worker costs 13.9 ms**, so never do it per request. Pool them.
- 1 MB through `postMessage` measured 0.148 ms copied against 0.121 ms transferred, and the transfer
  **detaches the source** (`byteLength` becomes 0). A `SharedArrayBuffer` write measured
  0.000051 ms, which is the real reason to use one.
- `SharedArrayBuffer` needs COOP and COEP headers in every engine; Chrome's grandfather exemption
  ended in 92 and the deprecation trial expired at 124. Chromium's newer `Document-Isolation-Policy`
  grants isolation per-frame without COOP or COEP, but Safari has filed a negative position, so ship
  COOP plus COEP and layer the newer header on top rather than depending on it.
- `Atomics.wait` is permitted on Node's main thread and blocks the loop completely. That is almost
  never what you want.

## Control flow as data

```js
const NODES = Object.freeze({
    leaf: (n, c) => n.run(c),
    seq:  (n, c) => n.kids.every(k => tick(k, c)),
    sel:  (n, c) => n.kids.some(k => tick(k, c)),
});
function tick(node, ctx) {
    const fn = NODES[node.type];
    if (!fn) throw new TypeError(`unknown node type ${node.type}`);   // programmer error
    return fn(node, ctx);
}
```

A state machine is a frozen transition table plus one `send` door, and a missing transition is a
**policy refusal rather than a throw**: `TRANSITIONS[state]?.[event]` returning `undefined` becomes
`{ ok: false, reason: 'no start from running' }`.

## Tooling and validation

- **ESLint 9 reached end of life in August 2026.** Version 10 is flat-config only, with `eslintrc`
  fully removed. Check which the project is on before adding a rule.
- **oxlint is roughly 44x faster** than ESLint on the same corpus in a direct measurement, which
  matches its own claimed range. Biome is the other fast option, with its own type inference that
  does not use tsc.
- `node:test` is stable, but its **coverage and watch mode are still Stability 1**, so
  `--experimental-test-coverage` is still required. Vitest remains the fuller option.
- Say which ran. In a language where a real cycle silently hands back a partial module, a passing
  test on the happy path proves less than usual.
