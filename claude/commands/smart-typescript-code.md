---
description: The TypeScript layer on top of /smart-javascript-code. satisfies and as const for config, discriminated unions with never-exhaustiveness as compile-checked control flow, branded ids with zero runtime cost, the strict flags worth adding beyond strict, Result unions for the guard ladder, and what tsc 7 removed and broke.
argument-hint: [the module, feature, or system to write, extend, or refactor]
---

Build: **$ARGUMENTS**

This is `/smart-javascript-code` plus the type layer. Everything there still applies: `Map` over
plain objects, `??` over `||`, `performance.now()` deadlines, `setImmediate` over recursive
microtasks, frozen dispatch tables. `/smart-simple-code` holds the reasoning behind all of it.

TypeScript's contribution to this style is specific: it turns three of the parent's conventions from
things you have to remember into things the compiler checks.

## Config: `satisfies`, not an annotation

```ts
const cfg = { retries: 3, mode: 'fast' } satisfies Cfg;   // checked, and keys stay literal
const cfg2: Cfg = { retries: 3, mode: 'fast' };           // checked, but mode widens to string
```

**`satisfies` validates the literal against the type while keeping the literal types.** The
annotation form widens `'fast'` to `string` and you lose every downstream narrowing. Use `satisfies`
for every config object.

**`as const` turns data into a type:**

```ts
const LEVELS = ['debug', 'info', 'warn'] as const;
type Level = typeof LEVELS[number];        // 'debug' | 'info' | 'warn'
```

One array is now both the runtime list you iterate and the union you check against, so adding an
entry cannot desynchronise them. This is the type-level version of the parent's "adding a variant is
a data entry and zero new code".

## Control flow as data, made exhaustive

```ts
type Node =
  | { readonly type: 'leaf'; readonly run: (c: Ctx) => boolean }
  | { readonly type: 'seq';  readonly kids: readonly Node[] }
  | { readonly type: 'sel';  readonly kids: readonly Node[] };

function tick(n: Node, c: Ctx): boolean {
  switch (n.type) {
    case 'leaf': return n.run(c);
    case 'seq':  return n.kids.every(k => tick(k, c));
    case 'sel':  return n.kids.some(k => tick(k, c));
    default: { const _never: never = n; return false; }   // the exhaustiveness guard
  }
}
```

**Verified in both directions.** Delete the `'sel'` case and tsc reports
`TS2322: Type '{ readonly type: "sel"; ... }' is not assignable to type 'never'`, naming the exact
member you forgot. Restore it and the file compiles clean. That `const _never: never` line is what
converts "we added a node type and forgot to handle it" from a production bug into a build failure,
and it costs nothing at runtime.

Use `readonly` on union members and on container fields (`readonly Node[]` for `kids`), so a tree
that is meant to be data cannot be mutated by the code walking it.

## Branded ids

```ts
declare const brand: unique symbol;
type Brand<T, B extends string> = T & { readonly [brand]: B };
type UserId = Brand<string, 'UserId'>;
type RoomId = Brand<string, 'RoomId'>;
```

**Zero runtime cost, and it catches the mixup.** Passing a `RoomId` where a `UserId` belongs gives
`TS2345`, naming both brands; a raw `string` gives the same error. Then `Map<UserId, State>` cannot
be keyed by the wrong id at all, which is the single highest-value type in an entity-store codebase
where every id is a string.

**Template literal types** do the same for event and channel names:

```ts
type Event = `${Entity}:${'created' | 'deleted'}`;   // rejects 'user:updated'
```

The error prints the full expanded union, so it doubles as documentation of what is allowed.

## The guard ladder as a type

```ts
type Result<T> = { ok: true; value: T } | { ok: false; reason: string };
```

- Policy refusals return a `Result`, and the discriminant means the caller cannot read `.value`
  without narrowing first.
- Programmer errors still `throw`, and errors keep `instanceof` and `cause` from the parent file.
- The two do not compete: results carry refusals, exceptions carry bugs.

## Compiler flags

`strict` turns on `noImplicitAny`, `strictNullChecks`, `strictFunctionTypes`, `strictBindCallApply`,
`strictPropertyInitialization`, `noImplicitThis`, `useUnknownInCatchVariables` and `alwaysStrict`.

**Two more are worth adding, and both directly serve this style:**

- **`noUncheckedIndexedAccess`** makes `arr[0]` into `number | undefined`. Verified firing with
  TS2322. Every lookup into an entity store is now forced through a guard, which is exactly the rule
  the parent file asks for by convention.
- **`exactOptionalPropertyTypes`** rejects `{ x: undefined }` for an optional `x?`. Verified with
  TS2375. **This is what makes an optional container dependency honest**: "absent" and "present but
  undefined" stop being the same thing, which is the distinction the whole nil-guard convention rests
  on.

## tsc 7, and what it broke

TypeScript 7 is the Go port. Type-checking semantics are intended to be identical, and across roughly
6,000 error-producing test cases only 74 differ. What changed is everything around the checker.

**Removed, as hard errors rather than deprecations** (TS5102/TS5108), all verified by running:

- `target: es5`
- `moduleResolution: node10`
- `esModuleInterop: false`
- `baseUrl`
- `outFile`

**`strict` is now on by default.** A bare `tsc file.ts` with no tsconfig reports
`TS7006: Parameter 'a' implicitly has an 'any' type`, which 5.x did not.

**Speed: quote 8 to 12x**, from real trees (VS Code's 1.5M lines went 77.8 s to 7.5 s; TypeORM
measured 13.5x; `--checkers 8` reaches 16.7x). A small corpus measures far lower because it is
startup-dominated, so do not quote a benchmark from a toy project.

**The gap that decides your tooling: TypeScript 7.0 ships no public compiler API.** It arrives in
7.1. The consequences are immediate and mutually exclusive:

- **typescript-eslint cannot support TS 7 yet**, because there is no stable JS API to build on.
- **oxlint's `--type-aware` mode requires TS 7.0 or newer.**

So the two type-aware linters sit on opposite sides of a version wall, and **you pick one**. Also
stranded until 7.1: custom transformers, `ts.transpileModule`, and the template tooling for Vue,
Svelte, Astro and Angular, since `tsserver` is replaced by LSP. TypeScript 6.0 is the bridge release
and the last of that line.

If you must choose: **oxlint type-aware measured 12 to 18x** faster than typescript-eslint on the
same rules (its own conservative figure, which a direct measurement of 145 ms against 2160 ms
matches; ignore the 20-40x in its README). Its engine, `tsgolint`, came out of the typescript-eslint
project and covers 59 of 61 type-aware rules. Biome's inference is its own, does not use tsc, and its
2026 roadmap does not prioritise expanding it, so treat its type-aware rules as partial.

## What not to do

- **`any` to make an error go away.** `unknown` plus a narrowing guard is the same amount of typing
  and actually checks something.
- **A type assertion where a guard belongs.** `as Foo` at a boundary is a lie the compiler will
  believe and the runtime will not.
- **Enums.** A union of string literals plus `as const` gives you the same checking, no runtime
  artifact, and no `const enum` inlining hazards.
- **Interfaces for data that is already a shape.** A `type` alias with a discriminant composes with
  unions; an interface does not.
- **A parallel type hierarchy that mirrors the runtime one.** If the type and the data can drift,
  they will. Derive the type from the data with `typeof` and `as const` instead.
