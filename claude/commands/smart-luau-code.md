---
description: The Luau specialisation of /smart-simple-code. Plain-table modules with free functions, export type at boundaries, unknown for untrusted input, the four-way failure ladder in a language with no exceptions worth catching, os.clock deadlines, and what luau-analyze and .luaurc actually enforce. Language only, not Roblox.
argument-hint: [the module, feature, or system to write, extend, or refactor]
---

Build: **$ARGUMENTS**

`/smart-simple-code` holds the reasoning for every rule below. This file is the Luau answer. For
Roblox work, use `/smart-roblox-code`, which is this file plus the client-server boundary.

## Module shape

```lua
--!strict
local Module = {}

export type Row = { readyAt: number, state: string, priority: number }
local Rows: { [Entity]: Row } = setmetatable({}, { __mode = 'k' })

local function getRow(entity: Entity): Row
    local row = Rows[entity]
    if not row then
        row = { readyAt = 0, state = 'idle', priority = 0 }   -- one literal, all fields
        Rows[entity] = row
    end
    return row
end

function Module.TrySomething(entity: Entity, now: number): (boolean, string)
    ...
end

return Module
```

- **A module is a table of functions plus module-level state.** Free functions cost nothing here:
  closures defined once at module scope are cached by the VM.
- **Build every state record in a single table literal.** Luau has a table-templates optimisation
  inspired by LuaJIT that only triggers when all fields are specified in the literal in one go.
  Assigning fields afterwards misses it.
- **Fixed field names, known at compile time.** Field access blends inline caching with hash
  prediction and is fast only when the name is a compile-time constant. A row of known fields is
  fast; a row keyed by runtime strings is not.
- `table.create(n)` when the maximum array size is known up front.
- **Weak-keyed state**: `setmetatable({}, { __mode = 'k' })` so a row dies with its entity. This is
  the one place a metatable earns its keep in a module.

**If you do write a class**, the official performance guidance is specific: store fields directly in
the table and methods in the metatable, and make it *one level*. "Avoid `__index` functions as well
as deep `__index` chains; an ideal object in Luau is a table with a metatable that points to itself
through `__index`." Do not cache methods in locals: `obj:Method()` compiles to a fused NAMECALL, and
the old Lua trick of hoisting the method is explicitly not recommended here.

## Types

- **`--!strict` at the top of new files.** The mode line is per-file: `--!nocheck` disables inference
  entirely, `--!nonstrict` is the default and infers `any` when it cannot tell, `--!strict` tracks
  types across statements.
- **`export type` is the only mechanism for a boundary type.** Bare `type` is file-local. The list of
  exported types is the module's real API, so keep row and internal shapes unexported.
- Luau is **structurally** typed: two tables are compatible if their shapes are. You do not need a
  nominal wrapper to make a value acceptable, and you cannot get nominal safety without a tag field.
- **Annotate untrusted input as `unknown`, never `any`.** `unknown` refuses to be used as anything
  until you refine it, which turns "I forgot to validate" into a type error. `any` does the opposite.
- Refinement narrows on `if x then`, `type(x) == 'string'`, and `assert`. Code after `error(...)` or
  `assert(false)` is treated as unreachable by both the checker and the linter.
- The **new type solver** reached general release in late 2025 and is the default: one shared
  inference pass for both modes, better refinements, read-only table properties, relaxed casting so
  you need fewer round trips through `any`, and user-defined type functions. The old solver remains
  available through 2026, so a codebase may still be on it. Check before assuming an error is real.

## Guards and the failure ladder

```lua
function Module.CanAct(entity: Entity, now: number): (boolean, string)
    if not entity then return false, 'no entity' end
    if not Module.IsAlive(entity) then return false, 'not alive' end
    if now < getRow(entity).readyAt then return false, 'cooldown' end
    return true, ''
end
```

- **Return `(false, reason)` for a policy refusal.** It type-checks as `(boolean, string)` and, unlike
  `error`, the checker and the reader both see it at the call site. This is the default.
- **`error` and `assert` are for programmer error only.** Prefer failing at load, where a typo in a
  definition table costs a boot failure rather than a support ticket.
- `pcall` returns `(true, ...)` or `(false, err)`; `xpcall` takes a handler that **can neither yield
  nor error**, which is the trap: a handler that touches a yielding API breaks the error path.
- Wrap designer-supplied or plugin-supplied callbacks in `pcall`. Do not wrap your own code in it to
  make errors go away.
- Missing content warns and keeps running. A `warn` a designer reads is worth more than an `error`
  that stops them working.

## Time

- **`os.clock()` is the deadline clock.** Sub-microsecond precision, monotonic, and with no defined
  baseline, which is exactly right for durations and exactly wrong for anything persisted or shared.
- `os.time()` is wall-clock Unix seconds. Use it for daily resets and save data, never for a cooldown.
- One comparison, one field, no per-frame writes:

```lua
if now < row.readyAt then return false, 'cooldown' end
row.readyAt = now + config.COOLDOWN
```

- Read the clock once per tick and pass `now` down. Two systems calling `os.clock()` separately will
  disagree, and the bug shows up only under load.

## Concurrency

- Luau is single-threaded with coroutines. `coroutine.close` releases a suspended thread's resources.
- Prefer whatever scheduler the host runtime gives you over raw `coroutine.resume`, because raw
  resume swallows an error into a return value that nobody checks.
- **Cancel by identity, not by handle.** Store a token, re-check it after resuming, and do nothing if
  it changed. This survives a scheduler that cannot cancel a running thread.
- Only plain tables cross a boundary between hosts or workers. Metatables, closures and userdata do
  not survive serialisation.

## Control flow as data

- A node is a plain record with a tag, arguments and children, built by a factory function. Node
  functions all share one signature and return a boolean.
- Use a singleton/literal union for the tag so the checker catches an unhandled case:
  `type NodeKind = 'sequence' | 'selector' | 'condition' | 'action'`.
- For priority-based state, one table of states, one table of priorities, and a single function that
  is the only thing allowed to write the state field.

## Config

- Config files are pure data: a table of tables, one comment per key, no requires and no functions.
- Read with a fallback (`cfg.WINDOW or 0.18`) so a missing key degrades rather than crashes.
- `table.freeze` a config table after construction to turn accidental writes into errors, and
  `table.isfrozen` to check. This is cheap and catches a whole class of bug where one system mutates
  another's constants.

## Tooling and validation

- **`luau-analyze` is the type checker and linter**, and ships with the language. The built-in linter
  has 28 warnings including `UnknownGlobal`, `LocalUnused`, `ImplicitReturn`, `MisleadingAndOr`,
  `TableOperations` and `DeprecatedApi`. Suppress one deliberately with `--!nolint NAME` and say why.
- **`.luaurc` is the project's answer, so read it first.** It is JSON with comments and sets
  `languageMode`, `lint` (a rule map, `"*"` for all), `lintErrors`, `typeErrors`, `globals`, and
  `aliases` for require-by-string. With no file at all the defaults are non-strict, type issues as
  errors, lint issues as warnings.
- Runtimes outside Roblox: **Lune** (Rust host, Node-like APIs) and **Lute**, the official one, whose
  `lute` CLI bundles a test runner, a linter and the type checker together.
- **There is no single standard test runner.** Lute ships one, Lune projects usually roll their own,
  and Jest-Lua comes from the Roblox lineage. Which one this project uses is a question to answer by
  looking, not by assuming.
- selene and stylua are widely used for linting and formatting, though not part of the language
  toolchain.

## Traps

- **`nil` is not `false` but both are falsy**, so `if x then` conflates "missing" with "off". When the
  difference matters, test `x ~= nil` explicitly.
- **`and`/`or` as a ternary breaks on `false`**: `cond and false or default` always yields `default`.
  The linter's `MisleadingAndOr` catches the common shape; it cannot catch all of them.
- **`#t` on a table with holes is undefined**, not "the biggest index". Track length yourself for
  sparse arrays.
- **Numeric keys and string keys in one table** is legal in Luau and a serialisation bug everywhere
  else. Pick one.
- `table.remove` inside a forward loop skips elements. Iterate backwards when removing.
- An `export type` you no longer use is still part of your API and still costs a reader. Delete it.
