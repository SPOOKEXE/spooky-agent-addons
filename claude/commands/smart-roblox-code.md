---
description: The Roblox specialisation of /smart-luau-code. The client-server trust boundary and where validation lives, remotes and what survives the wire, replication choices, GetServerTimeNow for cooldowns both sides agree on, the task and RunService scheduling model, Actors and parallel Luau, connection lifetime, and how a Roblox change is actually validated.
argument-hint: [the system, feature, or gameplay loop to write, extend, or refactor]
---

Build: **$ARGUMENTS**

This is `/smart-luau-code` plus the boundary. Everything in that file still applies: plain-table
modules, `export type` at boundaries, `(false, reason)` refusals, deadline timing, single-literal
state records. `/smart-simple-code` holds the reasoning behind all of it.

## The boundary decides the architecture

The official position, and it is worth quoting because it is stronger than most people write:
**"Assume every piece of data sent from the client has been manipulated, fabricated, or sent with
malicious intent."** The server is the ultimate source of truth for simulation state, rules,
progression and every critical decision.

The shape that follows:

1. **Receive** the client's request. It is a hint about intent, never a fact.
2. **Validate** that the action is possible and permissible, on the server, from server state.
3. **Act**, then replicate the result.

- **Keep logic and authoritative data in `ServerScriptService` and `ServerStorage` from day one.**
  Anything in a replicated container is readable by every client, so a definition table that decides
  drop rates does not go there. Moving it later is a migration; putting it there now is free.
- **One shared predicate, called by both sides.** A pure `CanDoX(state, request) -> (boolean, string)`
  in `ReplicatedStorage` that the client calls to grey out a button and the server calls to decide.
  One definition of the rule, two callers, and the client's copy is a convenience that the server
  never trusts. This is the idiomatic form rather than an officially named pattern, but it is what
  the mandated validation flow reduces to.
- The client owns feel: prediction, effects, UI, input buffering. The server owns truth: damage,
  currency, inventory, position of record.

## Remotes

- **`RemoteEvent` for one-way.** Fire and continue.
- **`RemoteFunction` yields until it gets a response**, which is fine server-bound and dangerous
  client-bound: **if the client never returns a value, the server yields forever.** Do not invoke a
  client from the server. Use a RemoteEvent and let the reply be another event.
- **`UnreliableRemoteEvent`** for continuous, non-critical data where the newest value supersedes the
  last one anyway.
- **Only plain data survives the wire, and the engine enforces it.** Functions are not replicated.
  Metatables are lost entirely. A table with both numeric and string keys is not supported. Non-string
  indices are converted to strings. An Instance the receiver cannot see arrives as `nil`. So a remote
  payload is a flat table of primitives, and anything else is a bug you will find in production.
- **One channel per feature, with an action string as the first argument**, and the action constants
  living in the shared module beside the predicate. This beats twenty remotes, and it puts the
  vocabulary of the feature in one place.
- **Never send per frame.** The named mistakes are replicating every frame when it is not needed,
  replicating on raw user input with no throttle, and sending more than is required. Frequent
  invocation spends real CPU per frame parsing packets. Design against the question the docs pose:
  what happens if this is used a thousand times a second? Answer it with a rate limit, and keep a
  separate, stricter budget for malformed payloads.
- There is no officially documented byte-per-second cap. The widely repeated figure is community
  lore, so do not design to a number you cannot cite; design to "as little as possible, as rarely as
  possible".

## Replication of state

- **Attributes** replicate automatically and are observable with `GetAttributeChangedSignal`. Ideal
  for a small number of per-instance values both sides read, and the natural home for a cooldown
  deadline, since the client can render the remaining time without asking.
- **Remotes** when you want explicit timing and control over who receives.
- **A diffed data table** for large player state: keep the authoritative copy on the server, send
  deltas on change, and give the client a read-only view with added/updated/removed signals.
- **Ordering across kinds is not guaranteed.** A property change and a `FireAllClients` may arrive in
  either order. Never write code that depends on one landing before the other; listen for the change
  you care about instead of assuming it already happened.

## Time across the boundary

- **`workspace:GetServerTimeNow()` is the only clock for a cooldown both sides must agree on.** It is
  monotonic and synchronised.
- `os.clock()` has no shared baseline and returns different results on server and client, so it is
  correct for local durations and wrong for anything crossing the wire.
- `tick()` is not monotonic, is timezone-dependent, and is deprecated. If existing code uses it for
  cooldowns, that is a bug waiting for a clock change, and worth saying so in the report.
- `os.time()` is second-granularity wall clock, for daily resets and save data.
- Store the deadline as an attribute on the character or instance: both sides read the same number,
  the client can predict, and the server still re-checks before acting.

## Scheduling

- **`task.spawn`** runs the thread **immediately** and inherits the current serial or parallel phase.
  It is not a way to make something asynchronous, it is a way to fan out. Your invariants must hold
  at the call.
- **`task.defer`** resumes at the end of the current resume point within the frame. This is what
  "start after everything has wired" is built from.
- **`task.delay(t, f)`** resumes on the first Heartbeat after `t` elapses.
- **`task.wait()`** yields and resumes on the next Heartbeat, returning the actual elapsed time. Use
  the return value; do not assume it was the time you asked for.
- **`task.cancel` cannot cancel a thread that is currently executing**, which is exactly why
  cancellation is done by identity: store a token, re-check it after the delay fires, and return if
  it changed.
- Legacy `wait()`, `spawn()` and `delay()` are throttled and superseded. Replace them when you touch
  the surrounding code, not as a sweep.

**Frame order**, and note the renames, since older code and older tutorials use the old names:

| event | side | old name |
| --- | --- | --- |
| PreRender | client | RenderStepped |
| PreAnimation | client | |
| PreSimulation | both | Stepped |
| PostSimulation | both | |
| Heartbeat | both, where most scripts run | |

`BindToRenderStep` is client-only and takes an explicit priority, which is the tool when ordering
against the camera actually matters.

## Parallel Luau

- Actors are the unit of execution isolation, and **scripts in the same actor always run sequentially
  with respect to each other**, so one actor buys you nothing. Many actors, or do not bother.
- Enter with `task.desynchronize()` or `:ConnectParallel()`, leave with `task.synchronize()`.
- **Scripts running in parallel generally cannot write to the data model.** Properties are tiered
  Unsafe, Read Parallel, Local Safe and Safe, and you must check the tier of every property you touch
  rather than assuming.
- `SharedTable` is the cross-actor structure with safe atomic updates.
- Parallel pays only when compute per batch clearly exceeds the cost of getting data in and out.
  Roblox publishes no official speedup figures, so measure your own case before committing to the
  restructure.

## Lifetime and cleanup

- **`Destroy()` sets `Parent` to nil, locks it, disconnects the instance's own connections, and
  destroys its children.** Then set your own references to nil, or the table holding them keeps the
  instance alive.
- **The leak is the connection that is not the instance's own**: a `RunService.Heartbeat` handler or a
  module-level signal connection that closes over an entity. `Destroy` will never touch it. Every such
  connection gets an owner that disconnects it, and the owner dies with the thing.
- **`CollectionService` tags are the fix for nondeterministic wiring order.** Connect
  `GetInstanceAddedSignal` **first**, then iterate `GetTagged()`. Done in that order you cannot miss
  an instance that appears between the two calls; done the other way round you can, and it will happen
  once a month in production. Tagged instances often carry connections and tables, so clean those up
  on removal.

## Validation

State which of these you actually did, because they prove different things.

- **Rojo plus luau-lsp** is the standard external toolchain. The language server resolves the
  DataModel tree from a sourcemap, so keep `rojo sourcemap --watch` running or the types are guesses.
- **Studio playtest**, single and multi-client. Multi-client is the only way to see a replication
  ordering bug.
- **Exercise the feature through an admin or dev command** rather than playing to it. Write that
  command with the feature.
- **Jest-Roblox is the maintained test framework**; TestEZ is legacy with community forks. If neither
  is present, say that rather than implying a suite exists.
- **Open Cloud Luau Execution** runs Luau headlessly in a place with full DataModel access, which is
  the lever for CI on gameplay logic.

## Traps

- **`WaitForChild` with no timeout in a script that can run before replication finishes** hangs
  forever and looks like a crash. Pass a timeout and handle nil.
- **StreamingEnabled means an instance that exists on the server may not exist on the client**, so
  client code that walks the workspace needs to tolerate absence rather than assert presence.
- **Attributes only hold a limited set of types.** Reaching for one to hold a table is a sign the
  state belongs in a replicated data table instead.
- A `RemoteFunction` handler that errors propagates the error to the caller as a failure with almost
  no context. Return `(false, reason)` instead.
- Anything under `ReplicatedStorage` is readable by every client, including the definitions you
  thought were private. Secrecy is decided by where the file lives, not by who requires it.
