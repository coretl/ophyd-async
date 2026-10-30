# 22. `fly` is a generator, and `ensure_stopped` waits

Date: 2026-10-20

## Status

Accepted

Amends ADR 0017: `MovableLogic.stop()` is now `ensure_stopped()`, and the fly hooks are different from the ones that ADR describes.

## Context

`FlyableLogic` had separate `on_kickoff` and `on_complete` hooks, and `stop()` returned as soon as it had asked the hardware to stop. This allowed three kinds of bug. A scan started in `on_kickoff` could outlive `unstage()`, because nothing owned it. A new `stage()` or `set()` could start while the previous motion was still decelerating. A `set()` caught by `stop()` before its move had started was not ended, and ran on after it.

## Decision

**`prepare_to_fly()` and `fly()` are async generators.** `prepare_to_fly` yields the context first, then any progress; `fly` yields `KICKED_OFF` once flying, then any progress. Everything a scan starts lives inside the generator, so closing or cancelling it is enough to stop it. Writing it as `async with AsyncStatus(...)` plus `yield` makes the ownership visible in the code, and a failure before or after the first yield maps cleanly on to the verb that failed. Two hooks that shared state through the logic, or a hook that returned a `(context, progress)` tuple, could not guarantee this: the progress iterator was not closed if `prepare()` failed, and a context that was itself a tuple was ambiguous. `prepare()` keeps the context only once the generator has finished, so a half-failed prepare leaves nothing to kick off. The first yield is checked at runtime, as a `WatcherUpdate` yielded first is an easy mistake that the type checker cannot see.

**The fly runs in the foreground of the status that `complete()` returns.** `kickoff()` makes a `WatchableAsyncStatus` that drives `fly`, and returns once it has yielded `KICKED_OFF`; `complete()` returns that same status. It already replays the latest update to a late watcher, and the state machine cancels it on `unstage()`, so no separate task, update forwarding or cleanup path is needed. `WatchableAsyncStatus` closes the generator it drives as soon as it is cancelled, rather than when it is collected. Cancelling `complete()`'s status therefore cancels the fly, and a fly cleanup that fails is reported by that status instead of its `CancelledError`.

**`ensure_stopped()` waits until the motion has ended.** `unstage()` cancels `fly`, then calls `ensure_stopped()`, so when `unstage()` returns nothing is running and the hardware is still. Returning early would leave the next `stage()` or `set()` racing the deceleration of the last one.

**One stop future ends a set.** Each `set()` passes a fresh future to the `stop` argument of its `WatchableAsyncStatus`, which races the operation against the future: when the future resolves first the operation is cancelled and the future's value becomes the outcome; statuses still have no public `stop()`, as bluesky treats a status as a task the device stops. The device keeps one future, the latest `set()`'s. `stop()` resolves it with `None` for `stop(success=True)`, so the `set()` finishes normally, or with `StoppedError("Device <name> was stopped.")`, a `RuntimeError`, which it raises. `RunEngine.pause` calls `stop(success=True)` on everything it has set and expects that to finish without an error. A new `set()` resolves the previous future with a `StoppedError` saying it was superseded, so only one `set()` runs at a time; its move is cancelled but the hardware is not stopped, as the new move retargets it. The future is resolved with a value rather than an exception, so a future nobody awaits raises nothing. `stop()` ends only a `set()`, never a `prepare()` or fly: those are ended by `unstage()` through the state machine, so a `stop()` cannot reach a verb it was not meant for. A stop that arrives before the move starts ends the `set()` without touching the hardware, and a second `stop()` finds the future already resolved and only calls `ensure_stopped()` again. `stop()` does not wait for the `set()` to finish failing before `ensure_stopped()`: the hardware should stop as soon as possible. Logics no longer track whether they were stopped: a `move()` may end early without an error when stopped, as an EPICS motor's put completion does, and the stop future decides the outcome.

**Only a flyer stages, and `unstage()` always stops.** A `StandardMovable` alone does nothing on `stage()` or `unstage()`: `stop()` is how it is stopped. `ensure_ready()` is a hook of `FlyableLogic` only, which defaults to `ensure_stopped()`, and only `StandardFlyable` calls it, on `stage()`; a device that moves and flies therefore stops a `set()` in progress at `stage()`, no worse than `caput MOTOR.STOP`. `StandardFlyable.unstage()` cancels a `prepare()` or fly, then always calls `ensure_stopped()`, whether or not the device was prepared. Sparing a `set()` there would only defer its stop: the bluesky RunEngine stops every movable it set when a run ends or pauses. The cost is that every staged `Motor` gets one `.STOP` at `unstage()`, because `Motor.ensure_stopped()` sends it unconditionally: `.DMOV` can flicker at standstill on some motors, so it is not read to decide, and the motor record does not write `.VAL` from `.RBV` on `.STOP`.

The resulting model:

| Verb | Mix-in | Does |
|------|--------|------|
| `stage()` | `StandardFlyable` | calls `logic.ensure_ready()` |
| `unstage()` | `StandardFlyable` | cancels a `prepare()` or fly in progress, then calls `logic.ensure_stopped()` |
| `stop()` | `StandardMovable` | ends the `set()` via the stop future, then calls `logic.ensure_stopped()` |
| `set()` | `StandardMovable` | supersedes the `set()` in progress |

The logic hooks `prepare_to_fly` and `fly` are async generators with a first-yield marker: the context for `prepare_to_fly`, `KICKED_OFF` for `fly`.

## Consequences

- The stage and unstage of the previous plan are the only thing that keeps a motion from overlapping the next one. `stage()` can start `ensure_ready()` while the `ensure_stopped()` from the previous `unstage()` is still running, and we accept that. We assume every plan stages its devices with the stage/unstage decorator, as the stock plans do, and the tests run plans back to back and after an interruption to show that this behaves. A plan that does not stage gets none of this.
- A `FlyableLogic` that also moves gets `ensure_stopped()` called on `stage()` by default, which stops a `set()` in progress. Override `ensure_ready()` if stopping is wrong there.
- A plan that fails while a `set()` or `complete()` is in flight sees bluesky raise a `FailedStatus` for it from `unstage()`, where it would have seen the plan's own exception first.
- `StandardFlyable` has no `stop(success)` of its own, and `stop()` on a device that moves and flies does not end a `prepare()` or fly. `ensure_stopped` still halts the hardware, so a `stop()` during the run-up (for example `bps.stop`, or a `RunEngine` pause of a device that was also `set()`, with `rewindable=False`) can leave `prepare()` finishing short of the start. We do not check the position, as motors have a tolerance and none is baked in here. A future `pause()` hook is the better place to fix this.
- `complete()` returns the fly status even after the fly has failed, until the next `prepare()` or `unstage()`, so the failure is reported rather than a `TransitionError`.
