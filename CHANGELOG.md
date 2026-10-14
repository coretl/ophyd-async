# Changelog

## Unreleased

### Breaking changes

- `StandardMovable.movable_logic` and `StandardFlyable.flyable_logic` are replaced by a single `logic` property, declared by each mix-in with its own type, so a Device that moves and flies implements it once with a logic inheriting both `MovableLogic` and `FlyableLogic` (see ADR 0021). Rename your `movable_logic`/`flyable_logic` `@cached_property` to `logic`; no deprecated aliases are left. Forgetting it is now a `TypeError` at instantiation rather than an `AttributeError` inside `Device.__init__`. `StandardFlyable` is now abstract, use `FlyableLogic.with_device()` for an ephemeral flyer.
- `FlyableLogic.on_prepare(value)` still returns the context, so an existing one keeps working, and may now return `(context, progress)` instead, where `progress` is an async iterator of `WatcherUpdate` that reports the slow part (such as moving to the start) to watchers. `start_prepare` and `finish_prepare` do not exist.
- `FlyableLogic.on_kickoff(ctx)` now returns the status of what is running (an `AsyncStatus`, or `None`) instead of a new context. `StandardFlyable` holds it, and cancels it on `unstage()` and when `complete()` is cancelled or fails.
- `FlyableLogic.on_complete(ctx)` is now `on_complete(ctx, status)`, where `status` is what `on_kickoff` returned. It may return an async iterator of `WatcherUpdate` to report progress, or wait itself and return `None`. All the hooks are `async def`.
- `FlyableLogic` and `StandardFlyable` take a third type parameter, `FlyStatusT`, the status `on_kickoff` returns, so `FlyableLogic[PrepareT, CtxT, FlyStatusT]`. It and `CtxT` default to `None`, so `FlyableLogic[PrepareT]` is a logic with no context and no status. `ophyd_async` now depends on `typing-extensions` for these defaults.
- Renamed `MotorFlyCtx` to `MotorPrepareCtx`, `PmacFlyCtx` and `PmacPrepareContext` to `PmacPrepareCtx`, and `SimMotorMoveLogic` to `SimMotorLogic`.
- A flyer must now be staged before it is prepared: `prepare()` is only allowed after `stage()`, `kickoff()` after `prepare()` and `complete()` after `kickoff()`, and a verb in the wrong state raises `TransitionError` (a `RuntimeError`). `stage()` raises unless the flyer is unstaged, so staging an already-staged flyer, or one with a verb running, now raises `TransitionError` where it used to silently reset it. A `complete()` that is cancelled or fails cancels the status `on_kickoff` returned and leaves the flyer staged. After `complete()` a flyer can be prepared again without restaging. Stock bluesky plans that stage their devices (`bp.count`, `bp.scan`) keep working; `bp.fly` neither stages nor prepares, so stage the flyer and prepare it before `bp.fly`, for example with `bpp.stage_wrapper`.
- `unstage()` now cancels the `prepare()`, `kickoff()` or `complete()` that is running. `stop()` on a Device that moves and flies still only affects `set()`.
- `SimMotor.stage()` and `SimMotor.unstage()` now stop the motor, as `Motor` does.

### Added

- `watch_readback()` reports the progress of a move by watching a `MovableLogic` readback, for use from `FlyableLogic` hooks.
- `SimMotor.prepare()` and `SimMotor.complete()` report progress to watchers, as `Motor` does.
- `TransitionError`, a `RuntimeError` raised by Standard devices for a verb called in the wrong state.
- `abstract_cached_property` in `ophyd_async.core`, for declaring a `@cached_property` that a subclass must implement.

### Fixed

- `complete()` finishing after `unstage()` no longer makes the device look prepared or completed to a new fly.
- `unstage()` now cancels a running fly, so the PMAC trajectory stops appending points and no status is left running after the hardware is stopped.
- `unstage()` now ends a running `complete()` when `kickoff()` returned no status, such as for the PandA.
- A failed `prepare()` no longer leaves the flyer prepared with the failed context, so there is nothing to kick off.
