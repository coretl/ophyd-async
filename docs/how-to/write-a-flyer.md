# How to write a flying Device

A *flyer* is a Device that hardware-triggers a scan: bluesky prepares it, kicks it
off, then waits for it to complete while the hardware runs autonomously. Rather than
implement the `Preparable`, `Flyable`, and `Stageable` protocols by hand, compose a
[](#FlyableLogic) instance into a [](#StandardFlyable). `StandardFlyable` drives the
phases in order and threads a context object between them:

- `prepare(value)` → `FlyableLogic.on_prepare(value)` works out the scan, sets the
  hardware up and moves to the start, returning a context object. It may report
  progress.
- `kickoff()` → `FlyableLogic.on_kickoff(ctx)` starts the scan, returns once it is up
  to speed, and returns the status of what is now running (or `None`).
- `complete()` → `FlyableLogic.on_complete(ctx, status)` blocks until the scan is done.
  It may report progress.

The flyer must be staged first: `prepare()` is only allowed after `stage()`, `kickoff()`
only after `prepare()`, and `complete()` only after `kickoff()`; a verb called out of
order raises `TransitionError`. After `complete()` the flyer can `prepare()` again for the
next point without restaging. `stage()` raises unless the flyer is unstaged, as does a
second `prepare()` while flying or a second concurrent `complete()`. A `complete()` that is
cancelled or fails cancels the status `on_kickoff` returned and leaves the flyer staged.

`stage()`/`unstage()` run the logic's `on_stage`/`on_unstage`, which default to `stop`.
`unstage()` works from any state: it cancels the verb in flight (which sees
`CancelledError`) and the status `on_kickoff` returned, as well as stopping the hardware.
`stop()` on a combined move and fly device only affects `set()`; a fly ends with
`complete()` or `unstage()`. The hook names are deliberately not bluesky verbs, so a logic passed to a plan
stub does not look like a device.

## Implementing FlyableLogic

Subclass [](#FlyableLogic) — it is generic over three types: `PrepareT`, the per-scan
info passed to `prepare()`, `CtxT`, the context threaded between the phases, and
`FlyStatusT`, the status `on_kickoff` returns. The last two default to `None`, so
`FlyableLogic[PrepareT]` carries no state and has nothing to wait for. The context is
fixed once `on_prepare` returns it, so use a frozen dataclass where practical; anything
that changes during a scan is an object the context holds. Every hook is an
`async def`. Fill in the hooks that your hardware needs:

| Method | Default behaviour | Override to… |
|--------|-------------------|--------------|
| `on_prepare(value)` | *abstract* | work out the scan, set up the hardware, and return a context, or a `(context, progress)` tuple to report progress |
| `on_kickoff(ctx)` | *abstract* | start the scan and return the status of what is running, or `None` |
| `on_complete(ctx, status)` | *abstract* | wait for the scan to finish, or return an async iterator of [](#WatcherUpdate) to report progress |
| `stop()` | no-op | disarm the hardware and wait for it to stop |
| `on_stage()` | calls `stop()` | set the hardware up on `stage()` if it differs from stop |
| `on_unstage()` | calls `stop()` | clean the hardware up on `unstage()` if it differs from stop |

To report progress, `on_prepare` returns a 2-tuple of the context and an async iterator
of [](#WatcherUpdate), and `on_complete` returns the iterator; otherwise they just do
their work and return. `StandardFlyable` iterates the iterator and stamps its own name
on each update. A context that is itself a 2-tuple ending in an async iterator would be
misread as progress.

There are two ways to get a Device out of a `FlyableLogic`: mix `StandardFlyable` into
a Device class (for hardware that is always a flyer, like a motor), or call
[](#FlyableLogic.with_device) to wrap a bare logic object in an ephemeral flyer inside
a plan (for hardware configured per scan, like a PandA block).

## A movable device that also flies

An EPICS [](#Motor) is the first case: it is a [](#StandardMovable) *and* a
`StandardFlyable` (and a [](#StandardReadable)). A single logic object,
`MotorFlyableMovableLogic`, implements both [](#MovableLogic) and `FlyableLogic`, so
its fly hooks can reuse the same `move`/`check_move` methods as a normal `set()`. The
per-scan info is a [](#FlyMotorInfo) and the threaded context is a small frozen
dataclass of everything `on_prepare` works out:

```{literalinclude} ../../src/ophyd_async/epics/motor.py
:language: python
:pyobject: MotorPrepareCtx
```

`on_prepare` validates the requested velocity and the run-up and run-down positions
against the motor's limits, and returns the context together with the progress of the
move to the run-up start. That progress is [](#watch_readback), which watches the
readback of the logic until the move is done:

```{literalinclude} ../../src/ophyd_async/epics/motor.py
:language: python
:pyobject: MotorFlyableMovableLogic.on_prepare
```

`on_kickoff` starts the move to the far end of the run-down and returns its status:

```{literalinclude} ../../src/ophyd_async/epics/motor.py
:language: python
:pyobject: MotorFlyableMovableLogic.on_kickoff
```

`on_complete` receives that status, and returns the progress while it runs:

```{literalinclude} ../../src/ophyd_async/epics/motor.py
:language: python
:pyobject: MotorFlyableMovableLogic.on_complete
```

Opting in to progress is up to each hook: the same [](#WatcherUpdate) machinery as
[](#StandardMovable.set) is used, but a flyer whose logic is *not* movable (see the next
section) just waits with no progress updates. See
[](../explanations/when-to-extend-movable.md) for more on `MovableLogic`.

Wire the logic into the Device with a `@cached_property`. `Motor` builds one shared
instance and returns it as its single `logic`, which both the movable and flyable
mix-ins declare (see [ADR 0021](../explanations/decisions/0021-single-logic-property.md) for why), so the move and fly paths act
on the same signals:

```{literalinclude} ../../src/ophyd_async/epics/motor.py
:language: python
:pyobject: Motor.logic
```

## An ephemeral flyer created in a plan

Some hardware is only a flyer for the duration of a scan and is configured differently
each time. The PandA position-compare block is an example: [](#StaticPcompFlyableLogic)
wraps a single `PcompBlock` and carries no state between phases, so it is a
`FlyableLogic[PcompInfo]`. Its `on_prepare` disarms then programs the block from
the per-scan [](#PcompInfo):

```{literalinclude} ../../src/ophyd_async/fastcs/panda/_fly_logic.py
:language: python
:pyobject: StaticPcompFlyableLogic.on_prepare
```

`on_kickoff` enables the block and waits for it to go active, while `on_complete` waits
for it to go inactive again (the context and status are `None`, so they are ignored):

```{literalinclude} ../../src/ophyd_async/fastcs/panda/_fly_logic.py
:language: python
:pyobject: StaticPcompFlyableLogic.on_kickoff
```

```{literalinclude} ../../src/ophyd_async/fastcs/panda/_fly_logic.py
:language: python
:pyobject: StaticPcompFlyableLogic.on_complete
```

`stop()` disarms the block; it is called by the default `on_stage`/`on_unstage`:

```{literalinclude} ../../src/ophyd_async/fastcs/panda/_fly_logic.py
:language: python
:pyobject: StaticPcompFlyableLogic.stop
```

Rather than declare this logic on a Device class, construct it in the plan from the
PandA block you want to drive and call [](#FlyableLogic.with_device) to get a ready
flyer to prepare, kickoff, and complete:

```python
flyer = StaticPcompFlyableLogic(panda.pcomp[1]).with_device(name="pcomp")
```

```{seealso}
[](./choose-right-baseclass.md) for how `StandardFlyable` fits alongside the other
base classes.
```
