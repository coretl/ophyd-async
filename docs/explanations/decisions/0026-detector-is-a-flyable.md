# 26. StandardDetector is a StandardFlyable

Date: 2026-08-28

## Status

Accepted

## Context

`StandardDetector` re-implemented `prepare`, `kickoff` and `complete` and threaded its own
state between them, while `StandardFlyable` already did that for every other flyer. The
two drifted apart: the detector guarded its stages by checking a context for `None`, so a
verb called too early failed with `AttributeError: 'NoneType' object has no attribute ...`.

`StandardFlyable` requires a single `FlyableLogic` in a `logic` property
([ADR 0021](0021-single-logic-property.md)), and a detector had three logic objects
registered by `add_detector_logics()`.

## Decision

### One `DetectorLogic` holds the trigger, acquire and data logics

`DetectorLogic` is what `logic` returns, and the orchestration (preparing the trigger
logic, making the data providers, arming, waiting) moves into it. It is a frozen dataclass
with no back-reference to its Device, so [ADR 0017](0017-standard-movable.md) holds
unqualified.

```python
class MyDetector(StandardDetector):
    def __init__(self, prefix: str, name: str = "") -> None:
        self.driver = MyDriverIO(prefix)
        super().__init__(name=name)

    @cached_property
    def logic(self) -> DetectorLogic:
        return DetectorLogic(
            trigger=MyTriggerLogic(self.driver),
            acquire=MyAcquireLogic(self.driver),
        )
```

- `add_detector_logics()` is removed, not kept as a forwarder: a detector's logics are fixed
  for its lifetime, and two ways to wire one Device is one too many.
- `StandardDetector` takes no `__init__` arguments, as they would interfere with multiple
  inheritance. `logic` is an `abstract_cached_property`; assigning `self.logic` in
  `__init__` cannot replace it, because `ABCMeta` checks abstract methods before `__init__`
  runs. `DetectorLogic.with_device()` builds an ad-hoc detector, like any flyer's.
- Datakeys are named after the Device, whose name only exists once `init_devices` calls
  `set_name`, so `StandardFlyable` passes the name to `prepare_to_fly`.

### The Device binds the collect verb at `prepare()`

A detector makes stream assets or event pages, never both
([ADR 0025](0025-bounded-data-logic-tier.md)). The bluesky bundler picks which to call
from the *static* type, so declaring both would silently drop the pages, and declaring
neither could not fly. So `StandardDetector.prepare()` overrides the flyer's and, when it
succeeds, binds the verb `DetectorLogic.produces` names as an instance attribute and
removes the other. A step scan restarts its bounded providers at each point, but they must
make the same datakeys, so it keeps the verb it was prepared with.

This is a bundler limitation: a device knows what it produced, and deciding from that is a
few lines in `RunBundler.collect`. Once released, both verbs become plain methods and the
override goes.

### Pageable and streamable data mix when the stream assets cover them

[ADR 0025](0025-bounded-data-logic-tier.md) made carrying both kinds of data logic an
error, but it is useful: an areaDetector stats total can go into the HDF file as an
NDAttribute *and* be read from the plugin's buffer. So:

- A bounded logic leaves out the datakeys the unbounded ones make, and sits the scan out,
  unarmed, if that is all of them.
- A registered signal whose name a provider key has is dropped from `read()` and
  `describe()` while prepared, and keeps its hint, since the replacement has its name.
- The provider key must have the signal's dtype and shape, optionally with a leading
  `collections_per_event` dimension. Anything else is a `ValueError` at `describe()`: two
  different things share a name, and one must be renamed.
- A registered signal that nothing suppresses is read once per event.

This is a general `StandardReadable` hook mapping a datakey name to `None` or a replacing
`DataKey`, matched by name so a signal is found wherever it is registered. `None` also drops
the hint, as it is for a signal that is switched off.

### The detector asks its logic which readables the data hides

The core `DetectorSuppressLogic`, `DetectorLogic.suppress`, names the readables the
detector's data hides. It is asked at the first `prepare()` of a stage and the answer kept
in the prepare context, so the frozen logic holds no state. Each later `prepare()` asks
again and raises on a difference, before any data logic starts, so a run is never split
part way and no PV the user owns is written.

### The detector flies with one `fly()` generator

`FlyableLogic.fly` is one async generator
([ADR 0022](0022-fly-generator-and-ensure-stopped.md)). Progress is collections written
since `prepare_to_fly` noted the count (`ctx.unbounded_start_collections`) out of those
requested. `unstage()` cancels the fly, so no stop logic is needed beyond
`ensure_stopped`.

A fly scan collects only what the data logics make: registered readables are read only by
`trigger()`.

### An acquisition is a generator too

`start_acquiring` and `wait_for_idle` had to hold what the first started on the logic, and
cancelling between them left it running. They become one async generator, `acquire()`, which
owns what it starts, so closing it at its `yield` ends it:

- An externally triggered detector arms in `prepare()`, so it is waiting on its trigger
  source when `kickoff()` returns, and the context holds it until `fly` or `trigger()` takes
  it. It cannot be prepared again before then, so nothing replaces it.
- One still armed at `unstage()` is closed then, not at garbage collection; stopping the
  detector stays the job of `ensure_stopped`.
- `ADAcquireLogic` writes `WaitForPlugins` (on by default) at every `stage()`, not as the
  IOC booted: without it a step scan silently reads plugin scalars of the previous frame,
  and the wait is free when a file writer is slowest anyway. `ADContAcqAcquireLogic` has
  none, as its driver never stops acquiring.

### One kickoff per prepare

The detector was prepared for N events and kicked off a few at a time (`events_to_kickoff`),
but nothing drove that: the RunEngine kicks off once, as for every other flyer. It goes;
`kickoff()` requests the whole scan, and a second needs a new `prepare()`. A detector with
no data logic is still a valid flyer for its triggering.

### One context per prepare, kept until the next `stage()`

One `DetectorCtx` threads from `prepare_to_fly` to `fly`; what exists only once flying
lives in `fly`, so no field waits for kickoff as `| None`. The logic holds no state: the
next `prepare()` gets the last context as `previous_ctx`, which is how unbounded providers
carry over and bounded ones are restarted ([ADR 0025](0025-bounded-data-logic-tier.md)).
The trigger logic is not re-prepared per step point.

- A failed `prepare()` leaves no context, so a retry is a first prepare of the stage: it
  takes the hidden set afresh and starts every data logic, whatever the failed one got to.
- The context outlives `complete()`, so `collect_while_completing` can collect after it.
- A step scan's collect publishes up to the point `trigger()` waited for, never re-reading
  the counter, so it publishes exactly what `trigger()` saw written.

`trigger()` runs in a `TRIGGERING` state, so `unstage()` cancels it, `kickoff()` cannot
overlap it, and a failed one leaves the detector staged. It prepares implicitly if nothing
has been.

## Consequences

- A verb called out of order raises a `TransitionError` naming it and the state (`<name>:
  describe not allowed from STAGED, allowed from: ...`), not `Prepare not run`, and a second
  `kickoff()` without a `prepare()` is rejected.
- A detector satisfies neither `WritesStreamAssets` nor `EventPageCollectable` until
  prepared; the RunEngine only dispatches on them after prepare.
- `StandardDetector` no longer implements `complete`, `kickoff`, `stage` or `unstage`, so
  lifecycle changes reach detectors too. It adds `__on_*` hooks for `read()`, `describe()`,
  `hints`, suppression and `unstage()`, registered in lists rather than overriding the
  verbs, so mix-ins combined on one Device do not shadow each other through the MRO. It
  overrides only `prepare()`, to bind the collect verb.
