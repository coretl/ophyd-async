# 24. StandardDetector is a StandardFlyable

Date: 2026-08-28

## Status

Accepted

## Context

`StandardDetector` re-implemented `prepare`, `kickoff` and `complete` and threaded its own
state between them, while [](#StandardFlyable) already existed to do exactly that for every
other flyer in the tree (`Motor`, PMAC, the PandA trigger logics). Two implementations of
one lifecycle drift apart: the detector guarded its stages by checking whether a context
object was `None`, so calling a verb too early failed with `AttributeError: 'NoneType'
object has no attribute ...` rather than saying what was wrong.

`StandardFlyable` requires a single [](#FlyableLogic) in a `logic` property
([](./0021-single-logic-property.md)), and a detector had three logic objects registered on
the Device by `add_detector_logics()`.

## Decision

### The three logics become children of one `DetectorLogic`

[](#DetectorLogic) holds the trigger, acquire and data logics, and is what `logic` returns.
The orchestration moves with them: preparing the trigger logic, making the data providers,
arming, and waiting for collections are all its methods now. It holds **no** back-reference
to its Device, so [](./0017-standard-movable.md) continues to hold unqualified.

A detector builds it in `logic`:

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

`add_detector_logics()` is removed rather than kept as a forwarder: a detector's logics are
fixed for its lifetime, so there is nothing a post-construction call can do that the
constructor cannot, and keeping it would leave two ways to wire the same Device.

`StandardDetector` itself takes no `__init__` arguments, since a `Standard*` class that did
would interfere with multiple inheritance. The logic is supplied by overriding `logic`,
which `abstract_cached_property` makes a requirement checked when the Device is
instantiated. Assigning `self.logic` from `__init__` cannot serve instead: `ABCMeta` checks
`__abstractmethods__` in `__call__`, before `__init__` runs, so the class must override the
property to be concrete. For an ad-hoc detector in a plan or a test,
`DetectorLogic.with_device()` builds a Device around a logic, exactly as
`FlyableLogic.with_device()` does for any other flyer.

### One thing the logic cannot get from its own fields

Data logics name their datakeys after the Device, whose name does not exist until
`init_devices` calls `set_name` on context exit. `StandardFlyable` runs hooks a subclass
registers with the context as soon as `prepare_to_fly` yields it. `StandardDetector`
registers one, which writes its name and the description of its registered signals into the
context, then calls `DetectorLogic.serve_data` with it to choose and start the data
providers. It passes its name to `get_hinted_fields()` too. The logic holds no reference to
the Device: `StandardFlyable` calls `prepare_to_fly` with the `TriggerInfo` alone, so the
context is where the Device leaves what serving needs.

### Which collect verb to expose is the Device's business

A detector produces stream assets or event pages depending on what its data logics make for
a given scan, never both (see [](./0023-bounded-data-logic-tier.md)). The bluesky bundler
decides which to ask for from the *static* type — `collect_pages()` is called only on a
device that is not a `WritesStreamAssets` — so a detector that declared both would have its
pages silently dropped, and one that declared neither could not fly at all.

So the hook `StandardDetector` registers for `prepare()` binds whichever verb applies as a
real instance attribute once the data is served, and removes the other. It is the Device that
does this, from what `DetectorLogic.produces` says the serving providers make; the logic is
not told where that goes, so it holds no back-reference. Only a `prepare()` that serves binds
or unbinds: nothing asks for a collect verb before a `prepare()`, and the step-boundary re-arm
on each point of a step scan cannot change which verb applies, because it changes no
provider. A failed `prepare()`, `kickoff()` or `complete()` leaves the verb bound until the
next `prepare()`, and calling it then raises.

Choosing from the static type is a bundler limitation rather than a fact about detectors: a
device knows what it produced, and the bundler could decide from *that*, erroring only if
one device reported both ways in the same run. That change is a few lines in
`RunBundler.collect`, and with it both verbs become plain methods and the binding here goes
away. Until it is released, binding per prepare is what works against a released bluesky.

### Mixing pageable and streamable is an error unless the pageable keys are suppressed

[](./0023-bounded-data-logic-tier.md) made carrying both kinds of data logic an error, because
the bundler treats stream assets and event pages as mutually exclusive. Carrying both is still
useful, though: the same quantity can be written durably into a file *and* read from a plugin's
buffer — an areaDetector stats total, which the HDF writer pulls in as an NDAttribute.

So the rule relaxes. A bounded logic is started after the unbounded ones and handed the
datakeys they make, which its provider leaves out. Where the stream assets make **every**
datakey a finite buffer would produce, the durable copy wins and the buffer sits the scan out,
unarmed. A registered signal is suppressed the same way by a provider key, and is
dropped from `read()` and `describe()` while the detector is prepared. Its hint stays, as the
provider key has its name.
A registered signal gives one value per event, so a provider key of its dtype and of its shape,
or of that shape with one extra leading dimension (`[collections_per_event, *shape]`), suppresses
it. Any other dtype or shape is a `ValueError` at `describe()`: two different things share a
name, so one must be renamed. A registered signal that nothing suppresses is read once per event.
Nothing has been started when that is decided, because a logic's `start_provider` returns
`None` without writing (same ADR).

Suppression is one case of a general `StandardReadable` hook: a mapping from datakey name to
either `None` or the datakey that replaces it, matched by name after the registered children
have been read, so a signal is found wherever it is registered. `None` drops the name from
`read()`, `describe()` and `hints`: it is for a signal that is switched off, which has no
replacement. A datakey drops it from `read()` and `describe()` after checking the datakey can
replace it, and leaves it in `hints`, because the replacement has the same name. A data logic
that hints the name of a signal it replaces makes `hints` raise, as two hints of one field
always do.

### The detector flies with one `fly()` generator

`FlyableLogic.fly` is a single async generator ([](./0022-fly-generator-and-ensure-stopped.md)):
it yields `KICKED_OFF` once flying, then `WatcherUpdate`s, and `complete()` finishes when it
returns. A detector's progress is "collections written out of collections requested", so
`DetectorLogic.fly` notes where the data providers have got to, arms an internally triggered
acquisition, yields `KICKED_OFF`, then yields an update each time the collections written move
on, and ends once the acquisition has returned to idle. `StandardFlyable` runs it in the task
of the status `complete()` returns and cancels it on `unstage()`, so the detector needs no stop
logic beyond its `ensure_stopped`, which `unstage()` always calls, like any flyer.

### An acquisition is a generator too

`DetectorAcquireLogic` started acquiring and waited for idle in two hooks, `start_acquiring` and
`wait_for_idle`, so whatever the first started -- the put-callback status of `Acquire`, the sim
detector's writing task -- had to be held on the logic between them. Cancelling the fly between
the two left it running until `ensure_stopped`, against the rule that a hook never leaves
anything running that it does not wait for.

So they become one async generator, `acquire()`, in the same idiom as `fly`: start, yield once
armed, wait for idle. It owns what it starts (`async with` a status, or cancelling a task in a
`finally`), so closing it at its `yield` ends it. It takes no trigger information: the same
generator serves both kinds of triggering, and only where it is entered differs. An externally
triggered detector arms in `prepare()`, so that it is waiting on its trigger source by the time
`kickoff()` returns; the armed acquisition is held on the logic until `fly` (or `trigger()`)
takes it and runs it to the end, closing it if cancelled. An internally triggered one is
entered by `fly` or `trigger()` itself. One still armed at `unstage()` is closed by
`ensure_stopped`, so its cleanup runs then, not when it is garbage collected after a restage.
A prepared detector cannot be prepared again before it is kicked off or triggered, so no
`prepare()` replaces an armed acquisition.

### One kickoff per prepare

The detector used to be prepared for N events and kicked off a few at a time, with an
`events_to_kickoff` signal saying how many the next kickoff covered and an `is_last_kickoff`
flag deciding when to wait for idle. Nothing in tree drove it: the RunEngine kicks off once
and completes what it kicked off, and every other flyer already worked that way.

Both go, and `kickoff()` requests the whole prepared scan. A second `kickoff()` without an
intervening `prepare()` now raises, from `StandardFlyable`, like any other flyer. A detector
with no data logic flies too, with nothing to collect: it is a valid flyer for its triggering
alone.

### Unbounded providers are the stage's, bounded providers the prepare's

`_FlyCtx` holds the `TriggerInfo` that `prepare()` resolved, the serving bounded providers,
the datakeys and suppressed signals they and the unbounded ones make, the progress baselines and
a count of the `trigger()`s since, and of the events published since the scan started, one
count for all providers. The serving unbounded providers and an armed acquisition
live on the logic for the stage instead, because they outlive a single prepare: a step scan
reuses an open file across its points. A provider carried to the next `prepare()` publishes
from that prepare's baseline. The data verbs — `read`, `describe`, `describe_collect`,
`collect_asset_docs`, `collect_pages`, `get_index` — call the logic with the context, which
`StandardFlyable` keeps after `complete()`, so `collect_while_completing` collects after
`complete()` has finished.

`trigger()` therefore needs no context of its own, beyond a count of triggers on the fly
context. It is part of the state machine all the same: it runs in a `TRIGGERING` state entered
from `PREPARED` and ends in `TRIGGERED`, so `unstage()` cancels it, `kickoff()` cannot overlap
it, and a `kickoff()` after it needs a new `prepare()`. A failed `trigger()` leaves the detector
staged, as a failed `kickoff()` or `complete()` does. It prepares implicitly if nothing has been, then from the second point on calls each
serving bounded logic: it is stopped, asked for a new provider and started, so a finite
buffer holding one event at a time is re-armed, while an unbounded source carries on. The count is what keeps the first point from
being re-armed on top of the `prepare()` that just armed it. The trigger logic is not re-prepared per point, so
a step scan does not repeat its puts.

### One context type, from `prepare_to_fly` to `fly`

`FlyableLogic` threads a single `CtxT` from `prepare_to_fly` to `fly`. What only exists once
flying lives in `fly` itself, since one generator spans `kickoff()` and `complete()`, so no
field of the context waits for kickoff as `| None`. The context may be mutable: the detector
counts its `trigger()`s and notes its baselines on it.

## Consequences

- `StandardDetector` subclasses build a `DetectorLogic` in `logic`;
  `det.add_detector_logics(*logics)` is gone.
- `kickoff()` → `kickoff()` without an intervening `complete()` is rejected, and so is a
  second `kickoff()` after one. The old detector allowed both, because it only checked that
  a prepare context existed.
- `events_to_kickoff` is removed from the public API.
- Calling a verb before `prepare()` says `<name>: prepare() must be called first` instead of
  `Prepare not run`. Before `stage()`, and after `unstage()`, every verb but `stage()` raises,
  for every `StandardFlyable`.
- A detector satisfies neither `WritesStreamAssets` nor `EventPageCollectable` until it has
  been prepared. Everything in the RunEngine that dispatches on them runs after prepare.
- `complete`, `kickoff`, `prepare`, `stage` and `unstage` are no longer implemented on
  `StandardDetector`, nor are the private steps of `StandardFlyable` and `StandardReadable`, so
  a change to the flyer lifecycle reaches detectors too. It adds to them only through the hooks
  they run: one to serve the data and bind the collect verbs when `prepare_to_fly` yields the
  context, and one giving the datakeys that replace registered signals.
- `trigger()` is cancelled by `unstage()`, and `kickoff()` raises while one runs.
- `DetectorAcquireLogic.start_acquiring()` and `wait_for_idle()` are replaced by `acquire()`.
