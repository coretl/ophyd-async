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
`init_devices` calls `set_name` on context exit, so the logic cannot be given it when it is
made. `StandardFlyable` passes the Device's name to `prepare_to_fly` instead; the detector
keeps it on the context, for the data providers it starts after yielding, and
`get_hinted_fields()` takes it the same way. The logic holds no reference to the Device.

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

### One name per statistic, and the plugin's switches decide which are used

Suppression only works if every tier calls a quantity the same thing. An areaDetector statistic
reaches the documents in up to three ways: the scalar read once per event, its stats time
series, and an NDAttribute in the HDF file. All three are named after the scalar signal
(`det-stats-total`), because that is the name a user already gets by registering it, and so
the highest tier serving a scan suppresses the others without any configuration. The time
series' `stat_signals` and `datakey_suffix`, and `setup_ndstats_sum`, which named the same
total `det-sum`, are removed.

Which statistics are used is decided by the plugin's own PVs, not by arguments to the data
logics: a statistic is used if the plugin's `EnableCallbacks` and the compute switch of its
group (`ComputeStatistics`, `ComputeCentroid`) are on, and it has a read format on the plugin.
The PVs are what make the IOC produce the data, they are already captured by settings, and a
second set of switches in Python could only disagree with them. So the stats time series has
no `always_enable` and never writes these PVs, and a scalar whose statistic is switched off is
dropped from `read()`, `describe()` and `hints` rather than read as a stale value. The HDF
writer keeps `always_enable`, defaulting to `True`, for compatibility with `ADWriterFactory`: it switches its plugin on at `stage()`, before the recording below is taken, and
writes nothing to `EnableCallbacks` after that.

`AreaDetector` records the plugins it uses once, at `stage()`, after the data logics have had
their say, because plugins are not usually staged themselves. Those are its `plugins`, the
writers of its factories, the NDAttribute sources of its writers and the plugins its factories
take by reference, and every plugin below them, so a stats plugin that is a child of an ROI
plugin is covered. For each it records the switches (`EnableCallbacks`, the compute switches,
the `Use` of each ROI stat channel), the `NDArrayPort`, and the `Name` of each ROI stat channel.
`describe()` and `read()` follow the recording, so they agree for the whole stage. Each
`prepare()` raises if a recorded PV has a different value now, naming the PVs, so a run cannot
be split by someone changing a PV part way through, and nothing is written to a PV the user
owns. The recording is never retaken: staging again records the new values, and `unstage()`
forgets them.

The exception is a switch the detector's own logic writes. A trigger logic switches on the
process plugin it filters with, and the continuous acquisition one its circular buffer, at every
`prepare()`, so recording them would report the detector's own write as a change by someone
else. The plugins a trigger logic holds are therefore left out of the recording. Ownership is by
logic, not by plugin type: a process plugin that only appears in `plugins` is the user's, and is
recorded like any other.

Only a plugin's own `NDArrayPort` is recorded, so a rewired port is caught, but the chain of
plugins upstream of it is not walked to see whether it still reaches the driver. A chain can
cross IOCs over PVA, so its reachability is not something one process can establish, and the
chain as wired at `stage()` is the user's responsibility.

A statistic that is switched off is such a `None`. Neither the stats time series nor the HDF writer hints
a statistic, so the plugin registered as a readable is what hints it, whichever tier serves it.
`AreaDetector` registers every plugin in its `plugins` as a readable, so that is always the case
for a plugin the user gave it, and a user who does not want a plugin read opts it out with
`set_readable_format(plugin, None)`. Data factories take such a plugin by reference, and only
the factories of file writers make a device: `ADDataFactory.stats_time_series(stats)` raises
in `AreaDetector.__init__` if `stats` is not in the detector's device tree, like an NDAttribute
source.

The HDF writer stays free of stats knowledge. `ADDataFactory.hdf(ndattribute_sources=)` lists
the plugins, as objects, whose NDAttributes it stamps: a plugin that implements the
`NDAttributeSource` protocol returns its NDAttributes, which the writer writes to the plugin's
`nd_attributes_file` and turns into datasets directly, each named after the scalar signal it
publishes. Nothing is stamped that is not listed, so two HDF writers stamp the statistics of
their own ROIs and no others. `AreaDetector` raises if a source is listed by two writers or is
not in its device tree. Any other plugin's, and the driver's, NDAttributes are still parsed from
the XML the user set, and keep the names in it, as they may be about something outside the
detector.

An ROI stat plugin is a source too. Each channel whose `Use` is on and which has a read format
on a scalar stamps that scalar, named after it, from the channel's asyn address. Its switches
are `EnableCallbacks` and the `Use` of every channel, and a channel that is not used has its
scalars dropped like a switched-off statistic. The channels are named when the detector is
staged, as `<plugin name>-<Name PV>`, with characters other than letters, digits and underscores
replaced by underscores and the channel number used if the PV is empty, so the scalars have
names the documents accept and say what the channel measures. Two channels that would be named
the same raise, and a `Name` PV that changes after that raises at `prepare()` like a switch.
The plugin is a readable whose channels are its readable children, so a user does not register
each channel. Its time series is published by `ADDataFactory.roistat_time_series(roistat)`
like the stats time series, one array per used channel per scalar with a read format, named
after that scalar, so it suppresses the scalar. It differs only where the plugin does: a
`TSControl` record erases and starts the buffer rather than a busy record, the buffer has no
circular mode, so it is always fixed length, and every channel has its own timestamp array.
All used channels are stamped from the same frame, so the first used channel's timestamps
serve them all. Like `NDStatsIO`, `NDROIStatNIO` has a `net` scalar with no read format, so
its `TSNet` array is published only when a user gives `net` one.

Both plugins count every frame in `TSCurrentPoint` but post their arrays only when the buffer
is full, on Stop or Read, or from a record that asks for a Read about once a second while
acquiring. A page collected between those posts would read arrays shorter than the count, so
each logic gives its `EventPageDataProvider` a refresh that asks for a Read before the arrays
are read: `TSRead` of the stats plugin's embedded time series, or Read on the ROI stat
plugin's `TSControl`. In both plugins a Read only posts the arrays up to the current point,
and does not stop or erase an acquiring buffer. The refresh is a callable on the provider
rather than a step of the data logic, because the provider is what decides when a page is
read. A Read put can complete before the plugin's I/O Intr waveform records have processed,
so the refresh alone does not guarantee that the arrays are up to date. The provider
therefore observes each array and the timestamps until each holds as many collections as the
page needs, for at most `ARRAY_WAIT_TIMEOUT` (longer than the one second between the IOC's
posts), and raises a `TimeoutError` naming the signal if one does not. It slices the values
it observed, never a later get, so a page never has an array shorter than the count it
indexes by. The arrays are waveforms of `NCHANS` elements in both plugins, and a post is cut
to that length, so a scan longer than that could never be served. Each logic reads the
timestamp waveform's `NELM` in `start_provider` and raises at `prepare()` if it is smaller
than the scan. The `TSNumPoints` record is not a reliable check: only the ROI stat plugin's
clamps it to `NCHANS`, the stats plugin's `DRVH` is far larger.

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
`kickoff()` returns; the armed acquisition is held on the context until `fly` (or
`trigger()`) takes it and runs it to the end, closing it if cancelled. An internally
triggered one is entered by `fly` or `trigger()` itself. One still armed at `unstage()` is
closed then, so its cleanup runs then, not when it is garbage collected after a restage;
stopping the detector stays the job of `ensure_stopped`.
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
