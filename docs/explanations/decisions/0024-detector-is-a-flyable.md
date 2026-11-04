# 24. StandardDetector is a StandardFlyable

Date: 2026-08-28

## Status

Accepted

## Context

`StandardDetector` re-implemented `prepare`, `kickoff` and `complete` and threaded its own
state between them, while [](#StandardFlyable) already did that for every other flyer. The
two drifted apart: the detector guarded its stages by checking a context for `None`, so a
verb called too early failed with `AttributeError: 'NoneType' object has no attribute ...`.

`StandardFlyable` requires a single [](#FlyableLogic) in a `logic` property
([](./0021-single-logic-property.md)), and a detector had three logic objects registered by
`add_detector_logics()`.

## Decision

### One `DetectorLogic` holds the trigger, acquire and data logics

[](#DetectorLogic) is what `logic` returns, and the orchestration (preparing the trigger
logic, making the data providers, arming, waiting) moves into it. It is a frozen dataclass
with no back-reference to its Device, so [](./0017-standard-movable.md) holds unqualified.

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
([](./0023-bounded-data-logic-tier.md)). The bluesky bundler picks which to call from the
*static* type, so declaring both would silently drop the pages, and declaring neither could
not fly. So `StandardDetector.prepare()` overrides the flyer's and, when it succeeds, binds
the verb `DetectorLogic.produces` names as an instance attribute and removes the other. A
step scan restarts its bounded providers at each point, but they must make the same
datakeys, so it keeps the verb it was prepared with.

This is a bundler limitation: a device knows what it produced, and deciding from that is a
few lines in `RunBundler.collect`. Once released, both verbs become plain methods and the
override goes.

### Pageable and streamable data mix when the stream assets cover them

[](./0023-bounded-data-logic-tier.md) made carrying both kinds of data logic an error, but
it is useful: an areaDetector stats total can go into the HDF file as an NDAttribute *and*
be read from the plugin's buffer. So:

- A bounded logic is started after the unbounded ones and leaves out the datakeys they make.
  If they make all of its datakeys, it sits the scan out, unarmed.
- A registered signal whose name a provider key has is dropped from `read()` and
  `describe()` while prepared, and keeps its hint, since the replacement has its name.
- The provider key must have the signal's dtype and shape, optionally with a leading
  `collections_per_event` dimension. Anything else is a `ValueError` at `describe()`: two
  different things share a name, and one must be renamed.
- A registered signal that nothing suppresses is read once per event.

This is a general `StandardReadable` hook, `_suppressed_datakeys_funcs`: datakey name to
`None` or a replacing `DataKey`, matched by name, so a signal is found wherever it is
registered. `None` also drops the hint: it is for a signal that is switched off.

### One name per statistic, and the plugin's switches decide which are used

Suppression only works if every tier names a quantity the same. An areaDetector statistic
can be a scalar read per event, a stats time series, or an HDF NDAttribute, and all three
are named after the scalar signal (`det-stats-total`), the name a user already gets by
registering it, so the highest tier serving a scan suppresses the others unconfigured. The
time series' `stat_signals` and `datakey_suffix`, and `setup_ndstats_sum`, go.

Which statistics are used is decided by the plugin's PVs: they make the IOC produce the
data and are captured by settings, and switches in Python could only disagree with them.

- A statistic is used if `EnableCallbacks` and its group's compute switch are on and it has
  a read format. A switched-off one is suppressed with `None`, not read as a stale value.
- The time series never write these PVs. The HDF writer keeps `always_enable` (default
  `True`) for `ADWriterFactory` compatibility, enabling its plugin at `stage()`.
- At `stage()` `AreaDetector` records the switches, `NDArrayPort` and ROI channel `Name` of
  every plugin it uses, down the tree, as plugins are not staged themselves. `describe()`
  and `read()` follow the recording, and `prepare()` raises naming any recorded PV that has
  changed, so a run is never split part way and no PV the user owns is written.
- Plugins a trigger logic holds are not recorded: it switches them on at every `prepare()`.
  Ownership is by logic, not type, so a process plugin only in `plugins` is recorded.
- The chain upstream of each `NDArrayPort` is not walked: it can cross IOCs over PVA, so one
  process cannot establish it, and the chain wired at `stage()` is the user's business.

Only the plugin registered as a readable hints a statistic, whichever tier serves it.
`AreaDetector` registers every plugin in `plugins` as a readable;
`set_readable_format(plugin, None)` opts one out. Data factories take plugins by reference
and raise in `AreaDetector.__init__` if one is not in the detector's tree.

The HDF writer stays free of stats knowledge: `ADDataFactory.hdf(ndattribute_sources=)`
lists `NDAttributeSource` plugins, each of which returns its NDAttributes, named after the
scalars they publish. Only listed plugins are stamped, so two writers stamp only their own
ROIs. NDAttributes from the user's XML keep their names, as they may be about something
outside the detector.

An ROI stat plugin is a source and a readable whose used channels are its children. Each
channel is named at `stage()` as `<plugin name>-<Name PV>`, sanitised, or by number if
`Name` is empty, so the names say what it measures; a clash raises, and a changed `Name`
raises at `prepare()` like a switch. `ADDataFactory.roistat_time_series(roistat)`
publishes one array per used channel per scalar, like the stats time series.

### Time series pages are read only once the arrays hold them

Both plugins post their arrays only when full, on Stop or Read, or about once a second, so a
page read between posts would see arrays shorter than its count:

- The provider is given a refresh that requests a Read before each page, posting the arrays
  so far without stopping the buffer. It lives on the provider, which decides when a page is
  read.
- A Read can complete before the waveforms process, so the provider `observe_value`s each
  array until it holds the page, for at most `ARRAY_WAIT_TIMEOUT`, and slices what it
  observed.
- The arrays hold `NCHANS` elements, so `start_provider` checks the timestamp waveform's
  `NELM` and raises at `prepare()` for a longer scan. `TSNumPoints` cannot serve: only the
  ROI stat plugin clamps it to `NCHANS`.

### The detector flies with one `fly()` generator

`FlyableLogic.fly` is one async generator ([](./0022-fly-generator-and-ensure-stopped.md)).
A detector's progress is collections written out of collections requested, so
`prepare_to_fly` notes what the unbounded providers have written
(`ctx.unbounded_start_collections`), and `fly` arms an internally triggered acquisition,
yields `KICKED_OFF`, yields an update as the count moves on, and returns once idle.
`unstage()` cancels it, so no stop logic is needed beyond `ensure_stopped`.

A fly scan collects only what the data logics make: registered readables are read only by
`trigger()`.

### An acquisition is a generator too

`DetectorAcquireLogic.start_acquiring` and `wait_for_idle` had to hold what the first
started on the logic, and cancelling between them left it running. They become one async
generator, `acquire()`: start, yield once armed, wait for idle. It owns what it starts, so
closing it at its `yield` ends it, and it needs no trigger information:

- An externally triggered detector arms in `prepare()`, so it is waiting on its trigger
  source when `kickoff()` returns, and the context holds it until `fly` or `trigger()`
  takes it. It cannot be prepared again before then, so nothing replaces it.
- An internally triggered one is entered by `fly` or `trigger()` itself.
- One still armed at `unstage()` is closed then, not at garbage collection; stopping the
  detector stays the job of `ensure_stopped`.

### One kickoff per prepare

The detector was prepared for N events and kicked off a few at a time (`events_to_kickoff`,
`is_last_kickoff`). Nothing drove that: the RunEngine kicks off once, as for every other
flyer. Both go; `kickoff()` requests the whole scan, and a second needs a new `prepare()`. A
detector with no data logic is still a valid flyer for its triggering.

### Unbounded providers carry over between prepares, bounded ones are restarted

One `DetectorCtx` threads from `prepare_to_fly` to `fly`; what exists only once flying lives
in `fly`, so no field waits for kickoff as `| None`. The logic holds no state, and the next
`prepare()` gets the last context as `previous_ctx`:

- Unbounded providers carry over if made for the same name, collections per event, period
  and flush period, because a step scan reuses an open file across its points.
- Bounded providers are restarted at every `prepare()` and every step point after the
  first, so a buffer holding one event is re-armed. The trigger logic is not re-prepared
  per point.
- `StandardFlyable` keeps the context after `complete()`, so `collect_while_completing`
  collects after `complete()` has finished.

`trigger()` runs in a `TRIGGERING` state, so `unstage()` cancels it, `kickoff()` cannot
overlap it, and a failed one leaves the detector staged. It prepares implicitly if nothing
has been.

## Consequences

- A verb called out of order raises a `TransitionError` naming it and the state
  (`<name>: describe not allowed from STAGED, allowed from: ...`), not `Prepare not run`,
  and a second `kickoff()` without a `prepare()` is rejected.
- A detector satisfies neither `WritesStreamAssets` nor `EventPageCollectable` until
  prepared; the RunEngine only dispatches on them after prepare.
- `StandardDetector` no longer implements `complete`, `kickoff`, `stage` or `unstage`, so
  lifecycle changes reach detectors too. It adds `__on_*` hooks for `read()`, `describe()`,
  `hints`, suppression and `unstage()`, and overrides `prepare()` to bind the collect verb.
- `add_detector_logics()`, `events_to_kickoff`, `start_acquiring()` and `wait_for_idle()`
  are gone.
