# 23. Add a bounded `DetectorDataLogic` tier

Date: 2026-07-17

## Status

Accepted

## Context

[](./0012-detector-rewrite.md) split data production out into `DetectorDataLogic`. After
[](./0025-runtime-readable-format.md) removed the single-event tier, one tier was left: a
logic prepared for an unbounded number of collections, feeding `collect_asset_docs()` and
`describe_collect()` and doing its own writes as it went.

Four open issues bottom out in two gaps.

### The data logic is not told enough at prepare time

Issue #1309 wants chunk size and flush rate configured from the scan parameters. An Xspress
running at 400 Hz with a 2 Hz flush rate wants 200 frames per chunk; the PandA wants a flush
period. Neither is expressible, because an unbounded logic is told nothing about timing:
`ADHDFDataLogic` reads `num_frames_chunks` back from the IOC and forces it to 1 if unset, the
PandA hardcodes `chunk_shape=(1024,)` behind a TODO, and Odin hardcodes `(1, *data_shape)`.

The parameter actually needed is the frame **period** (`livetime + deadtime`), not the number
of frames, since the requirement is a flush *rate* — and the period is known for step scans as
well as fly scans, where the total frame count is not.

Two workarounds were rejected: subclassing `TriggerInfo` and `prepare()` to carry a chunk size
(as Malcolm did) puts a data-writing concern in the trigger path, against
[](./0012-detector-rewrite.md)'s rule that a logic is handed only the fields it needs;
computing it in the `DetectorTriggerLogic` puts a data concern in the trigger logic.

### Detectors that produce data but cannot write files

Issues #1248 and #888 come from a Struck SIS3820 scaler in a VME crate, a MeasComp CTR-08, a
TetrAmm electrometer, an electron analyser. They want the trigger, arm, prepare and stage
machinery of `StandardDetector` but must emit event pages, and `StandardDetector` inherited
`WritesStreamAssets` unconditionally — the bundler calls `collect_pages()` only on a device
that is *not* one. (#888 asks for the writer to be optional, which the rewrite already did;
its real request is #1248, so it is a duplicate.)

These are not areaDetectors, and they put more pressure on the trigger/acquire/data split. A
CTR-08 has no driver/plugin split in the hardware: `MCS:EraseStart` both clears the arrays and
starts counting, and `MCS:NuseAll` is both the exposure count and the buffer size, where an
areaDetector acquires with `driver.acquire` and captures with a separate `TSAcquire` or HDF
`capture`.

### These are the same gap

Both groups must be told *how many* collections to expect at prepare time, because they own a
finite buffer that has to be sized before acquisition: the scaler's MCA array length *is* the
frame count, and the stats plugin's `TSNumPoints` must be set before the time series starts. A
device with a finite buffer also naturally produces its data as pages at the end of an event
rather than as stream datums as it goes.

So one new tier serves both, and #1309's period rides along with it.

### The areaDetector stats duality

An `NDPluginStats` exposes each statistic twice: as a scalar (`Total_RBV`) and as a time
series array (`TSTotal`). Only the scalar was modelled, as a registered signal read once per
event, so a detector with no file writer had no way to produce a value per collection for a
fly scan.

## Decision

### One method that starts and describes

```python
ProviderT = TypeVar("ProviderT", bound=DataProvider)

class DetectorDataLogic(Generic[ProviderT]):
    def get_hinted_fields(self, name: str) -> Sequence[str]: ...

class UnboundedDataLogic(DetectorDataLogic[ProviderT]):
    async def start_provider(
        self, name: str, collections_per_event: int,
        collection_period: float, flush_period: float,
    ) -> ProviderT | None: ...

class BoundedDataLogic(DetectorDataLogic[ProviderT]):
    async def start_provider(
        self, name: str, existing_datakeys: Sequence[str], num_collections: int,
        collections_per_event: int, collection_period: float, flush_period: float,
    ) -> ProviderT | None: ...
```

`name` is the detector name: a logic that needs a suffix on its datakeys, such as
`ADHDFDataLogic`'s `datakey_suffix`, applies it itself, as it does in `get_hinted_fields(name)`.

Whether a source must be told how many collections to take is a property of the logic's
*type*, so a logic inherits from exactly one of `BoundedDataLogic` and `UnboundedDataLogic`
and implements the one abstract method of that class.

`start_provider` does all the work: it works out the path, sets up and arms the writer, and
returns a provider describing what is being written. Returning `None` sits the logic out, and
it decides that without writing anything. At the first `prepare()` of a stage the detector
starts every unbounded logic, then every bounded one, handing each bounded logic the datakeys
the unbounded providers make. The bounded provider must make none of them: it publishes the
rest, or the logic returns `None` if nothing is left. The logics that return a provider serve
the stage until `unstage()`, and no two providers may make the same datakey.

Describing and starting were split at first, into a `make_*_provider` that described without
writing and a `start` handed the provider back, so the detector could choose between providers
before anything was written. That made the provider carry what `start` needed (a `PathInfo`, a
collection count) and was a poor fit for the logics that cannot describe their data until they
are writing: Odin, whose frame shape is only readable once the file processor runs, and the
PandA on PandABlocks-ioc, whose `DATASETS` table is empty until capture starts. Telling a bounded
logic what the unbounded providers already make removes the choice, so nothing needs to be
described before it is started. A single `prepare_*` per tier was tried before that, and
discarded for the same reason: it armed hardware to find out what it would get.

`num_collections` is the **total** for the scan (`TriggerInfo.number_of_collections`), and
both kinds are told the collection period, `exposures_per_collection * (livetime + deadtime)`,
so either can size a chunk or a buffer against it: a writer after a processing chain that
averages exposures sees one frame per collection.

### The tier is the type of the provider returned

There is no declaration of which tier a logic implements and no override detection: a logic
returning a [](#StreamableDataProvider) publishes stream assets for this scan, one returning a
[](#PageableDataProvider) publishes event pages, and one may return either depending on what it
is asked for. Whether a logic needs a fixed number of collections is its type, bounded or
unbounded, not a publication tier. Returning `None` means "not this stage", for example a plugin
that is switched off. It is neither an error nor warned about, as
agreed in #1364: the point is that a detector may carry more logics than one scan uses.

### `PageableDataProvider`, with readings derived from pages

The pageable tier returns one provider type, which emits pages. Its
`make_page(collections_from, collections_to, collections_per_event)` is stateless: the detector
tracks what it has published. Readings are derived from pages rather than being a separate
code path: `read()` asks each provider for the current step point's event only, which extracts
to a single reading, so a second `read()` gives the same answer.
`collections_per_event` appears in the datakey shape exactly as for a
`StreamableDataProvider`.

### The period is resolved in `prepare()`

`TriggerInfo.livetime` of 0 means "whatever is currently set on the detector", so the period
cannot simply be forwarded: an explicit `prepare(TriggerInfo(number_of_events=10))` would hand
the data logic a period of 0. Instead `DetectorTriggerLogic.default_trigger_info()` is extended
to return the current `livetime` and `deadtime`, and `prepare()` fills a zero `livetime` in
from it *after* preparing the trigger logic, which [](./0012-detector-rewrite.md) already
requires to come first, so the hardware holds the real values by then. This reads hardware
state, so like the implicit prepare in `trigger()` it happens only when
`OPHYD_ASYNC_PRESERVE_DETECTOR_STATE=YES`; otherwise a zero `livetime` stays 0 and data logics
get a period of 0, meaning "use what is on the hardware".

### Unbounded providers carry over, bounded providers are re-made

Within one stage, an unbounded provider that is serving is carried over to the next
`prepare()` without its logic being asked again, so a step scan does not reopen its file on
every point and a source that keeps counting is not restarted. It does start a new
`StreamResource` in the same file, its indices carrying on, as a run cannot refer to the
resource of an earlier one. It carries over while the
datakey name, `collections_per_event`, period and flush period are unchanged, because they are
baked into a `StreamResource` when it is made (the shape, the chunking) and areaDetector's
`num_frames_chunks` can only be set before `capture` starts. When any of them changes, the old
provider is stopped *before* its logic is started again, so a logic's `start_provider` never
writes while its old provider runs. A logic that returned `None` at the first `prepare()` of the
stage is not asked again until the next stage: it would count from 0 while a carried provider
counts on. So whether a logic serves is decided once, at the first `prepare()`; the file
writers with `always_enable=False` read `EnableCallbacks` there. The detector counts the events published since a scan started, once for all its
providers, so a carried provider's first page or stream datum of the next `prepare()` starts at
that scan's baseline, not at what the last one left unpublished.

`unstage()` stops everything, so a change to Settings (a plugin's `EnableCallbacks` switched
off, say) takes effect at the next `stage()`, not between the `prepare()` calls of one stage.
A carried writer whose plugin is switched off by hand mid-stage is not switched back on, as
Settings changes belong before `stage()`.

A bounded logic is started again on every `prepare()`, and on every `trigger()` after the first
one since that `prepare()` it is stopped and started again: a finite buffer is re-armed that
way, and an unbounded source keeps counting. One that served the first
point and returns `None` at a later one raises, as the scan's datakeys are already fixed. `kickoff()` never does
this. That is exactly the set of places a buffer should and should not be re-armed: the first
point of a step scan is already armed by the `prepare()` in front of it, and re-arming it
again would erase an empty buffer, putting a spurious stop/start on the arming control before
a frame had arrived.

A bounded streamable logic therefore gives one `StreamResource` per arm, which is one per step
point. A logic that wants one resource per scan should be an `UnboundedDataLogic`.

### The progress baseline is read live

`trigger()` reads every provider's starting progress after re-arming the bounded logics, and
`kickoff()` reads it too, so a source that keeps counting is measured from where it is. There is
one baseline per bound group: bounded providers are in step with each other (they start at 0),
and unbounded providers are in step with each other (they start wherever their source is), and
progress is reported in one frame. A stats time series is erased by the `TSAcquire` write in its
`start_provider`, which waits for `TSCurrentPoint` to read 0, so its baseline is 0. Whoever
resets a provider's collections-written counter does it in its logic's `start_provider`, and
waits until it reads 0 before returning. A CTR-08-style logic therefore does `EraseStart` in its
data provider, and its acquire logic does not erase; the baseline is then never stale.

### Erase controls belong to the data provider

Where one control both starts acquisition and clears the data buffer, the data logic fires it
in `start_provider`, because the erase is what resets the counter the detector reads. The
`DetectorAcquireLogic` keeps the stop control and the idle status, and does not erase. A data
logic's `start_provider` runs before `acquire()` arms, so the buffer is sized before the
control fires.

The trigger/acquire/data split therefore survives these devices, with the three logics sharing
one IO device exactly as the areaDetector logics share a `driver`. The split is of concerns,
not of hardware.

### Logic objects fill exactly one role

`DetectorLogic(trigger=..., acquire=..., data=(...))` takes each role by keyword, so an object
fills the role it is passed as and no other.

### A detector produces stream assets or event pages, never both

The bundler collects stream assets or event pages from a device, not both, so a detector
whose logics would produce both raises. How the
applicable verb is exposed, and the one case where the two may be carried together, are in
[](./0024-detector-is-a-flyable.md).

Which kind a detector produces is decided per prepare, so a single detector class can be either
depending on its `__init__` arguments — one `AravisDetector`, configured for files or for pages
— which was the original requirement. That it can also change *between runs*, when a file
writer is switched off and a stats time series takes over, is documented rather than enforced:
a run's descriptor is emitted at its start, so switching part way through would leave
`describe_collect()` and the documents disagreeing, and a Device cannot see run boundaries.

### Stats with a file writer go via NDAttributes

An areaDetector with an HDF writer does not need the stats time series: `ADHDFDataLogic`
already pulls NDAttribute datasets into the file, which is areaDetector's own mechanism for
exactly this. So registering the scalar with `set_readable_format` remains right for a
detector that writes files and wants a stats value in a step scan, and a stats time series is
right for one that writes none.

### One stats data logic for many arrays

An `NDPluginStats` has one time series control (`TSNumPoints`, `TSAcquire`, `TSAcquireMode`)
shared across roughly twenty arrays, so one stats data logic covers *many* arrays with the
control embedded, rather than one logic per array with a shared control object. This follows
the existing grain — a provider already emits many datakeys, as `ADHDFDataLogic` does for the
main dataset plus its NDAttributes — and avoids the coordination problem of several logics
contending over one `TSAcquire`. Which statistics are wanted becomes a constructor argument.

## Consequences

`DetectorDataLogic` implementations must be rewritten onto `BoundedDataLogic` or
`UnboundedDataLogic` and their `start_provider`, and take the new `collection_period` and
`flush_period` arguments. This is a breaking change for out-of-tree data logics; the library
is in alpha and no shim is provided. In-tree, `ADHDFDataLogic` sizes its chunk from
`TriggerInfo.flush_period` and the rate, and `PandaHDFDataLogic` sets its flush period signal
from the same field, replacing the TODOs that stood in for both.

`DetectorTriggerLogic.default_trigger_info()` implementations should return the current
`livetime` and `deadtime` alongside the frame count. Those that do not still work, but their
detectors fall back to reading the chunk size back from hardware.

Detectors that cannot write files become expressible: a data logic returning a
`PageableDataProvider` gives them `collect_pages()` and the whole of `StandardDetector`
besides. This closes #1248 and #888. Fly scanning an areaDetector that carries an HDF writer
plus a finite-buffer data logic starts working, closing #1364.

A data logic may decline a scan for reasons of its own: `ADHDFDataLogic` and
`StatsTimeSeriesDataLogic` take an `always_enable` flag, defaulting to today's behaviour of
switching the plugin on, which when `False` makes them follow whatever the plugin is set to
(at the first `prepare()` of a stage) and produce nothing when it is off.

A device whose exposure count and buffer size are the same PV has it written twice: by the
trigger logic as `number_of_exposures`, and by the data logic as `num_collections`. The two
agree only when `exposures_per_collection` is 1. This is latent rather than live, because such
devices do not average exposures and so cannot ask for anything else, but a detector that could
do both would need its data logic to read the value back rather than set it.
