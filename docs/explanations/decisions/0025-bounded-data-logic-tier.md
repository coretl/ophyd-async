# 25. Add a bounded `DetectorDataLogic` tier

Date: 2026-07-17

## Status

Accepted

## Context

[ADR 0012](0012-detector-rewrite.md) split data production out into `DetectorDataLogic`.
After [ADR 0024](0024-standard-detector-is-a-standard-readable.md) removed the
single-event tier, one tier was left: a logic prepared for an unbounded number of
collections, publishing stream assets as it wrote. Four open issues bottom out in two
gaps.

**The data logic is not told enough at prepare time.** Issue #1309 wants HDF chunks sized
from the scan: an Xspress at 400 Hz flushing at 2 Hz wants 200 frames per chunk. An
unbounded logic was told nothing about timing, so chunks were read back from the IOC or
hardcoded. What is needed is the frame *period*, since the requirement is a flush *rate*,
and it is known in step scans too. Rejected: a chunk size on a `TriggerInfo` subclass (as
Malcolm did) or computed by the trigger logic puts a data concern in the trigger path,
against [ADR 0012](0012-detector-rewrite.md)'s rule that a logic is handed only what it
needs.

**Detectors that produce data but cannot write files.** Issues #1248 and #888 (a duplicate)
come from scalers and electrometers, which want `StandardDetector` but must emit event
pages, and `StandardDetector` inherited `WritesStreamAssets` unconditionally.

**These are the same gap.** Both own a finite buffer that must be sized before acquisition,
so must be told *how many* collections to expect (a scaler's MCA length, a stats plugin's
`TSNumPoints`), and a finite buffer is naturally read as pages. So one new tier serves both,
and #1309's period rides along. It also lets a detector with no file writer publish a stats
time series (`TSTotal`) in a fly scan, where only the scalar was modelled.

## Decision

### One method that starts and describes

A logic inherits one of `BoundedDataLogic` and `UnboundedDataLogic`, as whether a source
must be told how many collections to take is a property of its type. Its one abstract
`start_provider` arms the writer and returns a provider describing what it writes. The
unbounded logics start first; a bounded one is also told the datakeys they make and the
scan's `num_collections`, publishes the rest, or returns `None` if none are left. No two
providers may make the same datakey.

Rejected:

- **Describe, then start**, so the detector could choose providers before anything was
  written. The provider had to carry what `start` needed, and Odin (frame shape known once
  the file processor runs) and the PandA (`DATASETS` empty until capture) cannot describe
  before writing. Telling a bounded logic what the unbounded providers make removes the
  choice.
- **One `prepare_*` per tier**: it armed hardware to find out what it would get.

Both kinds get the period per collection, not per exposure, because a writer after a chain
that averages exposures sees one frame per collection.

### The tier is the type of the provider returned

There is no declaration of tier: a logic returning a `StreamableDataProvider` publishes
stream assets, one returning a `PageableDataProvider` event pages, and one may return
either. `None` means "not this stage", such as a plugin switched off, and is not warned
about (#1364): a detector may carry more logics than one scan uses.

A pageable provider's readings are derived from its pages rather than a second code path.

### The provider holds the scan's view of the hardware

A provider is given the `collections_written` signal it counts by, as it already was its
`flush_signal`. The split is by phase: the logic owns the lifecycle, and the provider what
one scan reads back.

### The period is resolved in `prepare()`

A `livetime` of 0 means "whatever is set on the detector", so `prepare()` fills it from
`default_trigger_info()` *after* preparing the trigger logic, when the hardware holds the
real values. That reads hardware state, so like the implicit prepare in `trigger()` it is
opt-in (`OPHYD_ASYNC_PRESERVE_DETECTOR_STATE=YES`); otherwise a period of 0 means "use what
is on the hardware".

### HDF chunks are sized from the flush period

A chunk holds one flush period of frames, so a slow detector flushes small chunks and a fast
one packs many.

### Unbounded providers carry over, bounded providers are re-made

- An unbounded provider carries over to the next `prepare()` of a stage, so a step scan does
  not reopen its file per point and a counting source is not restarted. It starts a new
  `StreamResource` in the same file, as a run cannot refer to an earlier run's.
- It carries over only while the name, `collections_per_event`, period and flush period are
  unchanged, as they are baked into a `StreamResource` and `num_frames_chunks` can only be
  set before `capture`. Otherwise it is stopped before its logic starts again.
- It also carries over only if every unbounded provider has written the same whole number of
  events. A `StreamDatum` indexes events in the file, so frames that are not a whole event,
  or that one file has and another lacks, would point the next run's events at the wrong
  frames. Starting again gives each a fresh count.
- Whether a logic serves is decided at the first `prepare()` of a stage, as one asked later
  would count from 0 while a carried provider counts on.
- A bounded logic is restarted at every `prepare()` and every `trigger()` after the first
  since it, never at `kickoff()`: the first point is armed by its `prepare()`, and re-arming
  it would put a spurious stop/start on the arming control. One that served the first point
  and returns `None` later raises, as the scan's datakeys are fixed. Each arm gives a new
  `StreamResource`, so a logic wanting one per scan should be unbounded.

### The progress baseline is read live, and erasing belongs to the data provider

`trigger()` and `kickoff()` read every provider's count after re-arming. Whoever resets a
counter does it in `start_provider` and waits until it reads 0, so the baseline is never
stale. So a control that both clears the buffer and starts counting, like a MeasComp
CTR-08's `EraseStart`, is fired by the data logic, before `acquire()` arms; the acquire
logic keeps the stop control. The logics share one IO device: the split is of concerns, not
of hardware.

### Logic objects fill exactly one role

`DetectorLogic` takes each role by keyword, so an object fills only the role it is passed
as.

### A detector produces stream assets or event pages, never both

The bundler collects one or the other from a device, so a detector whose providers would
make both raises (the exception is in [ADR 0026](0026-detector-is-a-flyable.md)). The kind
is decided per prepare, so one detector class can be either, depending on its arguments.
Changing kind between runs is documented rather than enforced, as a Device cannot see run
boundaries.

## Consequences

- Out-of-tree data logics must be rewritten onto `start_provider`; there is no shim.
- `default_trigger_info()` should return the current `livetime` and `deadtime`, or the HDF
  writer uses the chunk size set on the IOC.
- Detectors that cannot write files become expressible, closing #1248 and #888.
- The areaDetector file writers keep `always_enable`: `True` means the logic owns the
  writer's `EnableCallbacks`, `False` (the default) that the user does, read at the first
  `prepare()` of a stage. Detectors that always enabled the writer, as before, pass `True`.
- A device whose exposure count and buffer size are one PV has it written by both the
  trigger and data logics, which agree only when `exposures_per_collection` is 1: latent, as
  such devices do not average exposures.
