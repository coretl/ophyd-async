# 25. Data logics start a provider for each scan

Date: 2026-07-17

## Status

Accepted

## Context

[ADR 0012](0012-detector-rewrite.md) split data production out into `DetectorDataLogic`.
After [ADR 0024](0024-standard-detector-is-a-standard-readable.md) removed the
single-event tier, one tier was left: a logic prepared for an unbounded number of
collections, publishing stream assets as it wrote.

**The data logic is not told enough at prepare time.** Issue #1309 wants HDF chunks sized
from the scan: an Xspress at 400 Hz flushing at 2 Hz wants 200 frames per chunk. An
unbounded logic was told nothing about timing, so chunks were read back from the IOC or
hardcoded. What is needed is the frame *period*, since the requirement is a flush *rate*,
and it is known in step scans too. Rejected: a chunk size on a `TriggerInfo` subclass (as
Malcolm did) or computed by the trigger logic puts a data concern in the trigger path,
against [ADR 0012](0012-detector-rewrite.md)'s rule that a logic is handed only what it
needs.

## Decision

### One method that starts and describes

A logic inherits `UnboundedDataLogic`: its source takes data for as long as it is told to.
Its one abstract `start_provider` arms the writer and returns a provider describing what it
writes. No two providers may make the same datakey.

Rejected:

- **Describe, then start**, so the detector could choose providers before anything was
  written. The provider had to carry what `start` needed, and Odin (frame shape known once
  the file processor runs) and the PandA (`DATASETS` empty until capture) cannot describe
  before writing.
- **One `prepare_*` per tier**: it armed hardware to find out what it would get.

A logic gets the period per collection, not per exposure, because a writer after a chain
that averages exposures sees one frame per collection.

### `None` means "not this stage"

`start_provider` returns a `StreamableDataProvider`, or `None` for "not this stage", such
as a plugin switched off, which is not warned about (#1364): a detector may carry more
logics than one scan uses.

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

### Providers carry over

- A provider carries over to the next `prepare()` of a stage, so a step scan does not reopen
  its file per point and a counting source is not restarted. It starts a new
  `StreamResource` in the same file, as a run cannot refer to an earlier run's.
- It carries over only while the name, `collections_per_event`, period and flush period are
  unchanged, as they are baked into a `StreamResource` and `num_frames_chunks` can only be
  set before `capture`. Otherwise it is stopped before its logic starts again.
- It also carries over only if every provider has written the same whole number of events. A
  `StreamDatum` indexes events in the file, so frames that are not a whole event, or that
  one file has and another lacks, would point the next run's events at the wrong frames.
  Starting again gives each a fresh count.
- Whether a logic serves is decided at the first `prepare()` of a stage, as one asked later
  would count from 0 while a carried provider counts on.

### The progress baseline is read live, and erasing belongs to the data provider

`prepare()` reads every provider's count once they are started, and progress counts from
there, so a source that already holds collections is not stale. Whoever resets a counter
does it in `start_provider` and waits until it reads 0, before `acquire()` arms; the
acquire logic keeps the stop control. The logics share one IO device: the split is of
concerns, not of hardware.

### Logic objects fill exactly one role

`DetectorLogic` takes each role by keyword, so an object fills only the role it is passed
as.

## Consequences

- Out-of-tree data logics must be rewritten onto `start_provider`; there is no shim.
- `default_trigger_info()` should return the current `livetime` and `deadtime`, or the HDF
  writer uses the chunk size set on the IOC.
- The areaDetector file writers keep `always_enable`: `True` means the logic owns the
  writer's `EnableCallbacks`, `False` (the default) that the user does, read at the first
  `prepare()` of a stage. Detectors that always enabled the writer, as before, pass `True`.
