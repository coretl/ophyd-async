# 27. One name per areaDetector statistic, chosen by the plugin's switches

Date: 2026-11-03

## Status

Accepted

## Context

[ADR 0025](0025-bounded-data-logic-tier.md) added a bounded tier so an areaDetector that
writes no file can publish a stats time series, and sized HDF chunks from the flush
period. [ADR 0026](0026-detector-is-a-flyable.md) let the data a detector makes suppress
the readables it replaces. An areaDetector statistic can reach the documents in three
ways: the scalar read once per event, its time series, and an NDAttribute in the HDF file.
Fly scanning with an HDF writer and a time series together was #1364.

## Decision

### Stats go via NDAttributes with a file writer, via the time series without

An HDF writer already pulls NDAttributes into the file, so only a detector that writes no
file needs `TimeSeriesDataLogic`. A plugin has one time series control shared by about
twenty arrays, so one logic covers them all rather than several contending over it. It
serves `NDStatsIO` and `NDROIStatIO` through one inner logic per plugin type, as the two
differ only in how they arm, post and stop.

### HDF chunks are capped at 4 MiB

A chunk holds one flush period of frames, so a fast detector with small frames (an Xspress3
MCA) packs many. The cap keeps a fast detector with large frames near what a filesystem
reads well. 4 MiB is a first guess for GPFS, left as an open question to reviewers.

The period is `exposures_per_collection * (livetime + deadtime)`, so with a `livetime` of 0 a
chunk is sized from the deadtime alone. A period of 0 sizes nothing: the chunk is the
`NumFramesChunks` the IOC holds, or 1 if that is 0.

### One name per statistic, and the plugin's switches decide which are used

Suppression only works if every tier names a quantity the same. A statistic read per event,
as a time series, or as an HDF NDAttribute is named after the scalar signal
(`det-stats-total`), the name a user already gets by registering it, so the highest tier
serving a scan suppresses the others unconfigured. `setup_ndstats_sum` goes.

Which statistics are used is decided by the plugin's PVs: they make the IOC produce the data
and are captured by settings, and switches in Python could only disagree with them.

- `NDPluginBaseIO.get_enabled_params()` returns the `NDParam`s a plugin's own switches have
  on (`ComputeStatistics`, an ROI channel's `Use`), or `None` if they gate none of its read
  signals. Overrides leave `EnableCallbacks` to the callers, which check it once.
- `published_params(plugin)`, the enabled params of an enabled plugin that have a read
  format, is what both the HDF writer's NDAttributes and the time series publish, so each
  tier publishes what the others would. The IO classes hold signals and static facts;
  `AreaDetector` and the data logics decide.
- `AreaDetector`'s `DetectorSuppressLogic` ([ADR 0026](0026-detector-is-a-flyable.md))
  walks the detector's *readable* tree through its `CHILD` formats, not the Device tree,
  so a stats plugin registered under an ROI is found. A disabled plugin hides all its read
  signals, as it produces nothing; an enabled one hides those its params leave out.
- Only a plugin whose switches no logic writes may gate its readables, so the hidden set
  can be read before the trigger logic is prepared. These are the plugins a logic is
  passed by reference, stats and ROI stat plugins, whose switches are only read. The
  process plugin and circular buffer are not `StandardReadable`, so declare nothing to read
  or hide and are not asked.
- The stats time series logic always enables the `TS:` time series plugin embedded in the
  stats plugin it reads, when it arms, as it is the logic's internal machinery. It writes no
  other switch of the stats plugin.
- A plugin's `NDArrayPort` is not checked nor the chain upstream walked: it can cross IOCs
  over PVA, and the wiring is the user's business. The one exception is the circular buffer
  of `ADContAcqAcquireLogic`, which must read directly from the driver, else `stage()` raises.

Only the plugin registered as a readable hints a statistic, whichever tier serves it.

A logic writes only the switches of the plugins it was built to own: the circular buffer
its acquire logic is given, the process plugin its trigger logic is given, its writer, and
the `TS:` sub-plugin. Every other plugin's switches it only reads, to decide what to
publish.

- `plugins` on `AreaDetector` is all the detector's plugins, by attribute name: those the
  caller passes and those the detector creates (the circular buffer, and the process plugin
  of `proc_suffix`, which a detector gives to its trigger logic too). It holds no `None`: a
  detector without a plugin leaves its name out. No logic writes the `EnableCallbacks`,
  `ComputeStatistics` or ROI channel `Use` of a stats or ROI stat plugin.
- The writers a factory creates (HDF, TIFF, JPEG) are the detector's own too. Its logics
  always switch on the circular buffer and the process plugin, and a writer's
  `always_enable` says whether its logic switches it on (`True`) or sits out the scans of a
  stage if the user left it off at the first `prepare()` (`False`, the default). A detector
  with no writer is valid: its data is read from the plugins.
- Every plugin of the detector is an attribute and registered as a readable;
  `set_readable_format(plugin, None)` opts one out. This replaces the per-plugin opt-in of
  [ADR 0024](0024-standard-detector-is-a-standard-readable.md), as a plugin that is not read
  cannot hint the statistics the other tiers publish. The factories are given all of them,
  so the HDF writer parses the NDAttributes of the process plugin.
- Data factories take plugins by reference, which must be in `plugins`: a plugin outside it
  is never connected, so `ADDataFactory.hdf(ndattribute_sources=)` and
  `ADDataFactory.time_series` raise `ValueError` when the detector is made.

The HDF writer stays free of stats knowledge: `ADDataFactory.hdf(ndattribute_sources=)`
lists the plugins it stamps, so two writers stamp only their own ROIs. NDAttributes from the
user's XML keep their names, as they may be about something outside the detector.

An ROI stat plugin is a source and a readable whose used channels are its children. Each
channel is named at `stage()` as `<plugin name>-<Name PV>`, sanitised, or by number if
`Name` is empty, so the names say what it measures. Naming at `stage()` keeps asking for
params free of side effects; a clash raises there.

### Time series pages are read only once the arrays hold them

Both plugins post their arrays only when full, on Stop or Read, or about once a second, so a
page read between posts would be short:

- The provider is given a refresh that requests a Read before each page, posting the arrays
  so far without stopping the buffer. It lives on the provider, which decides when a page is
  read.
- A Read can complete before the waveforms process, so the provider `observe_value`s each
  array until it holds the page, for at most `ARRAY_WAIT_TIMEOUT`.
- The arrays hold `NCHANS` elements, so `start_provider` checks the timestamp waveform's
  `NELM` and raises at `prepare()` for a longer scan. `TSNumPoints` cannot serve: only the
  ROI stat plugin clamps it to `NCHANS`.

## Consequences

- Fly scanning an areaDetector with an HDF writer and a time series works, closing #1364.
- `setup_ndstats_sum` is removed, and its `det-sum` is now `det-stats-total`.
- A statistic is switched on and off with the plugin's PVs, not with arguments to the data
  logics.
