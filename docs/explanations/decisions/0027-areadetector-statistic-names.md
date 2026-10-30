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
the same raise, and a `Name` PV that changes after that raises at `prepare()` like a switch. The plugin is a
readable whose channels are its readable children, so a user does not register each channel. There is no ROI stat time series: a fly scan that needs the channels while the
HDF writer sits out raises at `prepare()`, as a stats plugin only registered as a readable does.

## Consequences

- Fly scanning an areaDetector with an HDF writer and a time series works, closing #1364.
- `setup_ndstats_sum` is removed, and its `det-sum` is now `det-stats-total`.
- A statistic is switched on and off with the plugin's PVs, not with arguments to the data
  logics.
