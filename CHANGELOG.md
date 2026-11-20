# Changelog

## Unreleased

### Breaking changes

- `StandardMovable.movable_logic` and `StandardFlyable.flyable_logic` are replaced by one `logic` property: rename your `@cached_property`. A Device that moves and flies has one logic inheriting both `MovableLogic` and `FlyableLogic`. For an ephemeral flyer use `FlyableLogic.with_device()`.
- `FlyableLogic` hooks are replaced, and defining an old one raises `TypeError` naming its replacement:
  - `on_prepare(value)` becomes `prepare_to_fly(name, value, previous_ctx)`, an async generator that yields its context first, then optionally `WatcherUpdate`s.
  - `on_kickoff(ctx)` and `on_complete(ctx)` become one `fly(ctx)`, an async generator that yields `KICKED_OFF` once up to speed, then optionally `WatcherUpdate`s.
  - `stop()` becomes `ensure_stopped()`, which must wait until the hardware has stopped, and `on_stage()` becomes `ensure_ready()`.
- `MovableLogic.stop()` becomes `ensure_stopped()`; the old name still works with a `DeprecationWarning`.
- A flyer must be staged before `prepare()`, and cannot be prepared again until its fly (or a detector's `trigger()`) has completed. A verb called out of order raises `TransitionError`. `bp.fly` does not stage, so wrap it in `bpp.stage_wrapper`.
- Staging an already-staged Standard device raises `TransitionError`, as ophyd raises `RedundantStaging`.
- A `set()` ended by `stop()` or by a new `set()` raises `StoppedError`, a `RuntimeError`.
- `SimMotor` is now a flyer, so it must be staged before `prepare()`.
- Renamed `MotorFlyCtx` to `MotorPrepareCtx`, `PmacFlyCtx` and `PmacPrepareContext` to `PmacPrepareCtx`, and `SimMotorMoveLogic` to `SimMotorLogic`.
- `Motor` writes `.STOP` and waits for `.DMOV` on every `unstage()`, even when it looks stopped.
- Needs `typing-extensions>=4.14`.
- `StandardReadable` keeps one format per child, so registering a child again replaces its format. A subclass that added to the private `_describe_config_funcs`, `_read_config_funcs` or `_has_hints` adds a hook to `_describe_configuration_funcs`, `_read_configuration_funcs` or `_hints_funcs` instead.
- The deprecated `ConfigSignal` and `HintedSignal` are removed: use `StandardReadableFormat.CONFIG_SIGNAL` or `HINTED_SIGNAL`.
- `StandardDetector` is now a `StandardReadable`: register the signals it reads or reports as configuration with `set_readable_format()`. `add_config_signals()` is deprecated, and a detector with no hinted fields reports `hints` as `{}`.
- `DetectorTriggerLogic.config_sigs()` is removed: `get_deadtime()` is given every configuration signal of the detector, so register a signal it needs as `CONFIG_SIGNAL`.
- The `prepare_single` data logic tier is removed, with `ReadableDataProvider`, `SignalDataProvider` and `adcore.PluginSignalDataLogic`: register a plugin scalar such as `NDStatsIO.total` as `HINTED_SIGNAL` instead.
- The areaDetector driver and plugin classes declare their own read and configuration signals, so a plugin registered as a `CHILD` of an `AreaDetector` adds them to `read()` and `read_configuration()`.
- `ADAcquireLogic` sets `WaitForPlugins` at `stage()`, from its new `wait_for_plugins` argument (default on), rather than `AreaDetector` hiding it.
- `StandardDetector` is now a `StandardFlyable` whose `logic` is a `DetectorLogic` holding the trigger, acquire and data logics (see ADR 0026): build it in a `logic` `@cached_property` rather than passing logics to `__init__()`. `add_detector_logics()` is removed; use `DetectorLogic.with_device()` for an ad-hoc detector.
- `DetectorAcquireLogic.start_acquiring()` and `wait_for_idle()` become one `acquire()`, an async generator that starts acquiring, yields once armed, then waits until idle. It must clean up what it started if it is closed at its `yield`.
- A data logic inherits `BoundedDataLogic`, told the number of collections in the scan, or `UnboundedDataLogic`, and implements one `start_provider()`, replacing `prepare_unbounded()` (see ADR 0025). `prepare()` raises if two serving data logics make the same datakey.
- `DataProvider`, the new base of `StreamableDataProvider`, has an abstract `start_new_resources(origin)`. `make_stream_docs(collections_written, collections_per_event)` becomes `make_stream_docs(index)` and `make_datakeys(collections_per_event)` becomes `make_datakeys()`. `StreamResourceDataProvider` takes `collections_written` (was `collections_written_signal`) and `collections_per_event`, and loses `last_emitted`.
- A registered signal with the same name as a datakey a serving data provider makes is left out of `read()` and `describe()`, so the provider's data replaces it.
- `kickoff()` during or after a `StandardDetector.trigger()` raises `TransitionError`. After a failed `trigger()`, `unstage()` and `stage()` the detector before using it again.
- `TriggerInfo.number_of_events` must be positive.
- `adcore.ADWriterFactory` is renamed `ADDataFactory`, and the old name is deprecated. It is built with keywords only, `build` replacing `writer_cls`, `writer_suffix` and `writer_name`. Its factories take `always_enable`, default `False`: a plugin whose `EnableCallbacks` is off at the first `prepare()` sits out until the next `stage()`. Pass `always_enable=True` to keep the HDF writer always on.
- `setup_ndstats_sum` is removed: list the stats plugin in `ADDataFactory.hdf(ndattribute_sources=)` instead. Each `NDStatsIO` statistic is named after its scalar signal wherever it appears, so `det-sum` is now `det-stats-total`.
- `AreaDetector` registers every plugin as a `CHILD` readable, so their read, configuration and hinted signals appear in the detector's `read()`, `read_configuration()` and `hints`. Opt a plugin out with `set_readable_format(plugin, None)`.
- `AreaDetector` leaves out of `read()`, `describe()` and `hints` the signals of a plugin or statistic switched off at the first `prepare()` after `stage()`, and a later `prepare()` raises if those switches have changed.
- `NDROIStatIO.stage()` names each channel `<plugin name>-<Name PV>`, which changes the datakey of every ROI stat signal. Unused channels are left out of `read()`, `describe()` and `hints`.
- `adcore.prepare_filter` sets `enable_filter` to False for a single exposure.

### Added

- `KICKED_OFF`, `TransitionError`, `StoppedError` and `Logic`, the base of `MovableLogic` and `FlyableLogic`.
- `watch_readback()` reports the progress of a move from a `FlyableLogic` hook. `SimMotor.prepare()` and `complete()` now report progress, as `Motor` does.
- `cancel_and_wait()` cancels tasks and waits for all of them to end, even through a cancel of its caller, then raises the error of the first task, in the order given, that ended with one. If its caller was cancelled, it raises `CancelledError` instead.
- `plugins_with_proc()` adds the process plugin made from a `proc_suffix` to a detector's `plugins`, and raises `ValueError` if `plugins` already has a `"proc"`. Every areaDetector vendor detector and `ContAcqDetector` uses it, so `"proc"` is reserved: pass `proc_suffix` instead.
- `StandardReadable.set_readable_format()` changes what a child contributes to `read()`, `read_configuration()` and `hints` at runtime, like `Kind` in ophyd v1, and replaces the now deprecated `add_readables()`. `store_settings()` and `apply_settings()` save and restore these changes.
- `BoundedDataLogic`, `UnboundedDataLogic`, `DetectorLogic`, `DataProvider`, `PageableDataProvider` and `EventPageDataProvider` in `ophyd_async.core`. A detector whose providers make event pages publishes them from `collect_pages()`.
- `TriggerInfo.flush_period`, the seconds between flushes of the data being written. `ADHDFDataLogic` sizes its HDF chunks from it (#1309).
- A `StandardDetector` with no data logic can be kicked off.
- `DetectorSuppressLogic`, set as `DetectorLogic.suppress`, names the readables a detector's data hides.
- `adcore.TimeSeriesDataLogic` and `ADDataFactory.time_series(plugin)` publish the time series of an `NDPluginStats` or `NDPluginROIStat` as event pages, for an areaDetector that writes no file.
- `NDParam`, `NDPluginBaseIO.get_enabled_params()`, `NDStatsIO.ts_plugin`, `NDROIStatTSControl`, the `ts_*` time series signals of `NDStatsIO` and `NDROIStatNIO`, and `NDROIStatNIO.net`.
- `DetectorTrigger.EXTERNAL_START`, where one external trigger starts all the exposures, for a trigger logic that implements `prepare_start()`, as `EigerTriggerLogic` does.

### Changed

- Breaking: `WatchableAsyncStatus` and `WatchableAsyncStatus.wrap` accept only async generators. A class that implements only `__aiter__` and `__anext__` is no longer accepted.

### Fixed

- `unstage()` now cancels a running `prepare()` or fly, so nothing it started (PMAC trajectory appends, a PandA `complete()`) is left running.
- A failed or cancelled `prepare()` stops the move to the run-up start, and leaves nothing to kick off.
- `prepare()` progress now comes from the logic, not from watching setpoint writes (#1448).
- `observe_value()` and the `wait_for_value()` and `set_and_wait_*()` helpers unsubscribe before they return, not when garbage collected.
- `StandardReadable.add_children_as_readables()` registers the children it adds in the order they were declared, so `hints`, `read()` and `describe()` no longer vary from run to run.
- `set_and_wait_for_other_value()` cancelled before the first value arrives, such as by `stop()` during a move, no longer leaves its value-watching task running.
- A signal `get_value()`, `set()` or `trigger()` cancelled just as it finishes no longer swallows the cancellation on Python 3.11.
- Leaving an `async with AsyncStatus(...)` block, `set_and_wait_for_other_value()` and closing the sim `BlobAcquireLogic.acquire()` no longer swallow a cancellation of the caller that arrives while they wait for the task they cancelled: they wait for that task to end, then raise the cancellation.
- A `WatchableAsyncStatus` whose watcher raises closes the async generator it drives straight away, rather than when the generator is collected.
- `unstage()` now cancels a running `StandardDetector.trigger()`, and closes an acquisition that an externally triggered `prepare()` armed and nothing ran.
- A data provider carried over to the next `prepare()` no longer includes collections written before it in its first event page or stream datum.
