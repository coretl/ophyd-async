import asyncio
from collections import defaultdict
from typing import get_type_hints
from unittest.mock import call

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np
import pytest
from bluesky import RunEngine

from ophyd_async.core import (
    Array1D,
    DetectorAcquireLogic,
    DetectorLogic,
    DetectorTriggerLogic,
    EnableDisable,
    EventPageDataProvider,
    SignalR,
    StandardDetector,
    TriggerInfo,
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    non_zero,
    set_mock_put_proceeds,
    set_mock_value,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics import adcore
from ophyd_async.epics.adcore import (
    NDStatsTSAcquireMode,
    TimeSeriesDataLogic,
)
from ophyd_async.epics.core import EpicsOptions
from ophyd_async.testing import assert_has_calls


@pytest.fixture
async def stats() -> adcore.NDStatsIO:
    async with init_devices(mock=True):
        plugin = adcore.NDStatsIO("PREFIX:STATS:", name="det-stats")
    # The IOC's arrays hold NCHANS points, 2048 by default
    set_mock_value(plugin.ts_timestamp_nelm, 2048)
    # Switched on, so it publishes its basic statistics
    set_mock_value(plugin.enable_callbacks, EnableDisable.ENABLE)
    set_mock_value(plugin.compute_statistics, True)
    return plugin


# The statistics NDStatsIO gives a read format, so publishes with switches on
PUBLISHED = [
    "det-stats-min_value",
    "det-stats-min_x",
    "det-stats-min_y",
    "det-stats-max_value",
    "det-stats-max_x",
    "det-stats-max_y",
    "det-stats-mean_value",
    "det-stats-total",
]


async def test_start_sizes_and_arms_the_buffer(stats: adcore.NDStatsIO):
    set_mock_value(stats.ts_plugin.enable_callbacks, EnableDisable.DISABLE)
    # Left wrapping by the user, so it would never fill and stop
    set_mock_value(stats.ts_acquire_mode, NDStatsTSAcquireMode.CIRCULAR_BUFFER)
    logic = TimeSeriesDataLogic(stats)
    provider = await logic.start_provider("det-stats", [], 5, 5, 0.1, 0)
    assert isinstance(provider, EventPageDataProvider)
    assert await stats.ts_num_points.get_value() == 5
    assert await stats.ts_acquire_mode.get_value() == NDStatsTSAcquireMode.FIXED_LENGTH
    # ts_acquire=1 arms and clears the buffer
    assert await stats.ts_acquire.get_value() is True
    # The embedded time series plugin is the logic's own, and takes no points
    # while it is disabled
    assert await stats.ts_plugin.enable_callbacks.get_value() == EnableDisable.ENABLE
    # Each series is named after its scalar
    datakeys = await provider.make_datakeys()
    assert list(datakeys) == PUBLISHED
    assert datakeys["det-stats-total"]["shape"] == [5]


@pytest.mark.parametrize("points,raises", [(2048, False), (2049, True)])
async def test_start_raises_for_a_scan_longer_than_the_buffer(
    stats: adcore.NDStatsIO, points: int, raises: bool
):
    logic = TimeSeriesDataLogic(stats)

    start = logic.start_provider("det", [], points, 1, 0.1, 0)

    if raises:
        with pytest.raises(
            ValueError, match="needs 2049 points, but the plugin's arrays hold 2048"
        ):
            await start
    else:
        assert await start is not None


async def test_every_time_series_can_be_published(stats: adcore.NDStatsIO):
    """Every ts_* array but the timestamps is a param's, so can be published."""
    hints = get_type_hints(adcore.NDStatsIO)
    arrays = {
        signal
        for attr, signal in stats.children()
        if attr.startswith("ts_")
        and attr != "ts_timestamp"
        and hints[attr] == SignalR[Array1D[np.float64]]
    }
    # Compute every group and give every scalar a read format
    set_mock_value(stats.compute_centroid, True)
    for attr, scalar in stats.children():
        if f"ts_{attr}" in dict(stats.children()):
            stats.set_readable_format(scalar, Format.UNCACHED_SIGNAL)

    params = await stats.get_enabled_params()

    assert arrays == {param.ts_signal for param in params}


@pytest.mark.parametrize("reset_late", [False, True])
async def test_start_leaves_the_progress_counter_at_zero(
    stats: adcore.NDStatsIO, reset_late: bool
):
    """The count a step point is measured from is 0 once start_provider returns.

    A buffer that still holds a previous scan's points reports them until it is
    armed; the IOC resets TSCurrentPoint inside the TSAcquire write (ADCore
    NDPluginTimeSeries::writeInt32 -> acquireReset), but its monitor may arrive
    after the put's readback.
    """
    set_mock_value(stats.ts_current_point, 5)
    loop = asyncio.get_running_loop()

    def reset(value: bool):
        if value and reset_late:
            loop.call_later(0.01, set_mock_value, stats.ts_current_point, 0)
        elif value:
            set_mock_value(stats.ts_current_point, 0)

    callback_on_mock_put(stats.ts_acquire, reset)
    logic = TimeSeriesDataLogic(stats)

    assert await logic.start_provider("det", [], 5, 1, 0.1, 0) is not None

    assert await stats.ts_current_point.get_value() == 0


def test_ts_acquire_is_declared_as_a_busy_record():
    """TSAcquire completes its put only when the buffer is full, so only a put
    of 1 may wait for completion, as for the other areaDetector busy records."""
    (options,) = [
        meta
        for meta in get_type_hints(adcore.NDStatsIO, include_extras=True)[
            "ts_acquire"
        ].__metadata__
        if isinstance(meta, EpicsOptions)
    ]
    assert options.wait is non_zero


async def test_start_does_not_wait_for_the_buffer_to_fill(stats: adcore.NDStatsIO):
    logic = TimeSeriesDataLogic(stats)
    # The put of 1 to TSAcquire does not complete until the buffer is full
    set_mock_put_proceeds(stats.ts_acquire, False)

    await asyncio.wait_for(logic.start_provider("det", [], 5, 1, 0.1, 0), timeout=1)

    assert await stats.ts_acquire.get_value() is True
    set_mock_put_proceeds(stats.ts_acquire, True)
    await logic.ensure_stopped()
    assert_has_calls(stats.ts_acquire, [call.put(True), call.put(False)])
    assert await stats.ts_acquire.get_value() is False


@pytest.mark.parametrize(
    "plugin_enabled,compute_statistics,compute_centroid,expected_datakeys",
    [
        # Switched on, it publishes every statistic with a read format...
        (EnableDisable.ENABLE, True, False, PUBLISHED),
        # ...including a centroid given one, when its group is computed
        (EnableDisable.ENABLE, True, True, [*PUBLISHED, "det-stats-centroid_x"]),
        (EnableDisable.ENABLE, False, True, ["det-stats-centroid_x"]),
        # Nothing is computed, so it sits out
        (EnableDisable.ENABLE, False, False, None),
        # A disabled plugin computes nothing, so it sits out
        (EnableDisable.DISABLE, True, True, None),
    ],
)
async def test_publishes_what_the_switches_leave_on(
    stats: adcore.NDStatsIO,
    plugin_enabled: EnableDisable,
    compute_statistics: bool,
    compute_centroid: bool,
    expected_datakeys: list[str] | None,
):
    stats.set_readable_format(stats.centroid_x, Format.UNCACHED_SIGNAL)
    set_mock_value(stats.enable_callbacks, plugin_enabled)
    set_mock_value(stats.compute_statistics, compute_statistics)
    set_mock_value(stats.compute_centroid, compute_centroid)
    logic = TimeSeriesDataLogic(stats)

    made = await logic.start_provider("det", [], 5, 1, 0.1, 0)

    if expected_datakeys is None:
        assert made is None
        # Sitting out arms nothing
        assert await stats.ts_acquire.get_value() is False
    else:
        assert made is not None
        assert list(await made.make_datakeys()) == expected_datakeys
    # The switches are the plugin's to decide, never written by this logic
    switches = [
        stats.enable_callbacks,
        stats.compute_statistics,
        stats.compute_centroid,
    ]
    for switch in switches:
        get_mock_put(switch).assert_not_called()


@pytest.mark.parametrize(
    "existing_datakeys,expected_datakeys",
    [
        # Nothing made elsewhere, so it publishes every array
        ([], PUBLISHED),
        # Only the arrays the unbounded providers do not make
        ([*PUBLISHED[1:], "det"], PUBLISHED[:1]),
        # Nothing left to make, so it sits out
        (PUBLISHED, None),
    ],
)
async def test_publishes_only_the_arrays_not_already_made(
    stats: adcore.NDStatsIO,
    existing_datakeys: list[str],
    expected_datakeys: list[str] | None,
):
    logic = TimeSeriesDataLogic(stats)

    made = await logic.start_provider("det", existing_datakeys, 5, 1, 0.1, 0)

    if expected_datakeys is None:
        assert made is None
        # Sitting out arms nothing
        assert await stats.ts_acquire.get_value() is False
    else:
        assert made is not None
        assert list(await made.make_datakeys()) == expected_datakeys


async def test_ensure_stopped_stops_the_time_series(stats: adcore.NDStatsIO):
    logic = TimeSeriesDataLogic(stats)
    set_mock_value(stats.ts_acquire, True)
    await logic.ensure_stopped()
    assert await stats.ts_acquire.get_value() is False


async def test_ensure_stopped_waits_for_the_roi_stat_time_series_to_stop():
    async with init_devices(mock=True):
        roistat = adcore.NDROIStatIO("PREFIX:ROISTAT:", num_channels=1)
    set_mock_value(roistat.ts_acquiring, True)
    loop = asyncio.get_running_loop()

    def stop(value: adcore.NDROIStatTSControl):
        if value == adcore.NDROIStatTSControl.STOP:
            # The plugin reports it has stopped after the put returns
            loop.call_later(0.01, set_mock_value, roistat.ts_acquiring, False)

    callback_on_mock_put(roistat.ts_control, stop)

    await TimeSeriesDataLogic(roistat).ensure_stopped()

    assert await roistat.ts_acquiring.get_value() is False


class _JustInternal(DetectorTriggerLogic):
    async def prepare_internal(self, num: int, livetime: float, deadtime: float): ...

    async def default_trigger_info(self) -> TriggerInfo:
        return TriggerInfo()


class _FillStatsAcquireLogic(DetectorAcquireLogic):
    """Fills the mock time series when acquisition starts."""

    def __init__(self, stats: adcore.NDStatsIO, values: np.ndarray, times: np.ndarray):
        self.stats = stats
        self.values = values
        self.times = times

    async def acquire(self):
        for param in await self.stats.get_enabled_params():
            assert param.ts_signal is not None
            set_mock_value(param.ts_signal, self.values)
        set_mock_value(self.stats.ts_timestamp, self.times)
        set_mock_value(self.stats.ts_current_point, len(self.values))
        yield

    async def ensure_stopped(self): ...


@pytest.fixture
def stats_detector(stats: adcore.NDStatsIO) -> StandardDetector:
    """A writer-less detector whose only data logic is a stats time series."""
    values = np.array([5.0, 6.0, 7.0, 8.0])
    times = np.array([200.0, 201.0, 202.0, 203.0])
    return DetectorLogic(
        trigger=_JustInternal(),
        acquire=_FillStatsAcquireLogic(stats, values, times),
        data=[TimeSeriesDataLogic(stats)],
    ).with_device("det")


def test_step_scan_through_run_engine(RE: RunEngine, stats_detector: StandardDetector):
    """A writer-less detector describes and reads as internal data.

    Driven through the RunEngine so the descriptor its datakeys make is the one
    a real scan would emit, and so has to pass event-model validation.
    """
    det = stats_detector
    docs: dict[str, list] = defaultdict(list)

    @bpp.stage_decorator([det])
    @bpp.run_decorator()
    def plan():
        # prepare() comes after stage(), which resets the detector
        yield from bps.prepare(det, TriggerInfo(collections_per_event=4), wait=True)
        yield from bps.trigger_and_read([det])

    RE(plan(), lambda name, doc: docs[name].append(doc))

    (descriptor,) = docs["descriptor"]
    assert list(descriptor["data_keys"]) == PUBLISHED
    total = descriptor["data_keys"]["det-stats-total"]
    assert total["shape"] == [4]
    assert "external" not in total
    (event,) = docs["event"]
    np.testing.assert_array_equal(
        event["data"]["det-stats-total"], [5.0, 6.0, 7.0, 8.0]
    )
    assert event["timestamps"]["det-stats-total"] == 203.0


def test_fly_scan_emits_event_pages(RE: RunEngine, stats_detector: StandardDetector):
    """A writer-less detector collects its buffer as event pages."""
    det = stats_detector
    docs: dict[str, list] = defaultdict(list)

    @bpp.stage_decorator([det])
    @bpp.run_decorator()
    def plan():
        yield from bps.prepare(det, TriggerInfo(number_of_events=4), wait=True)
        yield from bps.declare_stream(det, name="primary")
        yield from bps.kickoff(det, wait=True)
        yield from bps.collect_while_completing(
            flyers=[det], dets=[det], flush_period=0.05
        )

    RE(plan(), lambda name, doc: docs[name].append(doc))

    (page,) = docs["event_page"]
    np.testing.assert_array_equal(
        page["data"]["det-stats-total"], [[5.0], [6.0], [7.0], [8.0]]
    )
    assert docs["stop"][0]["num_events"] == {"primary": 4}


async def test_step_scan_read_derives_single_reading(stats_detector: StandardDetector):
    """A step-scan read derives one reading holding the whole buffer, timed by
    the last point's ts_timestamp."""
    det = stats_detector
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=4))
    await det.trigger()
    reading = await det.read()
    total = reading["det-stats-total"]
    np.testing.assert_array_equal(total["value"], [5.0, 6.0, 7.0, 8.0])
    assert total["timestamp"] == 203.0
    await det.unstage()
