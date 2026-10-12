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
    DetectorAcquireLogic,
    DetectorLogic,
    DetectorTriggerLogic,
    EnableDisable,
    EventPageDataProvider,
    StandardDetector,
    TriggerInfo,
    callback_on_mock_put,
    init_devices,
    non_zero,
    set_mock_put_proceeds,
    set_mock_value,
)
from ophyd_async.epics import adcore
from ophyd_async.epics.adcore import (
    NDStatsTSAcquireMode,
    StatsTimeSeriesDataLogic,
)
from ophyd_async.epics.core import EpicsOptions
from ophyd_async.testing import assert_has_calls


@pytest.fixture
async def stats() -> adcore.NDStatsIO:
    async with init_devices(mock=True):
        plugin = adcore.NDStatsIO("PREFIX:STATS:")
    return plugin


async def test_start_sizes_and_arms_the_buffer(stats: adcore.NDStatsIO):
    logic = StatsTimeSeriesDataLogic(stats)
    made = await logic.make_data_provider(
        "det-stats", num_collections=5, period=0.1, flush_period=0.0
    )
    assert made is not None
    provider, ctx = made
    assert isinstance(provider, EventPageDataProvider)
    # Describing the buffer does not arm it
    assert await stats.ts_num_points.get_value() == 0

    await logic.start(ctx)
    assert await stats.ts_num_points.get_value() == 5
    assert await stats.ts_acquire_mode.get_value() == NDStatsTSAcquireMode.FIXED_LENGTH
    # ts_acquire=1 arms and clears the buffer
    assert await stats.ts_acquire.get_value() is True
    # The datakey defaults to the Total series under the bare name
    datakeys = await provider.make_datakeys(5)
    assert list(datakeys) == ["det-stats"]
    assert datakeys["det-stats"]["shape"] == [5]
    assert logic.get_hinted_fields("det-stats") == ["det-stats"]


async def test_start_leaves_the_progress_counter_at_zero(stats: adcore.NDStatsIO):
    """The count a step point is measured from is 0 once start() returns.

    A buffer that still holds a previous scan's points reports them until it is
    armed; the IOC resets TSCurrentPoint inside the TSAcquire write (ADCore
    NDPluginTimeSeries::writeInt32 -> acquireReset), which is what the fake does.
    """
    set_mock_value(stats.ts_current_point, 5)
    callback_on_mock_put(
        stats.ts_acquire,
        lambda value: set_mock_value(stats.ts_current_point, 0) if value else None,
    )
    logic = StatsTimeSeriesDataLogic(stats)
    made = await logic.make_data_provider(
        "det", num_collections=5, period=0.1, flush_period=0.0
    )
    assert made is not None
    # Describing the buffer does not touch it
    assert await stats.ts_current_point.get_value() == 5

    await logic.start(made[1])

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
    logic = StatsTimeSeriesDataLogic(stats)
    made = await logic.make_data_provider(
        "det", num_collections=5, period=0.1, flush_period=0.0
    )
    assert made is not None
    # The put of 1 to TSAcquire does not complete until the buffer is full
    set_mock_put_proceeds(stats.ts_acquire, False)

    await asyncio.wait_for(logic.start(made[1]), timeout=1)

    assert await stats.ts_acquire.get_value() is True
    set_mock_put_proceeds(stats.ts_acquire, True)
    await logic.stop()
    assert_has_calls(stats.ts_acquire, [call.put(True), call.put(False)])
    assert await stats.ts_acquire.get_value() is False


async def test_unbounded_scan_makes_no_provider(stats: adcore.NDStatsIO):
    """A finite buffer cannot serve an unbounded scan, so it sits it out."""
    logic = StatsTimeSeriesDataLogic(stats)
    assert (
        await logic.make_data_provider(
            "det", num_collections=0, period=0.1, flush_period=0.0
        )
        is None
    )


@pytest.mark.parametrize(
    "always_enable,plugin_enabled,makes_provider",
    [
        # The default switches the plugin on, whatever it was set to
        (True, EnableDisable.DISABLE, True),
        # Following the plugin, a disabled one produces nothing...
        (False, EnableDisable.DISABLE, False),
        # ...and an enabled one still works
        (False, EnableDisable.ENABLE, True),
    ],
)
async def test_follows_the_plugin_when_not_enabling_it(
    stats: adcore.NDStatsIO,
    always_enable: bool,
    plugin_enabled: EnableDisable,
    makes_provider: bool,
):
    set_mock_value(stats.enable_callbacks, plugin_enabled)
    logic = StatsTimeSeriesDataLogic(stats, always_enable=always_enable)

    made = await logic.make_data_provider(
        "det", num_collections=5, period=0.1, flush_period=0.0
    )
    assert (made is not None) is makes_provider

    if made is not None:
        await logic.start(made[1])
        # Only the default reaches out and switches the plugin on
        expected = EnableDisable.ENABLE if always_enable else plugin_enabled
        assert await stats.enable_callbacks.get_value() is expected


async def test_stop_stops_the_time_series(stats: adcore.NDStatsIO):
    logic = StatsTimeSeriesDataLogic(stats)
    set_mock_value(stats.ts_acquire, True)
    await logic.stop()
    assert await stats.ts_acquire.get_value() is False


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

    async def start_acquiring(self):
        set_mock_value(self.stats.ts_total, self.values)
        set_mock_value(self.stats.ts_timestamp, self.times)
        set_mock_value(self.stats.ts_current_point, len(self.values))

    async def wait_for_idle(self): ...

    async def ensure_stopped(self): ...


@pytest.fixture
def stats_detector(stats: adcore.NDStatsIO) -> StandardDetector:
    """A writer-less detector whose only data logic is a stats time series."""
    values = np.array([5.0, 6.0, 7.0, 8.0])
    times = np.array([200.0, 201.0, 202.0, 203.0])
    return DetectorLogic(
        trigger=_JustInternal(),
        acquire=_FillStatsAcquireLogic(stats, values, times),
        data=[StatsTimeSeriesDataLogic(stats)],
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
    assert descriptor["data_keys"]["det"]["shape"] == [4]
    assert "external" not in descriptor["data_keys"]["det"]
    (event,) = docs["event"]
    np.testing.assert_array_equal(event["data"]["det"], [5.0, 6.0, 7.0, 8.0])
    assert event["timestamps"]["det"] == 203.0


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
    np.testing.assert_array_equal(page["data"]["det"], [[5.0], [6.0], [7.0], [8.0]])
    assert docs["stop"][0]["num_events"] == {"primary": 4}


async def test_step_scan_read_derives_single_reading(stats_detector: StandardDetector):
    """A step-scan read derives one reading holding the whole buffer, timed by
    the last point's ts_timestamp."""
    det = stats_detector
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=4))
    await det.trigger()
    reading = await det.read()
    np.testing.assert_array_equal(reading["det"]["value"], [5.0, 6.0, 7.0, 8.0])
    assert reading["det"]["timestamp"] == 203.0
    await det.unstage()
