import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    Array1D,
    BoundedDataLogic,
    EnableDisable,
    EventPageDataProvider,
    SignalR,
    set_and_wait_for_value,
    wait_for_value,
)
from ophyd_async.epics.core import stop_busy_record

from ._io import NDROIStatIO, NDROIStatTSControl, NDStatsIO, NDStatsTSAcquireMode


async def _check_capacity(capacity: SignalR[int], num_collections: int) -> None:
    """Raise if the time series arrays hold fewer points than the scan needs."""
    points = await capacity.get_value()
    if points < num_collections:
        raise ValueError(
            f"{capacity.source}: the scan needs {num_collections} points, "
            f"but the plugin's arrays hold {points}"
        )


@dataclass
class StatsTimeSeriesDataLogic(BoundedDataLogic[EventPageDataProvider]):
    """Data logic publishing an `NDPluginStats` time series as event pages.

    For detectors that write no file: the stats plugin holds each statistic in a
    fixed-length buffer that must be sized before acquisition, so this logic
    sizes it from the scan's number of collections. One plugin has one time-series
    control (`ts_acquire`, `ts_num_points`) shared across many arrays, so one
    logic covers many arrays with the control embedded.

    It publishes the time series of each of `NDStatsIO.get_published_statistics`
    under the name of its scalar signal, so the plugin's EnableCallbacks and
    compute switches decide what is published, and this logic never writes them.
    It hints nothing: a hinted scalar of a plugin registered as a readable stays
    in the hints while its series replaces it.

    The buffer is sized and armed in `start_provider`, which the detector calls
    again at each step point after the first, by writing 1 to ``ts_acquire``,
    which erases the arrays and resets the current point before the detector's
    acquire logic drives the camera; its frames then feed the time series via
    NDArray callbacks. `start_provider` raises `ValueError` if the plugin's
    arrays hold fewer points than the scan needs. It also enables the embedded
    time series plugin, which the IOC starts disabled. Before each page it writes
    1 to ``ts_read``, so the plugin posts its arrays up to the current point. A
    detector that writes a file should pull stats into the file as NDAttributes
    instead (see `ADHDFDataLogic`).

    :param stats: the stats plugin whose time series to read
    """

    stats: NDStatsIO

    def _arrays(self) -> dict[str, SignalR[Array1D[np.float64]]]:
        return {
            stat.scalar.name: stat.time_series
            for stat in self.stats.get_published_statistics()
        }

    def __post_init__(self) -> None:
        self.collections_written = self.stats.ts_current_point

    async def start_provider(
        self,
        name: str,
        existing_datakeys: Sequence[str],
        num_collections: int,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> EventPageDataProvider | None:
        # The buffer is sized by count, not rate, so neither period is used
        arrays = {
            datakey: signal
            for datakey, signal in self._arrays().items()
            if datakey not in existing_datakeys
        }
        if not arrays:
            return None
        # Size the buffer and put it in fixed-length mode before arming it
        await _check_capacity(self.stats.ts_timestamp_nelm, num_collections)
        # The embedded time series plugin ignores the stats until it is enabled
        await asyncio.gather(
            self.stats.ts_plugin.enable_callbacks.set(EnableDisable.ENABLE),
            self.stats.ts_num_points.set(num_collections),
            self.stats.ts_acquire_mode.set(NDStatsTSAcquireMode.FIXED_LENGTH),
        )
        # Writing 1 to ts_acquire clears the arrays and resets ts_current_point to
        # 0. The put completes when the buffer is full, so do not wait for it.
        await set_and_wait_for_value(
            self.stats.ts_acquire, True, wait_for_set_completion=False
        )
        # The IOC resets the count within the put, but its monitor may arrive
        # after the put's readback, and the detector reads the count next
        await wait_for_value(self.stats.ts_current_point, 0, timeout=DEFAULT_TIMEOUT)
        # The plugin posts its arrays about once a second while acquiring, so
        # ask it to post them before each page
        refresh = partial(self.stats.ts_read.set, True)
        return EventPageDataProvider(
            arrays, self.stats.ts_timestamp, collections_per_event, refresh
        )

    async def ensure_stopped(self) -> None:
        await stop_busy_record(self.stats.ts_acquire)


@dataclass
class ROIStatTimeSeriesDataLogic(BoundedDataLogic[EventPageDataProvider]):
    """Data logic publishing an `NDPluginROIStat` time series as event pages.

    The ROI stat counterpart of `StatsTimeSeriesDataLogic`, for detectors that
    write no file. The plugin has one time-series control (`ts_control`,
    `ts_num_points`) shared by every channel, and each channel has its own arrays,
    so one logic covers every channel.

    It publishes the time series of each scalar of `NDROIStatNIO.get_published_scalars`
    of each of `NDROIStatIO.get_computed_channels` under the name of the scalar,
    so the plugin's EnableCallbacks and each channel's Use decide what is
    published, and this logic never writes them. It hints nothing. Each channel
    has its own timestamp array, and the first computed channel's is used.

    The buffer is sized and armed in `start_provider` by writing ERASE_START to
    ``ts_control``, which erases the arrays and resets the current point; the
    plugin stops acquiring by itself when the buffer is full. `start_provider`
    raises `ValueError` if the plugin's arrays hold fewer points than the scan
    needs. Before each page it writes READ to ``ts_control``, so the plugin posts
    its arrays up to the current point.

    :param roistat: the ROI stat plugin whose time series to read
    """

    roistat: NDROIStatIO

    def __post_init__(self) -> None:
        self.collections_written = self.roistat.ts_current_point

    async def start_provider(
        self,
        name: str,
        existing_datakeys: Sequence[str],
        num_collections: int,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> EventPageDataProvider | None:
        # The buffer is sized by count, not rate, so neither period is used
        channels = self.roistat.get_computed_channels()
        arrays = {
            scalar.name: channel.time_series[scalar]
            for channel in channels
            for scalar in channel.get_published_scalars()
            if scalar.name not in existing_datakeys
        }
        if not arrays:
            return None
        # The plugin keeps the current point when resized, so size it first and
        # then erase and start, which resets the current point to 0
        await _check_capacity(channels[0].ts_timestamp_nelm, num_collections)
        await self.roistat.ts_num_points.set(num_collections)
        await self.roistat.ts_control.set(NDROIStatTSControl.ERASE_START)
        # The IOC resets the count within the put, but its monitor may arrive
        # after the put's readback, and the detector reads the count next
        await wait_for_value(self.roistat.ts_current_point, 0, timeout=DEFAULT_TIMEOUT)
        # The plugin posts its arrays about once a second while acquiring, so
        # ask it to post them before each page
        timestamps = channels[0].ts_timestamp
        refresh = partial(self.roistat.ts_control.set, NDROIStatTSControl.READ)
        return EventPageDataProvider(arrays, timestamps, collections_per_event, refresh)

    async def ensure_stopped(self) -> None:
        await self.roistat.ts_control.set(NDROIStatTSControl.STOP)
        await wait_for_value(self.roistat.ts_acquiring, False, timeout=DEFAULT_TIMEOUT)
