import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    Array1D,
    BoundedDataLogic,
    EventPageDataProvider,
    SignalR,
    set_and_wait_for_value,
    wait_for_value,
)
from ophyd_async.epics.core import stop_busy_record

from ._io import NDStatsIO, NDStatsTSAcquireMode


@dataclass
class StatsTimeSeriesDataLogic(BoundedDataLogic[EventPageDataProvider]):
    """Data logic publishing an `NDPluginStats` time series as event pages.

    For detectors that write no file: the stats plugin holds each statistic in a
    fixed-length buffer that must be sized before acquisition, so this logic
    raises for a scan with no fixed number of collections, unless the unbounded
    data logics already make all its datakeys. One plugin has one time-series
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
    NDArray callbacks. A detector that writes a file should pull stats into the
    file as NDAttributes instead (see `ADHDFDataLogic`).

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
        await asyncio.gather(
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
        return EventPageDataProvider(
            arrays, self.stats.ts_timestamp, collections_per_event
        )

    async def ensure_stopped(self) -> None:
        await stop_busy_record(self.stats.ts_acquire)
