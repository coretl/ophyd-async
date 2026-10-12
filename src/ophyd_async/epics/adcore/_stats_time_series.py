import asyncio
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    Array1D,
    DetectorDataLogic,
    EnableDisable,
    EventPageDataProvider,
    SignalR,
    set_and_wait_for_value,
    wait_for_value,
)
from ophyd_async.epics.core import stop_busy_record

from ._io import NDStatsIO, NDStatsTSAcquireMode, plugin_is_enabled


@dataclass
class StatsTimeSeriesDataLogic(DetectorDataLogic[int]):
    """Data logic publishing an `NDPluginStats` time series as event pages.

    For detectors that write no file: the stats plugin holds each statistic in a
    fixed-length buffer that must be sized before acquisition, so this logic
    sits out a scan with no fixed number of collections. One plugin has
    one time-series control (`ts_acquire`, `ts_num_points`) shared across many
    arrays, so one logic covers many arrays with the control embedded.

    The buffer is sized and armed in `start`, and re-armed at each step point
    after the first, by writing 1 to ``ts_acquire``, which erases the arrays and
    resets the current point before the detector's acquire logic drives the
    camera; its frames then feed the time series via NDArray callbacks. A
    detector that writes a file should pull stats into the file as NDAttributes
    instead (see `ADHDFDataLogic`).

    :param stats: the stats plugin whose time series to read
    :param stat_signals: datakey suffix to the array signal for each statistic to
        expose; defaults to the ``Total`` series under the bare datakey name
    """

    stats: NDStatsIO
    stat_signals: Sequence[tuple[str, SignalR[Array1D[np.float64]]]] = field(
        default_factory=list
    )
    datakey_suffix: str = ""
    #: True switches the plugin on (EnableCallbacks) when starting. False follows
    #: the plugin's EnableCallbacks PV: if it is off, this logic makes no provider.
    always_enable: bool = True

    def _arrays(self, datakey_name: str) -> dict[str, SignalR[Array1D[np.float64]]]:
        signals = self.stat_signals or [("", self.stats.ts_total)]
        return {datakey_name + suffix: signal for suffix, signal in signals}

    async def make_data_provider(
        self,
        datakey_name: str,
        num_collections: int,
        period: float,
        flush_period: float,
    ) -> tuple[EventPageDataProvider, int] | None:
        # The buffer is sized by count, not rate, so neither period is used.
        if num_collections == 0:
            # A finite buffer cannot serve an unbounded scan
            return None
        if not self.always_enable and not await plugin_is_enabled(self.stats):
            return None
        # The datakeys are known from the arrays we were configured with, so
        # nothing has to be armed to describe them
        provider = EventPageDataProvider(
            self._arrays(datakey_name),
            self.stats.ts_current_point,
            self.stats.ts_timestamp,
            rearm=self._arm,
        )
        return provider, num_collections

    async def start(self, ctx: int) -> None:
        # Size the buffer to the collection count make_data_provider was given,
        # and put it in fixed-length mode, before arming it.
        coros: list[Awaitable] = [
            self.stats.compute_statistics.set(True),
            self.stats.ts_num_points.set(ctx),
            self.stats.ts_acquire_mode.set(NDStatsTSAcquireMode.FIXED_LENGTH),
        ]
        if self.always_enable:
            coros.append(self.stats.enable_callbacks.set(EnableDisable.ENABLE))
        await asyncio.gather(*coros)
        await self._arm()

    async def _arm(self) -> None:
        # Writing 1 to ts_acquire clears the arrays and resets ts_current_point to
        # 0. The put completes when the buffer is full, so do not wait for it.
        await set_and_wait_for_value(
            self.stats.ts_acquire, True, wait_for_set_completion=False
        )
        # The IOC resets the count within the put, but its monitor may arrive
        # after the put's readback, and the detector reads the count next
        await wait_for_value(self.stats.ts_current_point, 0, timeout=DEFAULT_TIMEOUT)

    async def stop(self) -> None:
        await stop_busy_record(self.stats.ts_acquire)

    def get_hinted_fields(self, datakey_name: str) -> Sequence[str]:
        return list(self._arrays(datakey_name))
