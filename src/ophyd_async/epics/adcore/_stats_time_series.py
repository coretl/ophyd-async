import asyncio
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ophyd_async.core import (
    Array1D,
    DetectorDataLogic,
    EnableDisable,
    EventPageDataProvider,
    SignalR,
)

from ._io import NDStatsIO, NDStatsTSAcquireMode, plugin_is_enabled


@dataclass
class StatsTimeSeriesDataLogic(DetectorDataLogic[int]):
    """Bounded data logic for an `NDPluginStats` time series.

    For detectors that write no file: the stats plugin holds each statistic in a
    fixed-length buffer that must be sized before acquisition, so this is a
    data logic that emits event pages. One plugin has
    one time-series control (`ts_acquire`, `ts_num_points`) shared across many
    arrays, so one logic covers many arrays with the control embedded.

    The buffer is sized and armed in `start` by writing 1 to
    ``ts_acquire``, which erases the arrays and resets the current point before
    the detector's acquire logic drives the camera; its frames then feed the time
    series via NDArray callbacks. A detector that writes a file should pull stats
    into the file as NDAttributes instead (see `ADHDFDataLogic`).

    :param stats: the stats plugin whose time series to read
    :param stat_signals: datakey suffix to the array signal for each statistic to
        expose; defaults to the ``Total`` series under the bare datakey name
    """

    stats: NDStatsIO
    stat_signals: Sequence[tuple[str, SignalR[Array1D[np.float64]]]] = field(
        default_factory=list
    )
    datakey_suffix: str = ""
    #: Whether to switch the plugin on when starting. Left True, the plugin is
    #: enabled as part of starting. Set False to follow whatever the plugin is
    #: set to instead: a disabled plugin then makes no provider.
    enable_callbacks: bool = True

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
        if not self.enable_callbacks and not await plugin_is_enabled(self.stats):
            return None
        # The datakeys are known from the arrays we were configured with, so
        # nothing has to be armed to describe them
        provider = EventPageDataProvider(
            self._arrays(datakey_name),
            self.stats.ts_current_point,
            self.stats.ts_timestamp,
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
        if self.enable_callbacks:
            coros.append(self.stats.enable_callbacks.set(EnableDisable.ENABLE))
        await asyncio.gather(*coros)
        # Writing 1 to ts_acquire clears the arrays and resets ts_current_point to
        # 0, so the buffer is armed and empty before the detector's frames arrive.
        # This is the data logic performing the erase itself, which is why
        # trigger()'s zero baseline is correct (see ADR 0022).
        await self.stats.ts_acquire.set(True)

    async def stop(self) -> None:
        await self.stats.ts_acquire.set(False)

    def get_hinted_fields(self, datakey_name: str) -> Sequence[str]:
        return list(self._arrays(datakey_name))
