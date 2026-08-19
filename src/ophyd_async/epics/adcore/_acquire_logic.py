import asyncio

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    AsyncStatus,
    DetectorAcquireLogic,
    EnableDisable,
    SignalR,
    set_and_wait_for_other_value,
    set_and_wait_for_value,
    wait_for_value,
)
from ophyd_async.epics.core import stop_busy_record, wait_for_good_state

from ._io import ADBaseIO, ADState, NDCBFlushOnSoftTrgMode, NDCircularBuffIO


class ADAcquireLogic(DetectorAcquireLogic):
    """Arm and disarm an areaDetector driver.

    :param driver: The areaDetector driver to arm.
    :param driver_armed_signal:
        The signal to watch for the driver having armed, if it is not `acquire`.
    :param wait_for_plugins:
        Whether the detector should wait for its plugins to drain before it
        reports acquisition as complete. `WaitForPlugins` makes the driver's
        `AcquireBusy` wait for the number of queued arrays to reach zero, and
        `wait_for_idle` blocks on the `acquire` put callback, so turning it on
        is what makes a step scan read plugin scalars for the frame it just
        took rather than the one before. Defaults to on, since racing is
        silent and the wait costs nothing when a file writer is the slowest
        thing in the chain anyway. Turn it off for a detector whose plugin
        chain must not gate acquisition.
    :param timeout: How long to wait for the driver to arm and to return to idle.
    """

    def __init__(
        self,
        driver: ADBaseIO,
        driver_armed_signal: SignalR[bool] | None = None,
        wait_for_plugins: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.driver = driver
        if driver_armed_signal is not None:
            self.driver_armed_signal = driver_armed_signal
        else:
            self.driver_armed_signal = driver.acquire
        self.wait_for_plugins = wait_for_plugins
        self.timeout = timeout
        self.acquire_status: AsyncStatus | None = None

    async def ensure_ready(self):
        # Stop first: WaitForPlugins is latched and has no _RBV, so set it once
        # per scan while the driver is known to be idle rather than on the
        # latency path before every arm. Always written, never left at whatever
        # the IOC happened to boot with, so a scan does not depend on who
        # touched the PV last.
        await self.ensure_stopped()
        await self.driver.wait_for_plugins.set(self.wait_for_plugins)

    async def start_acquiring(self):
        self.acquire_status = await set_and_wait_for_other_value(
            set_signal=self.driver.acquire,
            set_value=True,
            match_signal=self.driver_armed_signal,
            match_value=True,
            wait_for_set_completion=False,
            timeout=self.timeout,
        )

    async def wait_for_idle(self):
        if self.acquire_status:
            await self.acquire_status
        await wait_for_good_state(
            self.driver.detector_state,
            {ADState.IDLE, ADState.ABORTED},
            timeout=self.timeout,
        )

    async def ensure_stopped(self):
        await stop_busy_record(self.driver.acquire)


class ADContAcqAcquireLogic(DetectorAcquireLogic):
    """Gate continuous acquisition using the circular buffer (CB) plugin.

    The CB plugin must read directly from the driver, so `stage()` raises if its
    `NDArrayPort` is not the driver's port.

    Deliberately has no `wait_for_plugins` equivalent to [](#ADAcquireLogic).
    That setting gates the driver's `AcquireBusy` going to 0 at the end of an
    acquisition, and here the driver is always acquiring -- there is no such
    transition to wait on. `wait_for_idle` waits on the circular buffer's
    capture instead.

    :param driver: The continuously acquiring detector driver
    :param cb_plugin: The circular buffer plugin gating downstream plugins
    :param num_frames_to_skip:
        Non-negative number of incoming frames to skip before each gate opening.
        Defaults to 0, which opens the gate immediately.
    :param acquire_period_signal:
        Time between frames, used to scale frame wait timeouts. Defaults to the
        driver's `AcquirePeriod_RBV`; override it for drivers that cannot set
        their acquire period, so it does not reflect the real frame period.
    """

    def __init__(
        self,
        driver: ADBaseIO,
        cb_plugin: NDCircularBuffIO,
        num_frames_to_skip: int = 0,
        acquire_period_signal: SignalR[float] | None = None,
    ):
        self.driver = driver
        self.cb_plugin = cb_plugin
        self.num_frames_to_skip = num_frames_to_skip
        if acquire_period_signal is not None:
            self.acquire_period_signal = acquire_period_signal
        else:
            self.acquire_period_signal = driver.acquire_period

    async def ensure_ready(self):
        cb_input_port, driver_port = await asyncio.gather(
            self.cb_plugin.nd_array_port.get_value(),
            self.driver.port_name.get_value(),
        )
        # Frames queued in plugins between the driver and CB would pass the gate
        # long after they were exposed.
        if cb_input_port != driver_port:
            raise RuntimeError(
                f"CB plugin NDArrayPort is {cb_input_port!r}, but must be the "
                f"driver port {driver_port!r}"
            )
        await self.ensure_stopped()

        await asyncio.gather(
            self.cb_plugin.enable_callbacks.set(EnableDisable.ENABLE),
            self.cb_plugin.pre_count.set(0),
            # Set preset trigger count to 0 to allow triggering to continue indefinitely
            # until explicitly stopped.
            self.cb_plugin.preset_trigger_count.set(0),
            self.cb_plugin.flush_on_soft_trg.set(NDCBFlushOnSoftTrgMode.ON_NEW_IMAGE),
        )
        await set_and_wait_for_value(
            self.cb_plugin.capture,
            True,
            wait_for_set_completion=False,
            timeout=DEFAULT_TIMEOUT,
        )

    async def start_acquiring(self):
        num_frames_to_skip = self.num_frames_to_skip
        if num_frames_to_skip:
            initial_count, acquire_period = await asyncio.gather(
                self.cb_plugin.array_counter.get_value(cached=False),
                self.acquire_period_signal.get_value(),
            )
            # Count arrivals at the gate, not driver frames or monitor callbacks.
            await wait_for_value(
                self.cb_plugin.array_counter,
                lambda count: count >= initial_count + num_frames_to_skip,
                timeout=num_frames_to_skip * acquire_period + DEFAULT_TIMEOUT,
            )
        await self.cb_plugin.trigger_.set(True)

    async def wait_for_idle(self):
        num_images, acquire_period = await asyncio.gather(
            self.cb_plugin.post_count.get_value(),
            self.acquire_period_signal.get_value(),
        )

        await wait_for_value(
            self.cb_plugin.trigger_,
            False,
            timeout=num_images * acquire_period + DEFAULT_TIMEOUT,
        )

    async def ensure_stopped(self):
        await stop_busy_record(self.cb_plugin.capture)
