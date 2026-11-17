from collections.abc import AsyncGenerator

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    DetectorAcquireLogic,
    TriggerableCommand,
    wait_for_value,
)

from ._io import EigerDetectorIO


class EigerAcquireLogic(DetectorAcquireLogic):
    def __init__(
        self, detector: EigerDetectorIO, arm_when_ready: TriggerableCommand
    ) -> None:
        self.detector = detector
        self.arm_when_ready = arm_when_ready

    async def acquire(self) -> AsyncGenerator[None, None]:
        await self.arm_when_ready.trigger()
        yield
        await wait_for_value(self.detector.state, "idle", timeout=DEFAULT_TIMEOUT)

    async def ensure_stopped(self):
        await self.detector.disarm.trigger()
