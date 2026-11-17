from collections.abc import AsyncGenerator

from ophyd_async.core import (
    DetectorAcquireLogic,
    set_and_wait_for_other_value,
)

from ._block import PcapBlock


class PandaAcquireLogic(DetectorAcquireLogic):
    """For controlling arm/disarm of PandA data blocks."""

    def __init__(self, pcap: PcapBlock) -> None:
        self.pcap = pcap

    async def acquire(self) -> AsyncGenerator[None, None]:
        await set_and_wait_for_other_value(self.pcap.arm, True, self.pcap.active, True)
        # TODO: wait for idle, https://github.com/PandABlocks/PandABlocks-FPGA/issues/262
        yield

    async def ensure_stopped(self):
        await set_and_wait_for_other_value(
            self.pcap.arm, False, self.pcap.active, False
        )
