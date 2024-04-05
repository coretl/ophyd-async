import asyncio
from dataclasses import dataclass

from ophyd_async.core import TriggerLogic, wait_for_value
from ophyd_async.panda import SeqBlock, SeqTable


@dataclass
class SequenceTableInfo:
    sequence_table: SeqTable
    repeats: int
    prescale_as_us: float = 1  # microseconds


class StaticSeqTableTriggerLogic(TriggerLogic[SeqTable]):

    def __init__(self, seq: SeqBlock) -> None:
        self.seq = seq

    async def prepare(self, value: SequenceTableInfo):
        await asyncio.gather(
            self.seq.prescale_units.set("us"),
            self.seq.enable.set("ZERO"),
        )
        await asyncio.gather(
            self.seq.prescale.set(value.prescale_as_us),
            self.seq.repeats.set(value.repeats),
            self.seq.table.set(value.sequence_table),
        )

    async def kickoff(self) -> None:
        await self.seq.enable.set("ONE")
        await wait_for_value(self.seq.active, 1, timeout=1)

    async def complete(self) -> None:
        await wait_for_value(self.seq.active, 0, timeout=None)

    async def stop(self):
        await self.seq.enable.set("ZERO")
        await wait_for_value(self.seq.active, 0, timeout=1)
