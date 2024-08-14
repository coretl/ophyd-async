import asyncio

import pytest
from pydantic import ValidationError

from ophyd_async.core import DEFAULT_TIMEOUT, DeviceCollector, set_mock_value
from ophyd_async.epics.pvi import fill_pvi_entries
from ophyd_async.fastcs.panda import (
    CommonPandaBlocks,
    PcompInfo,
    SeqTable,
    SeqTableInfo,
    StaticPcompTriggerLogic,
    StaticSeqTableTriggerLogic,
    seq_table_row,
)


@pytest.fixture
async def mock_panda():
    class Panda(CommonPandaBlocks):
        def __init__(self, prefix: str, name: str = ""):
            self._prefix = prefix
            super().__init__(name)

        async def connect(self, mock: bool = False, timeout: float = DEFAULT_TIMEOUT):
            await fill_pvi_entries(
                self, self._prefix + "PVI", timeout=timeout, mock=mock
            )
            await super().connect(mock=mock, timeout=timeout)

    async with DeviceCollector(mock=True):
        mock_panda = Panda("PANDAQSRV:", "mock_panda")

    assert mock_panda.name == "mock_panda"
    return mock_panda


async def test_seq_table_trigger_logic(mock_panda):
    trigger_logic = StaticSeqTableTriggerLogic(mock_panda.seq[1])
    seq_table = SeqTable(
        [
        seq_table_row(outa1=True, outa2=True),
        seq_table_row(outa1=False, outa2=False),
        seq_table_row(outa1=True, outa2=False),
        seq_table_row(outa1=False, outa2=True),
        ]
    )
    seq_table_info = SeqTableInfo(sequence_table=seq_table, repeats=1)

    async def set_active(value: bool):
        await asyncio.sleep(0.1)
        set_mock_value(mock_panda.seq[1].active, value)

    await trigger_logic.prepare(seq_table_info)
    await asyncio.gather(trigger_logic.kickoff(), set_active(True))
    await asyncio.gather(trigger_logic.complete(), set_active(False))


async def test_pcomp_trigger_logic(mock_panda):
    trigger_logic = StaticPcompTriggerLogic(mock_panda.pcomp[1])
    pcomp_info = PcompInfo(
        start_postion=0,
        pulse_width=1,
        rising_edge_step=1,
        number_of_pulses=5,
        direction="Positive",
    )

    async def set_active(value: bool):
        await asyncio.sleep(0.1)
        set_mock_value(mock_panda.pcomp[1].active, value)

    await trigger_logic.prepare(pcomp_info)
    await asyncio.gather(trigger_logic.kickoff(), set_active(True))
    await asyncio.gather(trigger_logic.complete(), set_active(False))


@pytest.mark.parametrize(
    ["kwargs", "error_msg"],
    [
        (
            {
                "sequence_table": SeqTable([seq_table_row(outc2=1)]),
                "repeats": 0,
                "prescale_as_us": -1,
            },
            "Input should be greater than or equal to 0 "
            "[type=greater_than_equal, input_value=-1, input_type=int]",
        ),
        (
            {
                "sequence_table": SeqTable(
                    [
                    seq_table_row(outc2=True),
                    seq_table_row(outc2=False),
                    seq_table_row(outc2=True),
                    seq_table_row(outc2=False),
                    ]
                ),
                "repeats": -1,
            },
            "Input should be greater than or equal to 0 "
            "[type=greater_than_equal, input_value=-1, input_type=int]",
        ),
        (
            {
                "sequence_table": 1,
                "repeats": 1,
            },
            "Assertion failed, Rows must be a list or numpy array. "
            "[type=assertion_error, input_value=1, input_type=int]"
        ),
    ],
)
def test_malformed_seq_table_info(kwargs, error_msg):
    with pytest.raises(ValidationError) as exc:
        SeqTableInfo(**kwargs)
    assert error_msg in str(exc.value)
