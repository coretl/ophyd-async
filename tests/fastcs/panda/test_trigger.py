import asyncio

import numpy as np
import pytest
from pydantic import ValidationError
from scanspec.specs import Line, fly

from ophyd_async.core import DeviceCollector, set_mock_value
from ophyd_async.epics import motor
from ophyd_async.fastcs.core import fastcs_connector
from ophyd_async.fastcs.panda import (
    CommonPandaBlocks,
    PcompDirection,
    PcompInfo,
    ScanSpecInfo,
    ScanSpecSeqTableTriggerLogic,
    SeqTable,
    SeqTableInfo,
    SeqTrigger,
    StaticPcompTriggerLogic,
    StaticSeqTableTriggerLogic,
)


@pytest.fixture
async def mock_panda():
    class Panda(CommonPandaBlocks):
        def __init__(self, uri: str, name: str = ""):
            super().__init__(name=name, connector=fastcs_connector(self, uri))

    async with DeviceCollector(mock=True):
        mock_panda = Panda("PANDAQSRV:", "mock_panda")

    assert mock_panda.name == "mock_panda"
    return mock_panda


async def test_seq_table_trigger_logic(mock_panda):
    trigger_logic = StaticSeqTableTriggerLogic(mock_panda.seq[1])
    seq_table = (
        SeqTable.row(outa1=True, outa2=True)
        + SeqTable.row(outa1=False, outa2=False)
        + SeqTable.row(outa1=True, outa2=False)
        + SeqTable.row(outa1=False, outa2=True)
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
        direction=PcompDirection.positive,
    )

    async def set_active(value: bool):
        await asyncio.sleep(0.1)
        set_mock_value(mock_panda.pcomp[1].active, value)

    await trigger_logic.prepare(pcomp_info)
    await asyncio.gather(trigger_logic.kickoff(), set_active(True))
    await asyncio.gather(trigger_logic.complete(), set_active(False))


@pytest.fixture
async def sim_x_motor():
    async with DeviceCollector(mock=True):
        sim_motor = motor.Motor("BLxxI-MO-STAGE-01:X", name="sim_x_motor")

    set_mock_value(sim_motor.encoder_res, 0.02)

    yield sim_motor


@pytest.fixture
async def sim_y_motor():
    async with DeviceCollector(mock=True):
        sim_motor = motor.Motor("BLxxI-MO-STAGE-01:Y", name="sim_x_motor")

    set_mock_value(sim_motor.encoder_res, 0.2)

    yield sim_motor


async def test_seq_scanspec_trigger_logic(mock_panda, sim_x_motor, sim_y_motor) -> None:
    spec = fly(Line(sim_y_motor, 1, 2, 3) * ~Line(sim_x_motor, 1, 5, 5), 1)
    info = ScanSpecInfo(spec=spec, deadtime=0.1)
    trigger_logic = ScanSpecSeqTableTriggerLogic(mock_panda.seq[1])
    await trigger_logic.prepare(info)
    out = await trigger_logic.seq.table.get_value()
    assert (out.repeats == [1, 1, 1, 5, 1, 1, 1, 5, 1, 1, 1, 5, 1]).all()
    assert out.trigger == [
        SeqTrigger.BITA_0,
        SeqTrigger.BITA_1,
        SeqTrigger.POSA_GT,
        SeqTrigger.IMMEDIATE,
        SeqTrigger.BITA_0,
        SeqTrigger.BITA_1,
        SeqTrigger.POSA_LT,
        SeqTrigger.IMMEDIATE,
        SeqTrigger.BITA_0,
        SeqTrigger.BITA_1,
        SeqTrigger.POSA_GT,
        SeqTrigger.IMMEDIATE,
        SeqTrigger.BITA_0,
    ]
    assert (out.position == [0, 0, 25, 0, 0, 0, 275, 0, 0, 0, 25, 0, 0]).all()
    assert (out.time1 == [0, 0, 0, 900000, 0, 0, 0, 900000, 0, 0, 0, 900000, 0]).all()
    assert (out.time2 == [0, 0, 0, 100000, 0, 0, 0, 100000, 0, 0, 0, 100000, 0]).all()


@pytest.mark.parametrize(
    ["kwargs", "error_msg"],
    [
        (
            {
                "sequence_table_factory": lambda: SeqTable.row(outc2=1),
                "repeats": 0,
                "prescale_as_us": -1,
            },
            "Input should be greater than or equal to 0 "
            "[type=greater_than_equal, input_value=-1, input_type=int]",
        ),
        (
            {
                "sequence_table_factory": lambda: (
                    SeqTable.row(outc2=True)
                    + SeqTable.row(outc2=False)
                    + SeqTable.row(outc2=True)
                    + SeqTable.row(outc2=False)
                ),
                "repeats": -1,
            },
            "Input should be greater than or equal to 0 "
            "[type=greater_than_equal, input_value=-1, input_type=int]",
        ),
        (
            {
                "sequence_table_factory": lambda: 1,
                "repeats": 1,
            },
            "Assertion failed, Cannot construct Table from 1 "
            "[type=assertion_error, input_value=1, input_type=int]",
        ),
    ],
)
def test_malformed_seq_table_info(kwargs, error_msg):
    with pytest.raises(ValidationError) as exc:
        SeqTableInfo(sequence_table=kwargs.pop("sequence_table_factory")(), **kwargs)
    assert error_msg in str(exc.value)


def test_malformed_trigger_in_seq_table():
    def full_seq_table(trigger):
        return SeqTable(
            repeats=np.array([1], dtype=np.uint16),
            trigger=trigger,
            position=np.array([1], dtype=np.int32),
            time1=np.array([1], dtype=np.uint32),
            outa1=np.array([1], dtype=np.bool_),
            outb1=np.array([1], dtype=np.bool_),
            outc1=np.array([1], dtype=np.bool_),
            outd1=np.array([1], dtype=np.bool_),
            oute1=np.array([1], dtype=np.bool_),
            outf1=np.array([1], dtype=np.bool_),
            time2=np.array([1], dtype=np.uint32),
            outa2=np.array([1], dtype=np.bool_),
            outb2=np.array([1], dtype=np.bool_),
            outc2=np.array([1], dtype=np.bool_),
            outd2=np.array([1], dtype=np.bool_),
            oute2=np.array([1], dtype=np.bool_),
            outf2=np.array([1], dtype=np.bool_),
        )

    for attempted_table in [
        np.array(["A"], dtype="U32"),
        np.array(["Immediate"], dtype="U32"),
        {"Immediate"},
    ]:
        with pytest.raises(ValidationError) as exc:
            full_seq_table(attempted_table)
        assert "Input should be an instance of Sequence" in str(exc)

    with pytest.raises(ValidationError) as exc:
        full_seq_table(["A"])
    assert (
        "Input should be 'Immediate', 'BITA=0', 'BITA=1', 'BITB=0', 'BITB=1', "
        "'BITC...' [type=enum, input_value='A', input_type=str]"
    ) in str(exc)

    # Pydantic is able to infer type from these
    table = full_seq_table([SeqTrigger.IMMEDIATE])
    assert table.trigger == ["Immediate"] == [SeqTrigger.IMMEDIATE]
    table = full_seq_table(["Immediate"])
    assert table.trigger == ["Immediate"] == [SeqTrigger.IMMEDIATE]
