from xml.etree import ElementTree as ET

import pytest

from ophyd_async.core import init_devices
from ophyd_async.epics import adcore
from ophyd_async.epics.core import epics_signal_r


async def test_nd_attributes_plan_stub(RE):
    async with init_devices(mock=True):
        stat = adcore.NDStatsIO("PREFIX:STATS:")
    param = adcore.NDAttributeParam(
        name="det-sum",
        param="sum",
        datatype=adcore.NDAttributeDataType.DOUBLE,
        description="Sum of det frame",
    )
    pv = adcore.NDAttributePv(
        name="Temperature",
        signal=epics_signal_r(str, "LINKAM:TEMP"),
        description="The sample temperature",
        dbrtype=adcore.NDAttributePvDbrType.DBR_FLOAT,
    )
    RE(adcore.setup_ndattributes(stat, [pv, param]))
    text = await stat.nd_attributes_file.get_value()
    xml = ET.fromstring(text)
    assert xml[0].tag == "Attribute"
    assert xml[0].attrib == {
        "name": "Temperature",
        "type": "EPICS_PV",
        "source": "LINKAM:TEMP",
        "dbrtype": "DBR_FLOAT",
        "description": "The sample temperature",
    }

    assert xml[1].tag == "Attribute"
    assert xml[1].attrib == {
        "name": "det-sum",
        "type": "PARAM",
        "source": "sum",
        "addr": "0",
        "datatype": "DOUBLE",
        "description": "Sum of det frame",
    }


@pytest.mark.parametrize("arg", [0.1, "string", 1, True, False])
async def test_nd_attributes_plan_stub_gives_correct_error(RE, arg):
    async with init_devices(mock=True):
        stat = adcore.NDStatsIO("PREFIX:STATS:")
    with pytest.raises(
        ValueError,
        match=f"Invalid type for ndattributes: {type(arg)}. "
        + "Expected NDAttributePv or NDAttributeParam.",
    ):
        RE(adcore.setup_ndattributes(stat, [arg]))


def test_empty_description_is_left_out_of_the_xml():
    param = adcore.NDAttributeParam(
        name="det-sum", param="sum", datatype=adcore.NDAttributeDataType.DOUBLE
    )
    xml = ET.fromstring(adcore.ndattributes_to_xml([param]))
    assert "description" not in xml[0].attrib
