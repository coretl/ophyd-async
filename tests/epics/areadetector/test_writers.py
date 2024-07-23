from unittest.mock import patch

import pytest

from ophyd_async.core import (
    DeviceCollector,
    ShapeProvider,
    StaticPathProvider,
    set_mock_value,
)
from ophyd_async.epics.areadetector.writers import ADBaseDataType, HDFWriter, NDFileHDF


class DummyShapeProvider(ShapeProvider):
    def __init__(self) -> None:
        pass

    async def __call__(self) -> tuple:
        return (10, 10, ADBaseDataType.UInt16)


@pytest.fixture
async def hdf_writer(RE, static_path_provider: StaticPathProvider) -> HDFWriter:
    async with DeviceCollector(mock=True):
        hdf = NDFileHDF("HDF:")

    return HDFWriter(
        hdf,
        static_path_provider,
        name_provider=lambda: "test",
        shape_provider=DummyShapeProvider(),
    )


async def test_correct_descriptor_doc_after_open(hdf_writer: HDFWriter):
    set_mock_value(hdf_writer.hdf.file_path_exists, True)
    with patch("ophyd_async.core.signal.wait_for_value", return_value=None):
        descriptor = await hdf_writer.open()

    assert descriptor == {
        "test": {
            "source": "mock+ca://HDF:FullFileName_RBV",
            "shape": (10, 10),
            "dtype": "array",
            "dtype_numpy": "<u2",
            "external": "STREAM:",
        }
    }

    await hdf_writer.close()


async def test_collect_stream_docs(hdf_writer: HDFWriter):
    assert hdf_writer._file is None

    [item async for item in hdf_writer.collect_stream_docs(1)]
    assert hdf_writer._file
