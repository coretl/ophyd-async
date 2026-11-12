import asyncio
from collections import defaultdict
from typing import Annotated as A
from typing import get_type_hints
from unittest.mock import ANY
from xml.etree import ElementTree as ET

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np
import pytest
from bluesky import RunEngine

from ophyd_async.core import (
    Array1D,
    Device,
    EnableDisable,
    SignalR,
    StandardReadable,
    StaticPathProvider,
    TransitionError,
    TriggerInfo,
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    mock_puts_blocked,
    set_mock_value,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics import adaravis, adcore, adsimdetector
from ophyd_async.epics.core import PvSuffix, epics_signal_r
from ophyd_async.testing import assert_reading


async def test_step_scan_hdf_detector_with_stats_and_temp(
    static_path_provider: StaticPathProvider,
):
    stat = adcore.NDStatsIO("PREFIX:STATS:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector(
            "PREFIX:",
            adcore.ADDataFactory.hdf(static_path_provider, ndattribute_sources=[stat]),
            plugins={"stats": stat},
        )
        temp = epics_signal_r(float, "SAMP:TEMP:RBV")
    # Plugins are not registered automatically; opt this one in
    det.set_readable_format(stat, Format.CHILD)
    # The HDF writer stamps the stats plugin's NDAttributes; the driver stamps these
    ndattributes = [
        adcore.NDAttributePv(
            name="sample-temp",
            signal=temp,
            dbrtype=adcore.NDAttributePvDbrType.DBR_DOUBLE,
            description="Temperature of the sample",
        ),
    ]
    set_mock_value(
        det.driver.nd_attributes_file, adcore.ndattributes_to_xml(ndattributes)
    )
    set_mock_value(stat.enable_callbacks, EnableDisable.ENABLE)
    set_mock_value(stat.compute_statistics, True)
    set_mock_value(stat.nd_attributes_file_nelm, 2048)
    writer = det.get_plugin("hdf", adcore.NDFileHDF5IO)
    set_mock_value(det.driver.acquire_period, 0.1)
    set_mock_value(det.driver.acquire_time, 0.05)
    set_mock_value(det.driver.array_size_x, 1024)
    set_mock_value(det.driver.array_size_y, 768)

    await det.stage()
    # ensure_ready sets WaitForPlugins so read() sees this frame's scalars
    assert await det.driver.wait_for_plugins.get_value() is True
    config_description = await det.describe_configuration()
    assert {
        "det-driver-acquire_period": {
            "dtype": "number",
            "dtype_numpy": "<f8",
            "shape": [],
            "source": "mock+ca://PREFIX:cam1:AcquirePeriod_RBV",
        },
        "det-driver-acquire_time": {
            "dtype": "number",
            "dtype_numpy": "<f8",
            "shape": [],
            "source": "mock+ca://PREFIX:cam1:AcquireTime_RBV",
        },
    }.items() <= config_description.items()
    config = await det.read_configuration()
    assert {"det-driver-acquire_period", "det-driver-acquire_time"} <= config.keys()
    assert config["det-driver-acquire_period"]["value"] == 0.1
    assert config["det-driver-acquire_time"]["value"] == 0.05
    with pytest.raises(TransitionError, match="describe not allowed from STAGED"):
        await det.describe()
    # Need to tell it the path is valid
    set_mock_value(writer.file_path_exists, True)
    # When arm is pressed, then make a single frame
    callback_on_mock_put(
        det.driver.acquire, lambda v: set_mock_value(writer.num_captured, 1)
    )
    # Trigger a single frame then describe, get hints and read
    await det.trigger()
    path_info = static_path_provider()
    uri = f"file://localhost/{path_info.directory_path.as_posix().lstrip('/')}/{path_info.filename}.h5"
    description = await det.describe()
    assert {
        "det": {
            "dtype": "array",
            "dtype_numpy": "|i1",
            "external": "STREAM:",
            "shape": [1, 768, 1024],
            "source": uri,
        },
        # The HDF file serves the stats, under the names of their scalars
        "det-stats-total": {
            "dtype": "number",
            "dtype_numpy": "<f8",
            "external": "STREAM:",
            "shape": [1],
            "source": uri,
        },
        "sample-temp": {
            "dtype": "number",
            "dtype_numpy": "<f8",
            "external": "STREAM:",
            "shape": [1],
            "source": "ca://SAMP:TEMP:RBV",
        },
    }.items() <= description.items()
    assert "det" in det.hints["fields"]
    # The stats scalars are in the file instead of the reading
    reading = await det.read()
    assert not [key for key in reading if key.startswith("det-stats-")]
    ndattribute_keys = [
        "sample-temp",
        "det-stats-min_value",
        "det-stats-min_x",
        "det-stats-min_y",
        "det-stats-max_value",
        "det-stats-max_x",
        "det-stats-max_y",
        "det-stats-mean_value",
        "det-stats-total",
    ]
    num_resources = 1 + len(ndattribute_keys)
    docs = [doc async for doc in det.collect_asset_docs()]
    sr = docs[:num_resources]
    sd = docs[num_resources:]
    assert sr == [
        (
            "stream_resource",
            {
                "data_key": "det",
                "mimetype": "application/x-hdf5",
                "parameters": {
                    "chunk_shape": (1, 768, 1024),
                    "dataset": "/entry/data/data",
                },
                "uid": ANY,
                "uri": uri,
            },
        ),
        *(
            (
                "stream_resource",
                {
                    "data_key": key,
                    "mimetype": "application/x-hdf5",
                    "parameters": {
                        "chunk_shape": (16384,),
                        "dataset": f"/entry/instrument/NDAttributes/{key}",
                    },
                    "uid": ANY,
                    "uri": uri,
                },
            )
            for key in ndattribute_keys
        ),
    ]
    assert sd == [
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 0, "stop": 1},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": sr[i][1]["uid"],
                "uid": ANY,
            },
        )
        for i in range(num_resources)
    ]
    # Check we can prepare and do a second one
    await det.prepare(TriggerInfo(collection_timeout=0.1))
    callback_on_mock_put(
        det.driver.acquire, lambda v: set_mock_value(writer.num_captured, 2)
    )
    await det.trigger()
    docs = [doc async for doc in det.collect_asset_docs()]
    # Each prepare starts new resources in the same file, the indices carrying on
    new_sr, sd = docs[:num_resources], docs[num_resources:]
    assert new_sr == [(name, {**doc, "uid": ANY}) for name, doc in sr]
    assert {doc["uid"] for _, doc in new_sr}.isdisjoint(doc["uid"] for _, doc in sr)
    assert sd == [
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 1, "stop": 2},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": new_sr[i][1]["uid"],
                "uid": ANY,
            },
        )
        for i in range(num_resources)
    ]
    # And if we trigger again without updating num_captured then we timeout
    with pytest.raises(
        TimeoutError, match="Timeout Error while waiting 0.1s to update"
    ):
        await det.trigger()


async def test_step_scan_tiff_detector(
    static_path_provider: StaticPathProvider,
):
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector(
            "PREFIX:", adcore.ADDataFactory.tiff(static_path_provider)
        )

    writer = det.get_plugin("tiff", adcore.NDPluginFileIO)
    set_mock_value(det.driver.array_size_x, 1024)
    set_mock_value(det.driver.array_size_y, 768)

    await det.stage()
    # Need to tell it the path is valid
    set_mock_value(writer.file_path_exists, True)
    # When arm is pressed, then make a single frame
    callback_on_mock_put(
        det.driver.acquire, lambda v: set_mock_value(writer.num_captured, 1)
    )
    # Trigger a single frame then describe and read
    await det.trigger()
    path_info = static_path_provider()
    description = await det.describe()
    uri_dir = f"file://localhost/{path_info.directory_path.as_posix().lstrip('/')}/"
    assert description == {
        "det": {
            "dtype": "array",
            "dtype_numpy": "|i1",
            "external": "STREAM:",
            "shape": [1, 768, 1024],
            "source": uri_dir,
        },
    }
    assert det.hints == {"fields": ["det"]}
    readings = await det.read()
    assert readings == {}
    docs = [doc async for doc in det.collect_asset_docs()]
    sr = docs[:1]
    sd = docs[1:]
    assert sr == [
        (
            "stream_resource",
            {
                "data_key": "det",
                "mimetype": "multipart/related;type=image/tiff",
                "parameters": {
                    "chunk_shape": (1, 768, 1024),
                    "template": "ophyd_async_tests_{:06d}.tiff",
                },
                "uid": ANY,
                "uri": uri_dir,
            },
        ),
    ]
    assert sd == [
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 0, "stop": 1},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": sr[0][1]["uid"],
                "uid": ANY,
            },
        ),
    ]


async def test_flyscan_aravis_detector(static_path_provider: StaticPathProvider):
    async with init_devices(mock=True):
        det = adaravis.AravisDetector(
            "PREFIX:", adcore.ADDataFactory.hdf(static_path_provider)
        )

    writer = det.get_plugin("hdf", adcore.NDPluginFileIO)
    set_mock_value(det.driver.model, "A funny model")
    set_mock_value(det.driver.acquire_period, 0.1)
    set_mock_value(det.driver.acquire_time, 0.05)
    set_mock_value(det.driver.array_size_x, 1024)
    set_mock_value(det.driver.array_size_y, 768)
    await det.stage()
    # Check the model is in the configuration too
    config_description = await det.describe_configuration()
    assert {
        "det-driver-acquire_period": {
            "dtype": "number",
            "dtype_numpy": "<f8",
            "shape": [],
            "source": "mock+ca://PREFIX:cam1:AcquirePeriod_RBV",
        },
        "det-driver-acquire_time": {
            "dtype": "number",
            "dtype_numpy": "<f8",
            "shape": [],
            "source": "mock+ca://PREFIX:cam1:AcquireTime_RBV",
        },
        "det-driver-model": {
            "dtype": "string",
            "dtype_numpy": "|S40",
            "shape": [],
            "source": "mock+ca://PREFIX:cam1:Model_RBV",
        },
    }.items() <= config_description.items()
    config = await det.read_configuration()
    assert config["det-driver-acquire_period"]["value"] == 0.1
    assert config["det-driver-acquire_time"]["value"] == 0.05
    assert config["det-driver-model"]["value"] == "A funny model"
    # Prepare for N frames
    set_mock_value(writer.file_path_exists, True)
    await det.prepare(TriggerInfo(collections_per_event=3, number_of_events=5))
    # Check it is making 3 collections in each event
    path_info = static_path_provider()
    description = await det.describe()
    uri = f"file://localhost/{path_info.directory_path.as_posix().lstrip('/')}/{path_info.filename}.h5"
    assert description == {
        "det": {
            "dtype": "array",
            "dtype_numpy": "|i1",
            "external": "STREAM:",
            "shape": [3, 768, 1024],
            "source": uri,
        },
    }
    assert det.hints == {"fields": ["det"]}
    # Kick it off and start it completing
    await det.kickoff()
    status = det.complete()
    # Check it makes nothing
    docs = [doc async for doc in det.collect_asset_docs()]
    assert docs == []
    # Make 2 and 1/3 events and check they are produced
    set_mock_value(writer.num_captured, 7)
    docs = [doc async for doc in det.collect_asset_docs()]
    sr = docs[:1]
    assert docs == [
        (
            "stream_resource",
            {
                "data_key": "det",
                "mimetype": "application/x-hdf5",
                "parameters": {
                    "chunk_shape": (1, 768, 1024),
                    "dataset": "/entry/data/data",
                },
                "uid": ANY,
                "uri": uri,
            },
        ),
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 0, "stop": 2},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": ANY,
                "uid": ANY,
            },
        ),
    ]
    assert not status.done
    # Update and check it completes
    set_mock_value(writer.num_captured, 15)
    docs = [doc async for doc in det.collect_asset_docs()]
    assert docs == [
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 2, "stop": 5},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": ANY,
                "uid": ANY,
            },
        ),
    ]
    await status
    # Check that another kickoff without prepare is not allowed
    with pytest.raises(
        RuntimeError,
        match="cannot go to FLYING from FLOWN",
    ):
        await det.kickoff()
    # But preparing again is ok
    await det.prepare(TriggerInfo(collections_per_event=3, number_of_events=3))
    await det.kickoff()
    # Completing starts a new resource in the same file, the indices carrying on
    status = det.complete()
    set_mock_value(writer.num_captured, 24)
    docs = [doc async for doc in det.collect_asset_docs()]
    new_sr = docs[:1]
    assert new_sr == [(name, {**doc, "uid": ANY}) for name, doc in sr]
    assert new_sr[0][1]["uid"] != sr[0][1]["uid"]
    assert docs[1:] == [
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 5, "stop": 8},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": new_sr[0][1]["uid"],
                "uid": ANY,
            },
        ),
    ]
    await status


async def test_2_rois_with_hdf(tmp_path):
    path_provider = StaticPathProvider(
        lambda datakey_name=None: str(datakey_name), tmp_path
    )
    roi1 = adcore.NDROIIO("PREFIX:ROI1")
    roi2 = adcore.NDROIIO("PREFIX:ROI2")
    async with init_devices(mock=True):
        det = adaravis.AravisDetector(
            "PREFIX:",
            adcore.ADDataFactory.hdf(
                path_provider,
                writer_suffix="HDF1:",
                writer_name="hdf1",
                datakey_suffix="-roi1",
                array_description=lambda driver: adcore.NDArrayDescription(
                    shape_signals=(roi1.size_y, roi1.size_x),
                    data_type_signal=driver.data_type,
                    color_mode_signal=driver.color_mode,
                ),
            ),
            adcore.ADDataFactory.hdf(
                path_provider,
                writer_suffix="HDF2:",
                writer_name="hdf2",
                datakey_suffix="-roi2",
                array_description=lambda driver: adcore.NDArrayDescription(
                    shape_signals=(roi2.size_y, roi2.size_x),
                    data_type_signal=driver.data_type,
                    color_mode_signal=driver.color_mode,
                ),
            ),
            plugins={"roi1": roi1, "roi2": roi2},
        )
    hdf1 = det.get_plugin("hdf1", adcore.NDFileHDF5IO)
    hdf2 = det.get_plugin("hdf2", adcore.NDFileHDF5IO)
    await det.stage()

    # When arm is pressed, then make a single frame on each HDF
    def publish_captured(v):
        set_mock_value(hdf1.num_captured, 1)
        set_mock_value(hdf2.num_captured, 1)

    callback_on_mock_put(det.driver.acquire, publish_captured)
    # Setup the size of the rois and say the directory exists
    set_mock_value(roi1.size_x, 400)
    set_mock_value(roi1.size_y, 300)
    set_mock_value(roi2.size_x, 200)
    set_mock_value(roi2.size_y, 100)
    set_mock_value(hdf1.file_path_exists, True)
    set_mock_value(hdf2.file_path_exists, True)
    # Trigger a single frame then describe and read
    await det.trigger()
    description = await det.describe()
    path_info = path_provider()
    uri1 = f"file://localhost/{path_info.directory_path.as_posix().lstrip('/')}/det-roi1.h5"
    uri2 = f"file://localhost/{path_info.directory_path.as_posix().lstrip('/')}/det-roi2.h5"
    assert description == {
        "det-roi1": {
            "dtype": "array",
            "dtype_numpy": "|i1",
            "external": "STREAM:",
            "shape": [1, 300, 400],
            "source": uri1,
        },
        "det-roi2": {
            "dtype": "array",
            "dtype_numpy": "|i1",
            "external": "STREAM:",
            "shape": [1, 100, 200],
            "source": uri2,
        },
    }
    assert det.hints == {"fields": ["det-roi1", "det-roi2"]}
    docs = [doc async for doc in det.collect_asset_docs()]
    sr = docs[0:4:2]
    sd = docs[1:5:2]
    assert sr == [
        (
            "stream_resource",
            {
                "data_key": "det-roi1",
                "mimetype": "application/x-hdf5",
                "parameters": {
                    "chunk_shape": (1, 300, 400),
                    "dataset": "/entry/data/data",
                },
                "uid": ANY,
                "uri": uri1,
            },
        ),
        (
            "stream_resource",
            {
                "data_key": "det-roi2",
                "mimetype": "application/x-hdf5",
                "parameters": {
                    "chunk_shape": (1, 100, 200),
                    "dataset": "/entry/data/data",
                },
                "uid": ANY,
                "uri": uri2,
            },
        ),
    ]
    assert sd == [
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 0, "stop": 1},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": sr[0][1]["uid"],
                "uid": ANY,
            },
        ),
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 0, "stop": 1},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": sr[1][1]["uid"],
                "uid": ANY,
            },
        ),
    ]


async def test_simdetector_with_stats_signal():
    stat = adcore.NDStatsIO("PREFIX:STAT:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector("PREFIX:", plugins={"stats": stat})
    # Plugins are not registered automatically; opt this one in
    det.set_readable_format(stat, Format.CHILD)
    _switch_stats(stat, on=True)
    set_mock_value(stat.total, 1.8)
    await det.stage()
    assert await det.driver.wait_for_plugins.get_value() is True
    await det.trigger()
    description = await det.describe()
    assert "det-stats-total" in description
    assert description["det-stats-total"] == {
        "dtype": "number",
        "dtype_numpy": "<f8",
        "shape": [],
        "source": "mock+ca://PREFIX:STAT:Total_RBV",
    }
    reading = await det.read()
    assert reading["det-stats-total"]["value"] == 1.8
    assert "det-stats-total" in det.hints["fields"]
    assert "det-stats-mean_value" in det.hints["fields"]


@pytest.mark.parametrize("wait_for_plugins", [True, False])
async def test_acquire_logic_writes_wait_for_plugins_on_stage(wait_for_plugins: bool):
    # Always written, never left at whatever the IOC booted with, so a scan
    # cannot depend on who touched the PV last
    driver = adcore.ADBaseIO("PREFIX:DET:")
    async with init_devices(mock=True):
        det = adcore.AreaDetector(
            driver,
            acquire_logic=adcore.ADAcquireLogic(
                driver, wait_for_plugins=wait_for_plugins
            ),
            name="det",
        )
    set_mock_value(det.driver.wait_for_plugins, not wait_for_plugins)
    await det.stage()
    assert await det.driver.wait_for_plugins.get_value() is wait_for_plugins


async def test_step_scan_keep_numimages(
    static_path_provider: StaticPathProvider,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "YES")
    stat = adcore.NDStatsIO("PREFIX:STATS:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector(
            "PREFIX:",
            adcore.ADDataFactory.hdf(static_path_provider),
            plugins={"stats": stat},
        )
    # Plugins are not registered automatically, and the stats readables give one
    # value per event so cannot be used with 42 collections per event

    set_mock_value(det.driver.acquire_period, 0.1)
    set_mock_value(det.driver.acquire_time, 0.05)
    set_mock_value(det.driver.array_size_x, 1024)
    set_mock_value(det.driver.array_size_y, 768)
    await det.driver.num_images.set(42)
    assert await det.driver.num_images.get_value() == 42
    writer = det.get_plugin("hdf", adcore.NDFileHDF5IO)
    config = await det.read_configuration()
    assert {"det-driver-acquire_period", "det-driver-acquire_time"} <= config.keys()
    assert config["det-driver-acquire_period"]["value"] == 0.1
    assert config["det-driver-acquire_time"]["value"] == 0.05
    await det.stage()
    with pytest.raises(TransitionError, match="describe not allowed from STAGED"):
        await det.describe()
    # Need to tell it the path is valid
    set_mock_value(writer.file_path_exists, True)
    # When arm is pressed, then make a single frame
    callback_on_mock_put(
        det.driver.acquire, lambda v: set_mock_value(writer.num_captured, 42)
    )
    # Trigger a single frame then describe, get hints and read
    await det.trigger()
    assert await det.driver.num_images.get_value() == 42
    path_info = static_path_provider()
    description = await det.describe()
    uri = f"file://localhost/{path_info.directory_path.as_posix().lstrip('/')}/{path_info.filename}.h5"
    assert description["det"] == {
        "dtype": "array",
        "dtype_numpy": "|i1",
        "external": "STREAM:",
        "shape": [42, 768, 1024],  # 42 came from num_images
        "source": uri,
    }
    assert "det" in det.hints["fields"]
    assert await det.read() == {}
    sr, sd = [doc async for doc in det.collect_asset_docs()]
    assert sr == (
        "stream_resource",
        {
            "data_key": "det",
            "mimetype": "application/x-hdf5",
            "parameters": {
                "chunk_shape": (1, 768, 1024),
                "dataset": "/entry/data/data",
            },
            "uid": ANY,
            "uri": uri,
        },
    )

    assert sd == (
        "stream_datum",
        {
            "descriptor": "",
            "indices": {"start": 0, "stop": 1},
            "seq_nums": {"start": 0, "stop": 0},
            "stream_resource": sr[1]["uid"],
            "uid": ANY,
        },
    )

    # Check we can prepare and change the exposure. Changing the exposure changes
    # the frame period, which (per ADR 0025) determines the HDF chunk shape, so a
    # new StreamResource is opened rather than the old file being reused. Here the
    # chunk shape is unchanged because this writer leaves flush_period unset and so
    # reads the chunk size back from the IOC, but the period change still starts a
    # fresh file.
    # Starting a capture resets the IOC's NumCaptured, which start() waits for
    callback_on_mock_put(
        writer.capture, lambda v: set_mock_value(writer.num_captured, 0) if v else None
    )
    await det.prepare(
        TriggerInfo(collections_per_event=42, livetime=0.01, collection_timeout=0.1)
    )
    callback_on_mock_put(
        det.driver.acquire, lambda v: set_mock_value(writer.num_captured, 42)
    )
    assert await det.driver.acquire_time.get_value() == 0.01
    await det.trigger()
    readings = await det.read()
    assert "det-stats-total" not in readings
    new_sr, new_sd = [doc async for doc in det.collect_asset_docs()]
    assert new_sr[0] == "stream_resource"
    assert new_sr[1]["uid"] != sr[1]["uid"]
    assert new_sr[1]["parameters"]["chunk_shape"] == (1, 768, 1024)
    assert new_sd == (
        "stream_datum",
        {
            "descriptor": "",
            "indices": {"start": 0, "stop": 1},
            "seq_nums": {"start": 0, "stop": 0},
            "stream_resource": new_sr[1]["uid"],
            "uid": ANY,
        },
    )

    # And if we trigger again without updating num_captured then we timeout
    with pytest.raises(
        TimeoutError, match="Timeout Error while waiting 0.1s to update"
    ):
        await det.trigger()


async def test_kinetix_step_scan_with_averaging_filter(
    static_path_provider: StaticPathProvider,
):
    async with init_devices(mock=True):
        from ophyd_async.epics import adkinetix

        det = adkinetix.KinetixDetector(
            "PREFIX:",
            adcore.ADDataFactory.hdf(static_path_provider),
            proc_suffix="PROC:",
        )

    proc = det.get_plugin("proc", adcore.NDProcessIO)
    writer = det.get_plugin("hdf", adcore.NDFileHDF5IO)
    set_mock_value(det.driver.acquire_period, 0.1)
    set_mock_value(det.driver.acquire_time, 0.05)
    set_mock_value(det.driver.array_size_x, 512)
    set_mock_value(det.driver.array_size_y, 512)
    set_mock_value(writer.file_path_exists, True)
    set_mock_value(proc.filter_type, adcore.NDProcessFilterType.AVERAGE)

    await det.stage()

    # When arm is pressed, produce frames
    callback_on_mock_put(
        det.driver.acquire, lambda v: set_mock_value(writer.num_captured, 2)
    )

    # Prepare with exposures_per_collection=5 and 2 collections to enable
    # the averaging filter
    await det.prepare(
        TriggerInfo(
            exposures_per_collection=5,
            collections_per_event=2,
            livetime=0.05,
            exposure_timeout=1.0,
        )
    )

    # Verify process plugin was configured for averaging
    assert await proc.num_filter.get_value() == 5
    assert await proc.enable_filter.get_value() is True
    assert await proc.filter_type.get_value() == adcore.NDProcessFilterType.AVERAGE
    assert await proc.auto_reset_filter.get_value() is True
    assert await proc.data_type_out.get_value() == adcore.ADBaseDataType.AUTOMATIC
    assert (
        await proc.filter_callbacks.get_value()
        == adcore.NDProcessFilterCallbacks.ARRAY_N_ONLY
    )

    # Verify num_images is set to collections * exposures_per_collection = 2 * 5 = 10
    assert await det.driver.num_images.get_value() == 10

    # Trigger and collect
    await det.trigger()
    path_info = static_path_provider()
    description = await det.describe()
    uri = f"file://localhost/{path_info.directory_path.as_posix().lstrip('/')}/{path_info.filename}.h5"
    assert description == {
        "det": {
            "dtype": "array",
            "dtype_numpy": "|i1",
            "external": "STREAM:",
            "shape": [2, 512, 512],
            "source": uri,
        },
    }
    assert det.hints == {"fields": ["det"]}
    await assert_reading(det, {})
    docs = [doc async for doc in det.collect_asset_docs()]
    sr = docs[:1]
    sd = docs[1:]
    assert sr == [
        (
            "stream_resource",
            {
                "data_key": "det",
                "mimetype": "application/x-hdf5",
                "parameters": {
                    "chunk_shape": (1, 512, 512),
                    "dataset": "/entry/data/data",
                },
                "uid": ANY,
                "uri": uri,
            },
        ),
    ]
    assert sd == [
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 0, "stop": 1},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": sr[0][1]["uid"],
                "uid": ANY,
            },
        ),
    ]


# Use cases 6 and 7: several data logics on one detector, whichever one serves
# the scan chosen by the plugins' EnableCallbacks and compute switches. One name
# per quantity: the readable, the time series and the HDF NDAttribute of a
# statistic are all named after its scalar signal, e.g. det-stats-total.

# The statistics NDStatsIO gives a read format, so publishes when switched on
STATS_KEYS = [
    "det-stats-min_value",
    "det-stats-min_x",
    "det-stats-min_y",
    "det-stats-max_value",
    "det-stats-max_x",
    "det-stats-max_y",
    "det-stats-mean_value",
    "det-stats-total",
]


# The statistics NDStatsIO hints, whichever tier serves them
STATS_HINTS = [
    "det-stats-min_value",
    "det-stats-max_value",
    "det-stats-mean_value",
    "det-stats-total",
]


def _switch_stats(stats: adcore.NDStatsIO, on: bool) -> None:
    """Set the PVs that switch the basic statistics of a stats plugin."""
    set_mock_value(
        stats.enable_callbacks, EnableDisable.ENABLE if on else EnableDisable.DISABLE
    )
    set_mock_value(stats.compute_statistics, on)


def _time_series_arrays(stats: adcore.NDStatsIO) -> list[Device]:
    """The array of each statistic of a stats plugin, not its timestamps."""
    hints = get_type_hints(adcore.NDStatsIO)
    return [
        signal
        for attr, signal in stats.children()
        if attr.startswith("ts_")
        and attr != "ts_timestamp"
        and hints[attr] == SignalR[Array1D[np.float64]]
    ]


def _fill_time_series(
    det, plugins: list[adcore.NDStatsIO], points: int, also=lambda: None
):
    """Behave like the IOC: arming a time series erases it, acquiring counts the
    points, and the arrays are posted only when TSRead asks for them, as for a
    page collected between the plugin's once-a-second posts."""

    def erase(value):
        if value:
            for plugin in plugins:
                set_mock_value(plugin.ts_current_point, 0)

    def count(value):
        if value:
            for plugin in plugins:
                set_mock_value(plugin.ts_current_point, points)
            also()

    def post(plugin: adcore.NDStatsIO):
        for array in _time_series_arrays(plugin):
            set_mock_value(
                array,  # pyright: ignore[reportArgumentType]
                np.arange(points, dtype=np.float64),
            )
        set_mock_value(
            plugin.ts_timestamp,  # pyright: ignore[reportArgumentType]
            100.0 + np.arange(points),
        )

    for plugin in plugins:
        set_mock_value(plugin.ts_timestamp_nelm, 2048)
        callback_on_mock_put(plugin.ts_acquire, erase)
        callback_on_mock_put(plugin.ts_read, lambda _, plugin=plugin: post(plugin))
    callback_on_mock_put(det.driver.acquire, count)


def _fly_plan(det, info: TriggerInfo):
    @bpp.stage_decorator([det])
    @bpp.run_decorator()
    def plan():
        yield from bps.prepare(det, info, wait=True)
        yield from bps.declare_stream(det, name="primary", collect=True)
        yield from bps.kickoff(det, wait=True)
        yield from bps.collect_while_completing(
            flyers=[det], dets=[det], flush_period=0.05
        )

    return plan()


def _fly(
    RE: RunEngine,
    det,
    series: list[adcore.NDStatsIO],
    points: int,
    hdf: adcore.NDFileHDF5IO | None = None,
) -> dict[str, list]:
    also = lambda: None  # noqa: E731
    if hdf is not None:
        also = lambda: set_mock_value(hdf.num_captured, points)  # noqa: E731
    _fill_time_series(det, series, points=points, also=also)
    docs: dict[str, list] = defaultdict(list)
    RE(
        _fly_plan(det, TriggerInfo(number_of_events=points)),
        lambda n, d: docs[n].append(d),
    )
    return docs


async def _superset_detector(
    path_provider: StaticPathProvider, always_enable: bool = False
):
    """Use case 6: the readables, time series and HDF NDAttributes of one stats
    plugin, with its basic statistics switched on."""
    stats = adcore.NDStatsIO("PREFIX:STAT:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector(
            "PREFIX:",
            adcore.ADDataFactory.hdf(
                path_provider, always_enable=always_enable, ndattribute_sources=[stats]
            ),
            adcore.ADDataFactory.time_series(stats),
            plugins={"stats": stats},
        )
    _switch_stats(stats, on=True)
    set_mock_value(stats.nd_attributes_file_nelm, 2048)
    set_mock_value(det.driver.array_size_x, 1024)
    set_mock_value(det.driver.array_size_y, 768)
    set_mock_value(det.get_plugin("hdf", adcore.NDFileHDF5IO).file_path_exists, True)
    return det


# Use cases A and B: a detector with a stats plugin in plugins= only, then with its
# time series. A plugin in plugins= is a readable, so its hints come with it.


async def test_uc_a_detector_with_readable_stats(RE: RunEngine):
    stats = adcore.NDStatsIO("PREFIX:STAT:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector("PREFIX:", plugins={"stats": stats}, name="det")
    _switch_stats(stats, on=True)
    set_mock_value(stats.total, 12.0)

    @bpp.stage_decorator([det])
    @bpp.run_decorator()
    def step_plan():
        yield from bps.trigger_and_read([det])

    docs: dict[str, list] = defaultdict(list)
    RE(step_plan(), lambda name, doc: docs[name].append(doc))

    (descriptor,) = docs["descriptor"]
    assert list(descriptor["data_keys"]) == STATS_KEYS
    assert descriptor["hints"]["det"]["fields"] == STATS_HINTS
    (event,) = docs["event"]
    assert event["data"]["det-stats-total"] == 12.0


async def test_uc_a_a_plugin_opts_out_of_being_read():
    stats = adcore.NDStatsIO("PREFIX:STAT:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector("PREFIX:", plugins={"stats": stats}, name="det")
    _switch_stats(stats, on=True)
    det.set_readable_format(stats, None)

    await det.stage()
    await det.prepare(TriggerInfo())

    assert not [key for key in await det.describe() if "stats" in key]
    assert det.hints == {}


def test_uc_b_detector_with_a_stats_time_series(RE: RunEngine):
    stats = adcore.NDStatsIO("PREFIX:STAT:")
    with init_devices(mock=True):
        det = adsimdetector.SimDetector(
            "PREFIX:",
            adcore.ADDataFactory.time_series(stats),
            plugins={"stats": stats},
            name="det",
        )
    _switch_stats(stats, on=True)

    docs = _fly(RE, det, [stats], points=3)

    (descriptor,) = docs["descriptor"]
    assert list(descriptor["data_keys"]) == STATS_KEYS
    # The time series replaces the readable, whose hints stay
    assert descriptor["hints"]["det"]["fields"] == STATS_HINTS
    (page,) = docs["event_page"]
    assert page["data"]["det-stats-total"] == [[0.0], [1.0], [2.0]]
    # The arrays were posted for the page
    get_mock_put(stats.ts_read).assert_called_once_with(True)
    # It made no plugin of its own
    assert [name for name, _ in det.children() if "stat" in name] == ["stats"]


@pytest.mark.parametrize(
    "always_enable,hdf_enabled,hdf_serves",
    [
        # By its PV, HDF on: the file serves, every time series suppressed
        (False, EnableDisable.ENABLE, True),
        # By its PV, HDF off: the time series serves
        (False, EnableDisable.DISABLE, False),
        # Always enabled, HDF serves whatever its PV was
        (True, EnableDisable.DISABLE, True),
    ],
)
async def test_uc6_the_hdf_switch_picks_the_file_or_the_time_series(
    RE: RunEngine,
    static_path_provider: StaticPathProvider,
    always_enable: bool,
    hdf_enabled: EnableDisable,
    hdf_serves: bool,
):
    det = await _superset_detector(static_path_provider, always_enable)
    hdf = det.get_plugin("hdf", adcore.NDFileHDF5IO)
    set_mock_value(hdf.enable_callbacks, hdf_enabled)

    docs = _fly(RE, det, [det.get_plugin("stats", adcore.NDStatsIO)], 3, hdf=hdf)

    # The registered stats plugin hints the statistics, whoever serves them
    (descriptor,) = docs["descriptor"]
    assert descriptor["hints"]["det"]["fields"] == [
        *STATS_HINTS,
        *(["det"] if hdf_serves else []),
    ]
    if hdf_serves:
        assert docs["event_page"] == []
        keys = [doc["data_key"] for doc in docs["stream_resource"]]
        assert keys == ["det", *STATS_KEYS]
    else:
        assert docs["stream_resource"] == []
        (page,) = docs["event_page"]
        assert list(page["data"]) == STATS_KEYS
        assert page["data"]["det-stats-total"] == [[0.0], [1.0], [2.0]]


async def test_uc6_hdf_enabled_the_file_serves_every_stats_quantity(
    static_path_provider: StaticPathProvider,
):
    det = await _superset_detector(static_path_provider)
    hdf = det.get_plugin("hdf", adcore.NDFileHDF5IO)
    set_mock_value(hdf.enable_callbacks, EnableDisable.ENABLE)
    await det.stage()
    callback_on_mock_put(
        det.driver.acquire, lambda v: set_mock_value(hdf.num_captured, 1)
    )
    await det.trigger()

    description = await det.describe()

    assert description["det-stats-total"]["external"] == "STREAM:"
    assert "det-stats-total" not in await det.read()


@pytest.fixture
async def two_roi_detector(tmp_path):
    """Use case 7: two HDF writers and two time series, one per ROI."""
    path_provider = StaticPathProvider(
        lambda datakey_name=None: str(datakey_name), tmp_path
    )
    stats1 = adcore.NDStatsIO("PREFIX:STAT1:")
    stats2 = adcore.NDStatsIO("PREFIX:STAT2:")
    async with init_devices(mock=True):
        det = adaravis.AravisDetector(
            "PREFIX:",
            adcore.ADDataFactory.hdf(
                path_provider,
                writer_suffix="HDF1:",
                writer_name="hdf1",
                datakey_suffix="-roi1",
                always_enable=False,
            ),
            adcore.ADDataFactory.hdf(
                path_provider,
                writer_suffix="HDF2:",
                writer_name="hdf2",
                datakey_suffix="-roi2",
                always_enable=False,
            ),
            adcore.ADDataFactory.time_series(stats1),
            adcore.ADDataFactory.time_series(stats2),
            plugins={"stats1": stats1, "stats2": stats2},
        )
    set_mock_value(det.driver.array_size_x, 1024)
    set_mock_value(det.driver.array_size_y, 768)
    for name in ("hdf1", "hdf2"):
        plugin = det.get_plugin(name, adcore.NDFileHDF5IO)
        set_mock_value(plugin.enable_callbacks, EnableDisable.DISABLE)
        set_mock_value(plugin.file_path_exists, True)
    return det


@pytest.mark.parametrize(
    "stats1_on,stats2_on",
    [(True, True), (False, True)],
)
def test_uc7_hdf_disabled_the_time_series_switched_on_serve_in_one_page(
    RE: RunEngine, two_roi_detector, stats1_on: bool, stats2_on: bool
):
    det = two_roi_detector
    series = [det.get_plugin(n, adcore.NDStatsIO) for n in ("stats1", "stats2")]
    _switch_stats(series[0], stats1_on)
    _switch_stats(series[1], stats2_on)

    docs = _fly(RE, det, series, points=2)

    assert docs["stream_resource"] == []
    (page,) = docs["event_page"]
    totals = [
        f"det-{name}-total"
        for name, on in (("stats1", stats1_on), ("stats2", stats2_on))
        if on
    ]
    assert [key for key in page["data"] if key.endswith("-total")] == totals
    for total in totals:
        assert page["data"][total] == [[0.0], [1.0]]


# Ruling 1: a once-per-event readable is suppressed by a same-named higher tier at
# any collections_per_event, and read once per event when nothing suppresses it


@pytest.mark.parametrize("cpe", [1, 2])
@pytest.mark.parametrize("hdf_enabled", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
async def test_readable_stats_total_is_suppressed_by_any_higher_tier(
    static_path_provider: StaticPathProvider,
    bounded: bool,
    hdf_enabled: bool,
    cpe: int,
):
    """readable total < time series total (bounded) < HDF NDAttribute (unbounded)."""
    stats = adcore.NDStatsIO("PREFIX:STAT:")
    factories: list[adcore.ADDataFactory] = [
        adcore.ADDataFactory.hdf(
            static_path_provider, always_enable=False, ndattribute_sources=[stats]
        )
    ]
    if bounded:
        factories.append(adcore.ADDataFactory.time_series(stats))
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector(
            "PREFIX:", *factories, plugins={"stats": stats}, name="det"
        )
    _switch_stats(stats, on=True)
    set_mock_value(stats.ts_timestamp_nelm, 2048)
    set_mock_value(stats.nd_attributes_file_nelm, 2048)
    hdf = det.get_plugin("hdf", adcore.NDFileHDF5IO)
    set_mock_value(
        hdf.enable_callbacks,
        EnableDisable.ENABLE if hdf_enabled else EnableDisable.DISABLE,
    )
    set_mock_value(hdf.file_path_exists, True)
    set_mock_value(det.driver.array_size_x, 1024)
    set_mock_value(det.driver.array_size_y, 768)
    await det.stage()
    info = TriggerInfo(number_of_events=1, collections_per_event=cpe)

    await det.prepare(info)
    total = (await det.describe())["det-stats-total"]
    if bounded or hdf_enabled:
        # Suppressed: the key now describes a [cpe] collections, not one value
        assert total["shape"] == [cpe]
    else:
        assert total["shape"] == []


@pytest.mark.parametrize("cpe", [1, 2])
async def test_readable_with_no_data_logic_is_read_once_per_event(cpe: int):
    stat = adcore.NDStatsIO("PREFIX:STATS:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector("PREFIX:", plugins={"stats": stat}, name="det")
    _switch_stats(stat, on=True)
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=1, collections_per_event=cpe))

    assert (await det.describe())["det-stats-total"]["shape"] == []


@pytest.mark.parametrize(
    "enabled,computed,published",
    [
        (EnableDisable.ENABLE, True, True),
        # The plugin computes nothing while disabled
        (EnableDisable.DISABLE, True, False),
        (EnableDisable.ENABLE, False, False),
    ],
)
async def test_stats_scalar_follows_its_switches_at_first_prepare(
    enabled: EnableDisable, computed: bool, published: bool
):
    """The switches at the first prepare() decide which scalars are read."""
    stat = adcore.NDStatsIO("PREFIX:STATS:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector("PREFIX:", plugins={"stats": stat}, name="det")
    set_mock_value(stat.enable_callbacks, enabled)
    set_mock_value(stat.compute_statistics, computed)

    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()

    assert ("det-stats-total" in await det.describe()) is published
    assert ("det-stats-total" in await det.read()) is published
    assert ("det-stats-total" in det.hints.get("fields", [])) is published
    await det.unstage()


@pytest.mark.parametrize(
    "changed",
    [
        lambda stat: set_mock_value(stat.compute_statistics, False),
        lambda stat: set_mock_value(stat.enable_callbacks, EnableDisable.DISABLE),
    ],
    ids=["compute", "callbacks"],
)
async def test_prepare_raises_for_a_switch_changed_since_the_first_prepare(changed):
    stat = adcore.NDStatsIO("PREFIX:STATS:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector("PREFIX:", plugins={"stats": stat}, name="det")
    _switch_stats(stat, on=True)
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()
    changed(stat)

    with pytest.raises(
        RuntimeError, match=r"changed since the first prepare.*det-stats-total"
    ):
        await det.prepare(TriggerInfo())

    # Staging again takes the new value, and nothing was written to the PV
    await det.unstage()
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()
    assert "det-stats-total" not in await det.read()
    get_mock_put(stat.compute_statistics).assert_not_called()


async def test_retry_after_hidden_set_change_is_a_fresh_first_prepare():
    stat = adcore.NDStatsIO("PREFIX:STATS:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector("PREFIX:", plugins={"stats": stat}, name="det")
    _switch_stats(stat, on=True)
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()
    set_mock_value(stat.compute_statistics, False)

    with pytest.raises(RuntimeError, match=r"changed since the first prepare"):
        await det.prepare(TriggerInfo())
    # The failed prepare() left no recording, so the retry is a first prepare
    # that takes the changed value as its new baseline
    await det.prepare(TriggerInfo())


@pytest.mark.parametrize("exposures_per_collection", [1, 2])
async def test_the_process_plugin_is_switched_on_by_the_acquire_logic(
    exposures_per_collection: int,
):
    """The detector made the process plugin, so prepare() switches it on."""
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector("PREFIX:", proc_suffix="PROC:", name="det")
    proc = det.get_plugin("proc", adcore.NDProcessIO)
    set_mock_value(proc.enable_callbacks, EnableDisable.DISABLE)

    await det.stage()
    await det.prepare(TriggerInfo(exposures_per_collection=exposures_per_collection))
    await det.trigger()

    assert await proc.enable_callbacks.get_value() is EnableDisable.ENABLE


@pytest.mark.parametrize("on_at_first_prepare", [True, False])
async def test_hdf_by_settings_follows_its_plugin_at_first_prepare(
    tmp_path, on_at_first_prepare: bool
):
    """The writer's plugin at the first prepare() decides whether the writer serves."""
    path_provider = StaticPathProvider(
        lambda datakey_name=None: str(datakey_name), tmp_path
    )
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector(
            "PREFIX:",
            *(
                adcore.ADDataFactory.hdf(
                    path_provider,
                    writer_suffix=f"HDF{i}:",
                    writer_name=f"hdf{i}",
                    datakey_suffix=f"-roi{i}",
                    always_enable=False,
                )
                for i in (1, 2)
            ),
        )
    set_mock_value(det.driver.array_size_x, 1024)
    set_mock_value(det.driver.array_size_y, 768)
    hdf1, hdf2 = (det.get_plugin(f"hdf{i}", adcore.NDFileHDF5IO) for i in (1, 2))
    for hdf in (hdf1, hdf2):
        set_mock_value(hdf.file_path_exists, True)
    set_mock_value(hdf1.enable_callbacks, EnableDisable.ENABLE)
    frames = 0

    def acquire(value: bool) -> None:
        nonlocal frames
        if value:
            frames += 3
            for hdf in (hdf1, hdf2):
                set_mock_value(hdf.num_captured, frames)

    callback_on_mock_put(det.driver.acquire, acquire)
    served = {"det-roi1", "det-roi2"} if on_at_first_prepare else {"det-roi1"}
    info = TriggerInfo(number_of_events=3)

    def enable_hdf2(enabled: bool) -> None:
        value = EnableDisable.ENABLE if enabled else EnableDisable.DISABLE
        set_mock_value(hdf2.enable_callbacks, value)

    enable_hdf2(on_at_first_prepare)
    await det.stage()
    await det.prepare(info)
    await det.kickoff()
    await det.complete()
    assert set(await det.describe_collect()) == served
    # Events since prepare()
    assert await det.get_index() == info.number_of_events

    await det.unstage()


# Use case 8: a stats plugin per ROI, each stamped into the file of its own HDF
# writer. A writer stamps only the NDAttribute sources listed for it.


def _publish_only_total(stats: adcore.NDStatsIO) -> None:
    """Leave the total as the only statistic with a read format."""
    for signal, format in stats.get_readable_formats():
        if signal is not stats.total and format is not Format.CONFIG_SIGNAL:
            stats.set_readable_format(signal, None)


def _roi_array_description(roi: adcore.NDROIIO):
    return lambda driver: adcore.NDArrayDescription(
        shape_signals=(roi.size_y, roi.size_x),
        data_type_signal=driver.data_type,
        color_mode_signal=driver.color_mode,
    )


# Neither stats plugin is a plugins= entry: each is registered on its own ROI,
# so the suppress logic must walk the readable tree to hide roi2's switched-off
# total
@pytest.mark.parametrize("roi2_computes_stats", [True, False])
async def test_uc8_2_rois_each_with_stats_and_hdf(
    RE: RunEngine, tmp_path, roi2_computes_stats: bool
):
    path_provider = StaticPathProvider(
        lambda datakey_name=None: str(datakey_name), tmp_path
    )
    rois = [adcore.NDROIIO(f"PREFIX:ROI{i}:") for i in (1, 2)]
    for i, roi in enumerate(rois, start=1):
        roi.stat = adcore.NDStatsIO(f"PREFIX:STAT{i}:")
        _publish_only_total(roi.stat)
    roi1, roi2 = rois
    async with init_devices(mock=True):
        det = adaravis.AravisDetector(
            "PREFIX:",
            *(
                adcore.ADDataFactory.hdf(
                    path_provider,
                    writer_suffix=f"HDF{i}:",
                    writer_name=f"hdf{i}",
                    datakey_suffix=f"-roi{i}",
                    array_description=_roi_array_description(roi),
                    ndattribute_sources=[roi.stat],
                )
                for i, roi in enumerate(rois, start=1)
            ),
            plugins={"roi1": roi1, "roi2": roi2},
        )
    hdfs = [det.get_plugin(f"hdf{i}", adcore.NDFileHDF5IO) for i in (1, 2)]
    for roi in rois:
        roi.set_readable_format(roi.stat, Format.CHILD)
    for roi in rois:
        _switch_stats(roi.stat, on=True)
        set_mock_value(roi.stat.nd_attributes_file_nelm, 2048)
        set_mock_value(roi.size_x, 40)
        set_mock_value(roi.size_y, 30)
    set_mock_value(roi2.stat.compute_statistics, roi2_computes_stats)
    for hdf in hdfs:
        set_mock_value(hdf.file_path_exists, True)

    def capture_three_frames(value: bool) -> None:
        if value:
            for hdf in hdfs:
                set_mock_value(hdf.num_captured, 3)

    callback_on_mock_put(det.driver.acquire, capture_three_frames)
    docs: dict[str, list] = defaultdict(list)

    RE(
        _fly_plan(det, TriggerInfo(number_of_events=3)),
        lambda name, doc: docs[name].append(doc),
    )

    # Each writer's file has its own ROI and the totals of its own stats plugin
    uris = {doc["data_key"]: doc["uri"] for doc in docs["stream_resource"]}
    expected = {
        "det-roi1": ANY,
        "det-roi1-stat-total": uris["det-roi1"],
        "det-roi2": ANY,
    }
    if roi2_computes_stats:
        expected["det-roi2-stat-total"] = uris["det-roi2"]
    assert uris == expected
    assert uris["det-roi1"] != uris["det-roi2"]
    (descriptor,) = docs["descriptor"]
    assert set(descriptor["data_keys"]) == set(uris)
    # The totals stay hinted by the stats plugins that the files replace
    expected_hints = ["det-roi1-stat-total"]
    if roi2_computes_stats:
        expected_hints.append("det-roi2-stat-total")
    assert descriptor["hints"]["det"]["fields"] == [
        *expected_hints,
        "det-roi1",
        "det-roi2",
    ]


# Use case 9: an ROIStat plugin, modelled on the test-rig spectroscopy plans
# (DRV -> ROISTAT -> HDF, three channels named Blue, Green and Red, only their
# totals published). The HDF writer lists the plugin as an NDAttribute source,
# so when it serves the totals are in its file, and when it sits out they are
# read from the channels.

CHANNEL_NAMES = {1: "Blue", 2: "Green", 3: "Red"}


async def _roistat_detector(
    tmp_path, hdf_on: bool, channel_names: dict[int, str] = CHANNEL_NAMES
):
    path_provider = StaticPathProvider(
        lambda datakey_name=None: str(datakey_name), tmp_path
    )
    roistat = adcore.NDROIStatIO("PREFIX:ROISTAT:", num_channels=len(channel_names))
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector(
            "PREFIX:",
            adcore.ADDataFactory.hdf(
                path_provider, always_enable=False, ndattribute_sources=[roistat]
            ),
            plugins={"roistat": roistat},
            name="det",
        )
    set_mock_value(roistat.enable_callbacks, EnableDisable.ENABLE)
    set_mock_value(roistat.nd_attributes_file_nelm, 2048)
    for number, name in channel_names.items():
        channel = roistat.channels[number]
        set_mock_value(channel.name_, name)
        set_mock_value(channel.use, True)
        set_mock_value(channel.total, 10.0 * number)
        for scalar in (channel.min_value, channel.max_value, channel.mean_value):
            channel.set_readable_format(scalar, None)
    hdf = det.get_plugin("hdf", adcore.NDFileHDF5IO)
    set_mock_value(
        hdf.enable_callbacks, EnableDisable.ENABLE if hdf_on else EnableDisable.DISABLE
    )
    set_mock_value(hdf.file_path_exists, True)
    set_mock_value(det.driver.array_size_x, 64)
    set_mock_value(det.driver.array_size_y, 48)
    return det, roistat, hdf


def _scan_roistat_detector(RE: RunEngine, det, hdf: adcore.NDFileHDF5IO, hdf_on: bool):
    """Run a fly scan if the HDF writer serves it, otherwise a step scan."""
    docs: dict[str, list] = defaultdict(list)
    if hdf_on:
        callback_on_mock_put(
            det.driver.acquire,
            lambda value: set_mock_value(hdf.num_captured, 3) if value else None,
        )
        plan = _fly_plan(det, TriggerInfo(number_of_events=3))
    else:

        @bpp.stage_decorator([det])
        @bpp.run_decorator()
        def step_plan():
            yield from bps.trigger_and_read([det])

        plan = step_plan()
    RE(plan, lambda name, doc: docs[name].append(doc))
    return docs


@pytest.mark.parametrize("hdf_on", [True, False])
async def test_uc9_roistat_totals_are_in_the_file_or_read_from_the_channels(
    RE: RunEngine, tmp_path, hdf_on: bool
):
    det, roistat, hdf = await _roistat_detector(tmp_path, hdf_on)
    # Before stage the channels have their default names
    assert roistat.channels[1].total.name == "det-roistat-channels-1-total"

    docs = _scan_roistat_detector(RE, det, hdf, hdf_on)

    totals = [f"det-roistat-{name}-total" for name in CHANNEL_NAMES.values()]
    (descriptor,) = docs["descriptor"]
    assert set(descriptor["data_keys"]) >= set(totals)
    # Hinted whether they are in the file or read, along with the image if filed
    assert descriptor["hints"]["det"]["fields"] == (
        [*totals, "det"] if hdf_on else totals
    )
    if hdf_on:
        assert {doc["data_key"] for doc in docs["stream_resource"]} == {"det", *totals}
        assert all(key in descriptor["data_keys"] for key in totals)
        assert (
            descriptor["data_keys"]["det-roistat-Blue-total"]["external"] == "STREAM:"
        )
        # The plugin was told to stamp each total from its own channel's address
        xml = await roistat.nd_attributes_file.get_value()
        stamped = [
            (
                a.attrib["name"],
                a.attrib["source"],
                a.attrib["addr"],
                a.attrib["datatype"],
            )
            for a in ET.fromstring(xml)
        ]
        assert stamped == [
            ("det-roistat-Blue-total", "ROISTAT_TOTAL", "0", "DOUBLE"),
            ("det-roistat-Green-total", "ROISTAT_TOTAL", "1", "DOUBLE"),
            ("det-roistat-Red-total", "ROISTAT_TOTAL", "2", "DOUBLE"),
        ]
    else:
        assert docs["stream_resource"] == []
        (event,) = docs["event"]
        assert {key: event["data"][key] for key in totals} == {
            "det-roistat-Blue-total": 10.0,
            "det-roistat-Green-total": 20.0,
            "det-roistat-Red-total": 30.0,
        }


@pytest.mark.parametrize("hdf_on", [True, False])
async def test_uc9_a_roistat_channel_that_is_not_used_disappears(
    RE: RunEngine, tmp_path, hdf_on: bool
):
    det, roistat, hdf = await _roistat_detector(tmp_path, hdf_on)
    set_mock_value(roistat.channels[2].use, False)

    docs = _scan_roistat_detector(RE, det, hdf, hdf_on)

    (descriptor,) = docs["descriptor"]
    assert "det-roistat-Blue-total" in descriptor["data_keys"]
    assert "det-roistat-Red-total" in descriptor["data_keys"]
    assert not [key for key in descriptor["data_keys"] if "Green" in key]
    assert not [key for key in descriptor["hints"]["det"]["fields"] if "Green" in key]
    if hdf_on:
        assert not [
            doc for doc in docs["stream_resource"] if "Green" in doc["data_key"]
        ]
    else:
        (event,) = docs["event"]
        assert not [key for key in event["data"] if "Green" in key]
    # Its name was still taken from the PV
    assert roistat.channels[2].total.name == "det-roistat-Green-total"


async def test_uc9_roistat_channel_names_are_sanitised_at_stage(
    tmp_path,
):
    names = {1: "Ice Cream", 2: "", 3: "r.o/i-3", 4: "keep_Case9"}
    det, roistat, _ = await _roistat_detector(tmp_path, False, names)

    await det.stage()

    assert [roistat.channels[number].name for number in names] == [
        "det-roistat-Ice_Cream",
        "det-roistat-2",
        "det-roistat-r_o_i_3",
        "det-roistat-keep_Case9",
    ]
    assert roistat.channels[1].total.name == "det-roistat-Ice_Cream-total"


async def test_uc9_roistat_channels_with_the_same_sanitised_name_raise_at_stage(
    tmp_path,
):
    det, _, _ = await _roistat_detector(tmp_path, False, {1: "A B", 2: "A_B", 3: "C"})

    with pytest.raises(ValueError, match=r"more than one channel is named \['A_B'\]"):
        await det.stage()


async def test_uc9_a_roistat_channel_renamed_after_stage_is_renamed_at_the_next(
    tmp_path,
):
    det, roistat, _ = await _roistat_detector(tmp_path, False)
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()
    set_mock_value(roistat.channels[3].name_, "Infrared")

    await det.prepare(TriggerInfo())
    assert roistat.channels[3].total.name == "det-roistat-Red-total"

    await det.unstage()
    await det.stage()
    assert roistat.channels[3].total.name == "det-roistat-Infrared-total"


async def test_uc9_prepare_raises_for_a_roistat_channel_used_since_the_first(
    tmp_path,
):
    det, roistat, _ = await _roistat_detector(tmp_path, False)
    set_mock_value(roistat.channels[3].use, False)
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()
    set_mock_value(roistat.channels[3].use, True)

    with pytest.raises(
        RuntimeError, match=r"changed since the first prepare.*det-roistat-Red-total"
    ):
        await det.prepare(TriggerInfo())


# Use case 10: the time series of an ROIStat plugin, for a detector that writes no
# file. Each used channel publishes one array per statistic with a read format,
# named after the scalar its channel already reads, which it suppresses.


async def _roistat_time_series_detector(
    path_provider: StaticPathProvider | None = None,
):
    """Three channels named Blue, Green and Red, all used; with an HDF writer
    stamping the plugin's NDAttributes if given a path provider."""
    roistat = adcore.NDROIStatIO("PREFIX:ROISTAT:", num_channels=len(CHANNEL_NAMES))
    hdf_factories = (
        [
            adcore.ADDataFactory.hdf(
                path_provider, always_enable=False, ndattribute_sources=[roistat]
            )
        ]
        if path_provider
        else []
    )
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector(
            "PREFIX:",
            *hdf_factories,
            adcore.ADDataFactory.time_series(roistat),
            plugins={"roistat": roistat},
            name="det",
        )
    set_mock_value(roistat.enable_callbacks, EnableDisable.ENABLE)
    set_mock_value(roistat.nd_attributes_file_nelm, 2048)
    for number, name in CHANNEL_NAMES.items():
        set_mock_value(roistat.channels[number].ts_timestamp_nelm, 2048)
        set_mock_value(roistat.channels[number].name_, name)
        set_mock_value(roistat.channels[number].use, True)
    return det, roistat


def _fly_roistat_time_series(
    RE: RunEngine, det, roistat: adcore.NDROIStatIO, points: int, also=lambda: None
) -> dict[str, list]:
    """Behave like the IOC: ERASE_START resets the count, acquiring counts the
    points, and READ posts each channel's arrays, filled with
    10 * channel number + frame number, as for a page collected between the
    plugin's once-a-second posts. The timestamps differ per channel, which the
    IOC's do not, to show which channel's are used."""

    def post():
        for number, channel in roistat.channels.items():
            for array in [
                channel.ts_min_value,
                channel.ts_max_value,
                channel.ts_mean_value,
                channel.ts_total,
                channel.ts_net,
            ]:
                set_mock_value(
                    array,  # pyright: ignore[reportArgumentType]
                    10.0 * number + np.arange(points),
                )
            set_mock_value(
                channel.ts_timestamp,  # pyright: ignore[reportArgumentType]
                100.0 * number + np.arange(points),
            )

    def control(value):
        if value == adcore.NDROIStatTSControl.ERASE_START:
            set_mock_value(roistat.ts_current_point, 0)
        elif value == adcore.NDROIStatTSControl.READ:
            post()

    def count(value):
        if value:
            set_mock_value(roistat.ts_current_point, points)
            also()

    callback_on_mock_put(roistat.ts_control, control)
    callback_on_mock_put(det.driver.acquire, count)
    docs: dict[str, list] = defaultdict(list)
    RE(
        _fly_plan(det, TriggerInfo(number_of_events=points)),
        lambda name, doc: docs[name].append(doc),
    )
    return docs


async def test_uc10_roistat_time_series_publishes_each_used_channel(RE: RunEngine):
    det, roistat = await _roistat_time_series_detector()
    # Green is not used, Red publishes only its total, and Blue its net too
    set_mock_value(roistat.channels[2].use, False)
    red = roistat.channels[3]
    for scalar in (red.min_value, red.max_value, red.mean_value):
        red.set_readable_format(scalar, None)
    blue = roistat.channels[1]
    blue.set_readable_format(blue.net, Format.UNCACHED_SIGNAL)

    docs = _fly_roistat_time_series(RE, det, roistat, points=3)

    blue_keys = [
        f"det-roistat-Blue-{stat}"
        for stat in ("min_value", "max_value", "mean_value", "total", "net")
    ]
    keys = [*blue_keys, "det-roistat-Red-total"]
    (descriptor,) = docs["descriptor"]
    # The arrays replace the scalars of the same name, and the unused channel is gone
    assert list(descriptor["data_keys"]) == keys
    total = descriptor["data_keys"]["det-roistat-Blue-total"]
    assert total["source"] == "mock+ca://PREFIX:ROISTAT:1:TSTotal"
    assert total["shape"] == [1]
    net = descriptor["data_keys"]["det-roistat-Blue-net"]
    assert net["source"] == "mock+ca://PREFIX:ROISTAT:1:TSNet"
    # The channels hint their scalars, whoever serves them, and net is not hinted
    hinted = [key for key in keys if key != "det-roistat-Blue-net"]
    assert descriptor["hints"]["det"]["fields"] == hinted
    assert docs["event"] == []
    (page,) = docs["event_page"]
    assert list(page["data"]) == keys
    assert page["data"]["det-roistat-Blue-total"] == [[10.0], [11.0], [12.0]]
    assert page["data"]["det-roistat-Blue-net"] == [[10.0], [11.0], [12.0]]
    assert page["data"]["det-roistat-Red-total"] == [[30.0], [31.0], [32.0]]
    # Every array is timed by the first used channel's timestamps
    assert page["timestamps"]["det-roistat-Red-total"] == [100.0, 101.0, 102.0]
    # The buffer was sized to the scan, then erased and started, its arrays
    # posted for the page, and stopped on unstage
    assert get_mock_put(roistat.ts_num_points).call_args_list[-1].args == (3,)
    assert [
        call.args[0] for call in get_mock_put(roistat.ts_control).call_args_list
    ] == [
        adcore.NDROIStatTSControl.STOP,
        adcore.NDROIStatTSControl.ERASE_START,
        adcore.NDROIStatTSControl.READ,
        adcore.NDROIStatTSControl.STOP,
    ]


async def test_uc10_roistat_time_series_is_erased_only_once_sized():
    det, roistat = await _roistat_time_series_detector()
    callback_on_mock_put(
        roistat.ts_control,
        lambda value: set_mock_value(roistat.ts_current_point, 0),
    )
    await det.stage()
    controls = get_mock_put(roistat.ts_control)
    controls.reset_mock()

    with mock_puts_blocked(roistat.ts_num_points):
        status = det.prepare(TriggerInfo(number_of_events=3))
        while not get_mock_put(roistat.ts_num_points).called:
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        # The plugin keeps its current point when resized, so not erased yet
        assert not controls.called
    await status

    assert [call.args[0] for call in controls.call_args_list] == [
        adcore.NDROIStatTSControl.ERASE_START
    ]


@pytest.mark.parametrize("points,raises", [(2048, False), (2049, True)])
async def test_uc10_prepare_raises_for_a_scan_longer_than_the_roistat_buffer(
    points: int, raises: bool
):
    det, roistat = await _roistat_time_series_detector()
    # The arrays hold NCHANS points
    set_mock_value(roistat.channels[1].ts_timestamp_nelm, 2048)
    await det.stage()

    prepare = det.prepare(TriggerInfo(number_of_events=points))

    if raises:
        with pytest.raises(
            ValueError, match="needs 2049 points, but the plugin's arrays hold 2048"
        ):
            await prepare
    else:
        await prepare


@pytest.mark.parametrize(
    "enable_callbacks,used,serves",
    [
        (EnableDisable.ENABLE, True, True),
        # Off at the first prepare(): the time series sits out, and the scalars go
        (EnableDisable.DISABLE, True, False),
        (EnableDisable.ENABLE, False, False),
    ],
)
async def test_uc10_roistat_time_series_follows_the_switches_at_first_prepare(
    enable_callbacks: EnableDisable, used: bool, serves: bool
):
    det, roistat = await _roistat_time_series_detector()
    set_mock_value(roistat.enable_callbacks, enable_callbacks)
    for channel in roistat.channels.values():
        set_mock_value(channel.use, used)
    callback_on_mock_put(
        roistat.ts_control,
        lambda value: set_mock_value(roistat.ts_current_point, 0),
    )

    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=2))

    shapes = {
        key: datakey["shape"]
        for key, datakey in (await det.describe()).items()
        if "roistat" in key
    }
    assert shapes == (
        {
            f"det-roistat-{name}-{stat}": [1]
            for name in CHANNEL_NAMES.values()
            for stat in ("min_value", "max_value", "mean_value", "total")
        }
        if serves
        else {}
    )
    erase_starts = [
        call
        for call in get_mock_put(roistat.ts_control).call_args_list
        if call.args[0] == adcore.NDROIStatTSControl.ERASE_START
    ]
    assert len(erase_starts) == (1 if serves else 0)


@pytest.mark.parametrize("hdf_enabled", [True, False])
async def test_uc10_the_hdf_switch_picks_the_file_or_the_roistat_time_series(
    RE: RunEngine,
    static_path_provider: StaticPathProvider,
    hdf_enabled: bool,
):
    det, roistat = await _roistat_time_series_detector(static_path_provider)
    hdf = det.get_plugin("hdf", adcore.NDFileHDF5IO)
    set_mock_value(hdf.file_path_exists, True)
    set_mock_value(det.driver.array_size_x, 64)
    set_mock_value(det.driver.array_size_y, 48)
    set_mock_value(
        hdf.enable_callbacks,
        EnableDisable.ENABLE if hdf_enabled else EnableDisable.DISABLE,
    )
    for channel in roistat.channels.values():
        for scalar in (channel.min_value, channel.max_value, channel.mean_value):
            channel.set_readable_format(scalar, None)

    docs = _fly_roistat_time_series(
        RE, det, roistat, points=2, also=lambda: set_mock_value(hdf.num_captured, 2)
    )

    totals = [f"det-roistat-{name}-total" for name in CHANNEL_NAMES.values()]
    if hdf_enabled:
        assert docs["event_page"] == []
        keys = [doc["data_key"] for doc in docs["stream_resource"]]
        assert keys == ["det", *totals]
    else:
        assert docs["stream_resource"] == []
        (page,) = docs["event_page"]
        assert list(page["data"]) == totals


class _CustomPlugin(StandardReadable, adcore.NDPluginBaseIO):
    value: A[SignalR[float], PvSuffix("Value_RBV"), Format.HINTED_UNCACHED_SIGNAL]


async def test_a_custom_readable_plugin_keeps_its_readable_after_prepare():
    custom = _CustomPlugin("PREFIX:CUSTOM:")
    async with init_devices(mock=True):
        det = adsimdetector.SimDetector(
            "PREFIX:", plugins={"custom": custom}, name="det"
        )
    set_mock_value(custom.value, 3.0)
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()

    assert "det-custom-value" in await det.describe()
    assert (await det.read())["det-custom-value"]["value"] == 3.0
