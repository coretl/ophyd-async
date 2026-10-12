import re
from unittest.mock import call

import pytest

from ophyd_async.core import (
    DetectorTrigger,
    EnableDisable,
    StaticPathProvider,
    TriggerInfo,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics import adcore
from ophyd_async.testing import assert_has_calls


@pytest.fixture
async def cont_acq_detector() -> adcore.AreaDetector[adcore.ADBaseIO]:
    async with init_devices(mock=True):
        det = adcore.ContAcqDetector(prefix="PREFIX:")

    set_mock_value(
        det.driver.image_mode,
        adcore.ADImageMode.CONTINUOUS,
    )
    set_mock_value(det.driver.acquire_time, 0.8)
    set_mock_value(det.driver.acquire_period, 1.0)
    set_mock_value(det.driver.acquire, True)
    return det


async def test_cont_acq_controller_invalid_trigger_mode(
    cont_acq_detector: adcore.AreaDetector[adcore.ADBaseIO],
):
    trigger_info = TriggerInfo(trigger=DetectorTrigger.EXTERNAL_EDGE)
    await cont_acq_detector.stage()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Trigger type DetectorTrigger.EXTERNAL_EDGE not supported, "
            "supported types are: [INTERNAL]"
        ),
    ):
        await cont_acq_detector.prepare(trigger_info)


ERROR_MESSAGE = (
    "Driver must be acquiring in continuous mode to use the cont acq interface"
)


async def test_cont_acq_controller_not_in_continuous_mode(
    cont_acq_detector: adcore.AreaDetector[adcore.ADBaseIO],
):
    set_mock_value(cont_acq_detector.driver.image_mode, adcore.ADImageMode.SINGLE)
    await cont_acq_detector.stage()
    with pytest.raises(RuntimeError, match=ERROR_MESSAGE):
        await cont_acq_detector.trigger()


async def test_cont_acq_controller_not_acquiring(
    cont_acq_detector: adcore.AreaDetector[adcore.ADBaseIO],
):
    set_mock_value(cont_acq_detector.driver.acquire, False)
    await cont_acq_detector.stage()
    with pytest.raises(RuntimeError, match=ERROR_MESSAGE):
        await cont_acq_detector.trigger()


async def test_cont_acq_controller_invalid_exposure_time(
    cont_acq_detector: adcore.AreaDetector[adcore.ADBaseIO],
):
    await cont_acq_detector.stage()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Detector exposure time currently set to 0.8, but requested exposure is 1.0"
        ),
    ):
        await cont_acq_detector.prepare(TriggerInfo(livetime=1.0))


async def test_cont_acq_controller_success(
    cont_acq_detector: adcore.AreaDetector[adcore.ADBaseIO],
):
    await cont_acq_detector.stage()
    await cont_acq_detector.trigger()
    assert_has_calls(
        cont_acq_detector,
        [
            call.cb.capture.put(False),
            call.cb.enable_callbacks.put(EnableDisable.ENABLE),
            call.cb.pre_count.put(0),
            call.cb.post_count.put(1),
            call.cb.preset_trigger_count.put(1),
            call.cb.flush_on_soft_trg.put(adcore.NDCBFlushOnSoftTrgMode.ON_NEW_IMAGE),
            call.cb.capture.put(True),
        ],
    )


@pytest.fixture
async def cont_acq_hdf_detector(
    static_path_provider: StaticPathProvider,
) -> adcore.AreaDetector[adcore.ADBaseIO]:
    async with init_devices(mock=True):
        det = adcore.ContAcqDetector(
            "PREFIX:", adcore.ADWriterFactory.hdf(static_path_provider)
        )
    set_mock_value(det.driver.image_mode, adcore.ADImageMode.CONTINUOUS)
    set_mock_value(det.driver.acquire_time, 0.8)
    set_mock_value(det.driver.acquire_period, 1.0)
    set_mock_value(det.driver.acquire, True)
    set_mock_value(det.driver.array_size_x, 1024)
    set_mock_value(det.driver.array_size_y, 768)
    set_mock_value(det.get_plugin("hdf", adcore.NDFileHDF5IO).file_path_exists, True)
    return det


async def test_uc9_continuous_acquisition_step_then_fly_scan(
    cont_acq_hdf_detector: adcore.AreaDetector[adcore.ADBaseIO],
):
    """Use case 9: the driver stays acquiring, the buffer captures N per trigger.

    Streamable and bounded: stream assets for step and fly, the buffer sized by
    the prepare that precedes each.
    """
    det = cont_acq_hdf_detector
    cb = det.get_plugin("cb", adcore.NDCircularBuffIO)
    hdf = det.get_plugin("hdf", adcore.NDFileHDF5IO)
    captured = 0

    def capture(value: bool) -> None:
        # Each capture of the circular buffer writes this many frames
        nonlocal captured
        if value:
            captured += frames_per_capture
            set_mock_value(hdf.num_captured, captured)

    callback_on_mock_put(cb.capture, capture)

    # Step scan: two points of 2 frames each, one StreamResource
    await det.stage()
    frames_per_capture = 2
    await det.prepare(TriggerInfo(collections_per_event=2))
    await det.trigger()
    await det.trigger()
    assert await cb.post_count.get_value() == 2
    docs = [doc async for doc in det.collect_asset_docs()]  # pyright: ignore[reportAttributeAccessIssue]
    assert [name for name, _ in docs] == [
        "stream_resource",
        "stream_datum",
    ]
    assert docs[1][1]["indices"] == {"start": 0, "stop": 2}

    # Fly scan: 3 events of 2 frames, captured in one go
    frames_per_capture = 6
    await det.prepare(TriggerInfo(number_of_events=3, collections_per_event=2))
    assert await cb.post_count.get_value() == 6
    await det.kickoff()
    await det.complete()
    docs = [doc async for doc in det.collect_asset_docs()]  # pyright: ignore[reportAttributeAccessIssue]
    assert [d["indices"] for n, d in docs if n == "stream_datum"] == [
        {"start": 2, "stop": 5}
    ]
    await det.unstage()
