import pytest

from ophyd_async.core import (
    StandardReadableFormat,
    StaticPathProvider,
    TriggerInfo,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics import adcore


@pytest.mark.parametrize(
    "factory",
    [
        adcore.ADDataFactory.hdf,
        adcore.ADDataFactory.jpeg,
        adcore.ADDataFactory.tiff,
    ],
)
@pytest.mark.parametrize("hinted", [True, False])
def test_writer_image_hints(static_path_provider: StaticPathProvider, factory, hinted):
    driver = adcore.ADBaseIO("PREFIX:DRV:")
    stats = adcore.NDStatsIO("PREFIX:STAT:")
    det = adcore.AreaDetector(
        driver,
        "PREFIX:",
        factory(static_path_provider, datakey_suffix="-image", hinted=hinted),
        plugins={"stats": stats},
        name="det",
    )
    # No hinted fields means no hints at all
    assert det.hints == ({"fields": ["det-image"]} if hinted else {})
    det.set_readable_format(stats.total, StandardReadableFormat.HINTED_SIGNAL)
    # Readable-format signals are hinted ahead of the data logics' fields
    assert det.hints == {
        "fields": ["det-stats-total"] + (["det-image"] if hinted else [])
    }


@pytest.mark.parametrize(
    "factory",
    [
        adcore.ADDataFactory.hdf,
        adcore.ADDataFactory.jpeg,
        adcore.ADDataFactory.tiff,
    ],
)
def test_writer_image_hints_enabled_by_default(
    static_path_provider: StaticPathProvider, factory
):
    det = adcore.AreaDetector(
        adcore.ADBaseIO("PREFIX:DRV:"),
        "PREFIX:",
        factory(static_path_provider),
        name="det",
    )
    assert det.hints == {"fields": ["det"]}


# ---------------------------------------------------------------------------
# AreaDetector.__init__ guards
# ---------------------------------------------------------------------------


def test_area_detector_requires_prefix_when_factories_given(
    static_path_provider: StaticPathProvider,
):
    driver = adcore.ADBaseIO("PREFIX:DRV:")
    with pytest.raises(
        ValueError,
        match="^prefix is required when writer_factories are given$",
    ):
        adcore.AreaDetector(
            driver,
            None,
            adcore.ADDataFactory.hdf(static_path_provider),
            name="det",
        )


def test_area_detector_rejects_duplicate_writer_names(
    static_path_provider: StaticPathProvider,
):
    driver = adcore.ADBaseIO("PREFIX:DRV:")

    with pytest.raises(
        ValueError,
        match=r"^Duplicate writer_name\(s\) in writer_factories: \['hdf'\]$",
    ):
        adcore.AreaDetector(
            driver,
            "PREFIX:",
            adcore.ADDataFactory.hdf(static_path_provider, writer_name="hdf"),
            adcore.ADDataFactory.hdf(static_path_provider, writer_name="hdf"),
            name="det",
        )


def test_plugin_named_like_a_factory_attribute_raises(
    static_path_provider: StaticPathProvider,
):
    driver = adcore.ADBaseIO("PREFIX:DRV:")

    with pytest.raises(ValueError, match=r"attribute 'stats'"):
        adcore.AreaDetector(
            driver,
            "PREFIX:",
            adcore.ADDataFactory.stats_time_series(),
            plugins={"stats": adcore.NDStatsIO("PREFIX:STATS:")},
            name="det",
        )


def test_plugin_named_like_the_driver_raises():
    driver = adcore.ADBaseIO("PREFIX:DRV:")

    with pytest.raises(ValueError, match=r"attribute 'driver'"):
        adcore.AreaDetector(
            driver, plugins={"driver": adcore.NDStatsIO("PREFIX:S:")}, name="det"
        )


def test_deprecated_writer_factory_still_builds_the_new_factories(
    static_path_provider: StaticPathProvider,
):
    with pytest.warns(DeprecationWarning, match="ADWriterFactory"):
        hdf = adcore.ADWriterFactory.hdf(static_path_provider, enable_callbacks=False)
    with pytest.warns(DeprecationWarning, match="stats_time_series"):
        stats = adcore.ADWriterFactory.stats(enable_callbacks=False)

    driver = adcore.ADBaseIO("PREFIX:DRV:")
    assert isinstance(hdf, adcore.ADDataFactory)
    for factory in (hdf, stats):
        _, logic = factory("PREFIX:", driver, [])
        assert logic.always_enable is False  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# AreaDetector.get_plugin guards
# ---------------------------------------------------------------------------


def test_get_plugin_missing_raises_attribute_error():
    driver = adcore.ADBaseIO("PREFIX:DRV:")
    det = adcore.AreaDetector(driver=driver, name="det")
    with pytest.raises(AttributeError, match="^det has no plugin named 'hdf'$"):
        det.get_plugin("hdf")


def test_get_plugin_wrong_type_raises_type_error():
    driver = adcore.ADBaseIO("PREFIX:DRV:")
    plugins = {"stats": adcore.NDStatsIO("PREFIX:STAT:")}
    det = adcore.AreaDetector(driver=driver, plugins=plugins, name="det")

    assert isinstance(det.get_plugin("stats", adcore.NDStatsIO), adcore.NDStatsIO)

    with pytest.raises(
        TypeError,
        match=r"^Expected det\.stats to be a NDPluginFileIO, got NDStatsIO$",
    ):
        det.get_plugin("stats", adcore.NDPluginFileIO)


# ---------------------------------------------------------------------------
# get_ndarray_resource_info error paths
# ---------------------------------------------------------------------------


async def test_get_ndarray_resource_info_undefined_datatype(
    static_path_provider: StaticPathProvider,
):
    async with init_devices(mock=True):
        det = adcore.AreaDetector(
            adcore.ADBaseIO("PREFIX:DRV:"),
            "PREFIX:",
            adcore.ADDataFactory.hdf(static_path_provider),
            name="det",
        )
    set_mock_value(det.driver.data_type, adcore.ADBaseDataType.UNDEFINED)
    set_mock_value(det.get_plugin("hdf", adcore.NDFileHDF5IO).file_path_exists, True)
    await det.stage()
    with pytest.raises(
        ValueError,
        match=r"^mock\+ca://PREFIX:DRV:DataType_RBV is blank, this is not supported",
    ):
        await det.prepare(TriggerInfo())


async def test_get_ndarray_resource_info_unsupported_color_mode(
    static_path_provider: StaticPathProvider,
):
    async with init_devices(mock=True):
        det = adcore.AreaDetector(
            adcore.ADBaseIO("PREFIX:DRV:"),
            "PREFIX:",
            adcore.ADDataFactory.hdf(static_path_provider),
            name="det",
        )
    set_mock_value(det.driver.color_mode, adcore.ADBaseColorMode.BAYER)
    set_mock_value(det.get_plugin("hdf", adcore.NDFileHDF5IO).file_path_exists, True)
    await det.stage()
    with pytest.raises(
        RuntimeError,
        match=r"^Unsupported ColorMode Bayer! Only Mono and RGB1 are supported\.",
    ):
        await det.prepare(TriggerInfo())
