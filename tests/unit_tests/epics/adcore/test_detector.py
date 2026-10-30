import pytest

from ophyd_async.core import (
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
    stats = adcore.NDStatsIO("PREFIX:STAT:")
    det_with_stats = adcore.AreaDetector(
        adcore.ADBaseIO("PREFIX:DRV:"),
        "PREFIX:",
        factory(static_path_provider, datakey_suffix="-image", hinted=hinted),
        plugins={"stats": stats},
        name="det",
    )
    det = adcore.AreaDetector(
        adcore.ADBaseIO("PREFIX:DRV2:"),
        "PREFIX:",
        factory(static_path_provider, datakey_suffix="-image", hinted=hinted),
        name="det2",
    )
    # No hinted fields means no hints at all
    assert det.hints == ({"fields": ["det2-image"]} if hinted else {})
    # A plugin in plugins= is a readable, hinted ahead of the data logics' fields
    assert det_with_stats.hints == {
        "fields": [
            "det-stats-min_value",
            "det-stats-max_value",
            "det-stats-mean_value",
            "det-stats-total",
            *(["det-image"] if hinted else []),
        ]
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
            adcore.ADDataFactory.hdf(static_path_provider, writer_name="stats"),
            plugins={"stats": adcore.NDStatsIO("PREFIX:STATS:")},
            name="det",
        )


def test_plugin_named_like_the_driver_raises():
    driver = adcore.ADBaseIO("PREFIX:DRV:")

    with pytest.raises(ValueError, match=r"attribute 'driver'"):
        adcore.AreaDetector(
            driver, plugins={"driver": adcore.NDStatsIO("PREFIX:S:")}, name="det"
        )


def test_area_detector_rejects_an_ndattribute_source_listed_twice(
    static_path_provider: StaticPathProvider,
):
    stats = adcore.NDStatsIO("PREFIX:STATS:")

    with pytest.raises(
        ValueError, match="listed more than once in ndattribute_sources"
    ):
        adcore.AreaDetector(
            adcore.ADBaseIO("PREFIX:DRV:"),
            "PREFIX:",
            adcore.ADDataFactory.hdf(
                static_path_provider, writer_name="hdf1", ndattribute_sources=[stats]
            ),
            adcore.ADDataFactory.hdf(
                static_path_provider, writer_name="hdf2", ndattribute_sources=[stats]
            ),
            plugins={"stats": stats},
            name="det",
        )


def test_area_detector_rejects_an_ndattribute_source_outside_its_device_tree(
    static_path_provider: StaticPathProvider,
):
    stats = adcore.NDStatsIO("PREFIX:STATS:")

    with pytest.raises(ValueError, match="not in the device tree of the detector"):
        adcore.AreaDetector(
            adcore.ADBaseIO("PREFIX:DRV:"),
            "PREFIX:",
            adcore.ADDataFactory.hdf(static_path_provider, ndattribute_sources=[stats]),
            name="det",
        )


@pytest.mark.parametrize("in_tree", [False, True])
def test_stats_time_series_takes_a_plugin_that_is_in_the_device_tree(in_tree: bool):
    stats = adcore.NDStatsIO("PREFIX:STATS:")
    plugins = {"stats": stats} if in_tree else {}

    def make_detector():
        return adcore.AreaDetector(
            adcore.ADBaseIO("PREFIX:DRV:"),
            "PREFIX:",
            adcore.ADDataFactory.stats_time_series(stats),
            plugins=plugins,
            name="det",
        )

    if in_tree:
        det = make_detector()
        # It made no plugin of its own
        assert [name for name, _ in det.children()] == ["driver", "stats"]
    else:
        with pytest.raises(ValueError, match="not in the device tree of the detector"):
            make_detector()


def test_a_data_factory_needs_a_writer_class_or_a_plugin():
    factory = adcore.ADDataFactory(
        writer_cls=None,
        writer_suffix="",
        writer_name="",
        datakey_suffix="",
        array_description=None,
        data_logic_factory=lambda used, desc, driver, plugins: None,  # type: ignore
    )

    with pytest.raises(ValueError, match="needs a writer_cls or a plugin"):
        factory("PREFIX:", adcore.ADBaseIO("PREFIX:DRV:"), [])


def test_area_detector_accepts_an_ndattribute_source_below_a_plugin(
    static_path_provider: StaticPathProvider,
):
    roi = adcore.NDROIIO("PREFIX:ROI:")
    roi.stat = adcore.NDStatsIO("PREFIX:STATS:")

    adcore.AreaDetector(
        adcore.ADBaseIO("PREFIX:DRV:"),
        "PREFIX:",
        adcore.ADDataFactory.hdf(static_path_provider, ndattribute_sources=[roi.stat]),
        plugins={"roi": roi},
        name="det",
    )


def test_deprecated_writer_factory_still_builds_the_new_factories(
    static_path_provider: StaticPathProvider,
):
    with pytest.warns(DeprecationWarning, match="ADWriterFactory"):
        hdf = adcore.ADWriterFactory.hdf(static_path_provider)
    assert isinstance(hdf, adcore.ADDataFactory)


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


@pytest.fixture
async def ad_with_stats_and_roi():
    async with init_devices(mock=True):
        stats = adcore.NDStatsIO("PREFIX:STAT:")
        roi = adcore.NDROIIO("PREFIX:ROI:")
        det = adcore.AreaDetector(
            adcore.ADBaseIO("PREFIX:DRV:"),
            plugins={
                "stats": stats,
                "roi": roi,
            },
            name="det",
        )
    return det, stats, roi


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
