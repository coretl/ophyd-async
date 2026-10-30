import asyncio
import warnings
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Any, Generic
from xml.etree import ElementTree as ET

import numpy as np

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    BoundedDataLogic,
    EnableDisable,
    PathInfo,
    PathProvider,
    SignalR,
    StreamResourceDataProvider,
    StreamResourceInfo,
    UnboundedDataLogic,
    set_and_wait_for_value,
    wait_for_value,
)
from ophyd_async.epics.core import stop_busy_record

from ._io import (
    ADBaseColorMode,
    ADBaseDataType,
    ADBaseIO,
    ADFileWriteMode,
    NDArrayBaseIO,
    NDFileHDF5IO,
    NDPluginBaseIO,
    NDPluginBaseIOT,
    NDPluginFileIO,
    NDStatsIO,
    plugin_is_enabled,
)
from ._ndattribute import (
    NDAttributeDataType,
    NDAttributePvDbrType,
    NDAttributeSource,
    ndattributes_to_xml,
)
from ._stats_time_series import StatsTimeSeriesDataLogic


@dataclass
class NDArrayDescription:
    """Signals that describe the shape and data type of an NDArray frame.

    :param shape_signals: Signals providing the frame dimensions (e.g.
        ``size_y``, ``size_x``).  Zero-valued entries are filtered out, so it
        is safe to include ``array_size_z`` for 2-D detectors.
    :param data_type_signal: Signal providing the pixel data type.
    :param color_mode_signal: Signal providing the colour mode (MONO or RGB1).
    """

    shape_signals: Sequence[SignalR[int]]
    data_type_signal: SignalR[ADBaseDataType]
    color_mode_signal: SignalR[ADBaseColorMode]


async def get_ndarray_resource_info(
    array_description: NDArrayDescription,
    data_key: str,
    parameters: dict[str, Any],
    frames_per_chunk: int = 1,
) -> StreamResourceInfo:
    # Grab the dimensions and datatype of the NDArray
    shape, datatype, color_mode = await asyncio.gather(
        asyncio.gather(*[sig.get_value() for sig in array_description.shape_signals]),
        array_description.data_type_signal.get_value(),
        array_description.color_mode_signal.get_value(),
    )
    # Remove entries in shape that are zero
    shape = [x for x in shape if x > 0]
    if datatype is ADBaseDataType.UNDEFINED:
        raise ValueError(
            f"{array_description.data_type_signal.source} is blank, "
            "this is not supported"
        )
    if color_mode == ADBaseColorMode.RGB1:
        shape = [3, *shape]
    elif color_mode != ADBaseColorMode.MONO:
        raise RuntimeError(
            f"Unsupported ColorMode {color_mode}! Only Mono and RGB1 are supported."
        )
    return StreamResourceInfo(
        data_key=data_key,
        shape=tuple(shape),
        chunk_shape=(frames_per_chunk, *shape),
        dtype_numpy=np.dtype(datatype.value.lower()).str,
        parameters=parameters,
    )


async def get_ndattribute_dtype_source(
    elements: Sequence[NDArrayBaseIO],
    sources: Sequence[NDAttributeSource] = (),
) -> dict[str, tuple[str, str]]:
    """The dtype and source of each NDAttribute stamped on a frame.

    :param elements: parse the NDAttributes in the `nd_attributes_file` of each
    :param sources: write the NDAttributes of each to its `nd_attributes_file`
    """
    per_element = await asyncio.gather(
        *(_parse_ndattribute_dtype_source(element) for element in elements),
        *(_write_ndattribute_dtype_source(source) for source in sources),
    )
    return {name: value for found in per_element for name, value in found.items()}


async def _write_ndattribute_dtype_source(
    source: NDAttributeSource,
) -> dict[str, tuple[str, str]]:
    ndattributes = source.get_ndattributes()
    await source.nd_attributes_file.set(ndattributes_to_xml(ndattributes))
    return {
        ndattribute.name: (ndattribute.datatype.value, "")
        for ndattribute in ndattributes
    }


async def _parse_ndattribute_dtype_source(
    element: NDArrayBaseIO,
) -> dict[str, tuple[str, str]]:
    maybe_xml = await element.nd_attributes_file.get_value()
    ndattribute_dtypes: dict[str, tuple[str, str]] = {}
    # This is the check that ADCore does to see if it is an XML string
    # rather than a filename to parse
    if "<Attributes>" in maybe_xml:
        root = ET.fromstring(maybe_xml)
        for child in root:
            if child.attrib.get("type", "EPICS_PV") == "EPICS_PV":
                dbrtype = child.attrib.get("dbrtype", "DBR_NATIVE")
                if dbrtype == "DBR_NATIVE":
                    raise RuntimeError(
                        f"NDAttribute {child.attrib['name']} has dbrtype "
                        "DBR_NATIVE, which is not supported"
                    )
                dtype_numpy = NDAttributePvDbrType[dbrtype].value
                source = "ca://" + child.attrib["source"]
            else:
                datatype = child.attrib.get("datatype", "INT")
                dtype_numpy = NDAttributeDataType[datatype].value
                source = ""
            ndattribute_dtypes[child.attrib["name"]] = (dtype_numpy, source)
    return ndattribute_dtypes


async def prepare_file_paths(
    path_info: PathInfo, file_template: str, writer: NDPluginFileIO
):
    # Set the directory creation depth first, since dir creation callback happens
    # when directory path PV is processed.
    await writer.create_directory.set(path_info.create_dir_depth)
    # When setting the path for windows based AD IOCs, areaDetector adds a '/'
    # rather than a '\\', which will cause the readback to never register the
    # same value.
    # Ensure that trailing separator is added to the directory path to avoid this.
    if isinstance(path_info.directory_path, PureWindowsPath):
        directory_path = f"{path_info.directory_path}\\"
    else:
        directory_path = f"{path_info.directory_path}/"
    await asyncio.gather(
        writer.file_path.set(directory_path),
        writer.file_name.set(path_info.filename),
        writer.file_template.set(file_template),
        writer.auto_increment.set(True),
        writer.file_number.set(0),
        writer.file_write_mode.set(ADFileWriteMode.STREAM),
    )
    # Check the path exists on the host
    if not await writer.file_path_exists.get_value():
        msg = f"Path {directory_path} doesn't exist or not writable!"
        raise FileNotFoundError(msg)
    # Overwrite num_capture to go forever
    await writer.num_capture.set(0)


async def _ensure_writer_ready(writer: NDPluginFileIO, always_enable: bool) -> None:
    """Stop the writer, and switch it on if `always_enable`.

    Called at stage, before `AreaDetector` records the plugins' switches.
    """
    await stop_busy_record(writer.capture)
    if always_enable:
        await writer.enable_callbacks.set(EnableDisable.ENABLE)


async def _start_capture(writer: NDPluginFileIO) -> None:
    await set_and_wait_for_value(writer.capture, True, wait_for_set_completion=False)
    # Capturing resets NumCaptured, which the detector reads as its baseline
    await wait_for_value(writer.num_captured, 0, timeout=DEFAULT_TIMEOUT)


@dataclass
class ADHDFDataLogic(UnboundedDataLogic[StreamResourceDataProvider]):
    """Data logic for AreaDetector HDF5 writer plugin.

    :param array_description: Signals describing the NDArray shape and data type.
    :param path_provider: Callable that provides path information for file writing.
    :param writer: The NDFileHDFIO plugin instance.
    :param driver: The AreaDetector driver instance to extract NDAttributes from.
    :param plugins: Additional NDPluginBaseIO instances to parse NDAttributes
        from, in the XML the user set on each.
    :param ndattribute_sources: Plugins to write the NDAttributes of, named after
        what they publish. A plugin in `plugins` that is also listed here is
        written, not parsed.
    :param datakey_suffix: Suffix to append to the data key for the main dataset
    :param hinted: Include the image in hints. Set False for BestEffortCallback.
    """

    array_description: NDArrayDescription
    path_provider: PathProvider
    writer: NDFileHDF5IO
    driver: NDArrayBaseIO | None = None
    plugins: Sequence[NDPluginBaseIO] = ()
    ndattribute_sources: Sequence[NDAttributeSource] = ()
    datakey_suffix: str = ""
    hinted: bool = True
    #: True switches the plugin on (EnableCallbacks) at stage, so it serves every
    #: scan. False sits out the scans of a stage if the plugin is off at stage.
    always_enable: bool = True

    def __post_init__(self) -> None:
        self.collections_written = self.writer.num_captured

    async def ensure_ready(self) -> None:
        await _ensure_writer_ready(self.writer, self.always_enable)

    async def start_provider(
        self,
        name: str,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> StreamResourceDataProvider | None:
        if not self.always_enable and not await plugin_is_enabled(self.writer):
            return None
        datakey_name = name + self.datakey_suffix
        path_info = self.path_provider(datakey_name)
        # Size the HDF chunk to hold a flush period of collections, e.g. 400 Hz
        # (2.5 ms) with a 0.5 s flush period gives 200 frames per chunk (#1309).
        # Otherwise use the chunk size set on the IOC, forcing a fresh-startup 0
        # to 1.
        if flush_period > 0 and collection_period > 0:
            frames_per_chunk = max(1, round(flush_period / collection_period))
        else:
            frames_per_chunk = await self.writer.num_frames_chunks.get_value() or 1
        elements = (self.driver, *self.plugins) if self.driver else (*self.plugins,)
        parsed = [
            element
            for element in elements
            if not any(element is source for source in self.ndattribute_sources)
        ]
        main_dataset, ndattribute_dtype_sources = await asyncio.gather(
            get_ndarray_resource_info(
                array_description=self.array_description,
                data_key=datakey_name,
                parameters={"dataset": "/entry/data/data"},
                frames_per_chunk=frames_per_chunk,
            ),
            get_ndattribute_dtype_source(parsed, self.ndattribute_sources),
        )
        ndattribute_datasets = [
            StreamResourceInfo(
                data_key=attr_name,
                shape=(),
                # NDAttributes appear to always be configured with
                # this chunk size
                chunk_shape=(16384,),
                dtype_numpy=dtype_numpy,
                source=source,
                parameters={"dataset": f"/entry/instrument/NDAttributes/{attr_name}"},
            )
            for attr_name, (dtype_numpy, source) in ndattribute_dtype_sources.items()
        ]
        # Write back the chunk size the main dataset is described with, so the
        # file matches the chunk_shape in its stream resource either way
        await self.writer.num_frames_chunks.set(frames_per_chunk)
        # Setup the HDF writer
        coros: list[Awaitable] = [
            self.writer.chunk_size_auto.set(True),
            self.writer.num_extra_dims.set(0),
            self.writer.lazy_open.set(True),
            self.writer.swmr_mode.set(True),
            self.writer.xml_file_name.set(""),
            prepare_file_paths(
                path_info=path_info, file_template="%s%s.h5", writer=self.writer
            ),
        ]
        await asyncio.gather(*coros)
        await _start_capture(self.writer)
        return StreamResourceDataProvider(
            uri=f"{path_info.directory_uri}{path_info.filename}.h5",
            resources=[main_dataset] + ndattribute_datasets,
            mimetype="application/x-hdf5",
            collections_per_event=collections_per_event,
            flush_signal=self.writer.flush_now,
        )

    async def ensure_stopped(self) -> None:
        await stop_busy_record(self.writer.capture)

    def get_hinted_fields(self, name: str) -> Sequence[str]:
        return [name + self.datakey_suffix] if self.hinted else []


@dataclass
class ADMultipartDataLogic(UnboundedDataLogic[StreamResourceDataProvider]):
    """Data logic for multipart AreaDetector file writers (e.g. JPEG, TIFF).

    :param array_description: Signals describing the NDArray shape and data type.
    :param path_provider: Callable that provides path information for file writing.
    :param writer: The NDFilePluginIO instance.
    :param extension: File extension for the written files (e.g. ".jpg", ".tiff").
    :param mimetype:
        Mimetype for the written files (e.g. "multipart/related;type=image/jpeg").
    :param datakey_suffix: Suffix to append to the data key for the main dataset
    :param hinted: Include the image in hints. Set False for BestEffortCallback.
    :param always_enable:
        True switches the plugin on (EnableCallbacks) at stage, so it serves
        every scan. False sits out the scans of a stage if the plugin is off at
        stage.
    """

    array_description: NDArrayDescription
    path_provider: PathProvider
    writer: NDPluginFileIO
    extension: str
    mimetype: str
    datakey_suffix: str = ""
    hinted: bool = True
    always_enable: bool = True

    def __post_init__(self) -> None:
        self.collections_written = self.writer.num_captured

    async def ensure_ready(self) -> None:
        await _ensure_writer_ready(self.writer, self.always_enable)

    async def start_provider(
        self,
        name: str,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> StreamResourceDataProvider | None:
        # A multipart writer writes one file per frame, so there is nothing to
        # size from the periods
        if not self.always_enable and not await plugin_is_enabled(self.writer):
            return None
        datakey_name = name + self.datakey_suffix
        path_info = self.path_provider(datakey_name)
        main_dataset = await get_ndarray_resource_info(
            array_description=self.array_description,
            data_key=datakey_name,
            parameters={"template": path_info.filename + "_{:06d}" + self.extension},
        )
        # Setup the file writer
        await asyncio.gather(
            self.writer.lazy_open.set(True),
            prepare_file_paths(
                path_info=path_info,
                file_template="%s%s_%6.6d" + self.extension,
                writer=self.writer,
            ),
        )
        await _start_capture(self.writer)
        return StreamResourceDataProvider(
            # TODO: remove the type ignore after
            # https://github.com/bluesky/ophyd-async/issues/1186
            uri=path_info.directory_uri,
            resources=[main_dataset],
            mimetype=self.mimetype,
            collections_per_event=collections_per_event,
        )

    async def ensure_stopped(self) -> None:
        await stop_busy_record(self.writer.capture)

    def get_hinted_fields(self, name: str) -> Sequence[str]:
        return [name + self.datakey_suffix] if self.hinted else []


@dataclass
class ADDataFactory(Generic[NDPluginBaseIOT]):
    """Factory that builds a data logic, and the writer plugin it needs if any.

    Construct using the static methods `hdf`, `jpeg`, `tiff` or
    `stats_time_series`, then pass one or more instances to `AreaDetector` as
    positional `*writer_factories` arguments.  When the detector is initialised
    `__call__` is invoked with the detector's PV `prefix`, its `driver`, and the
    flat list of extra `plugins`; it returns the writer device, or `None` if the
    factory makes none, and the corresponding data logic.

    Only the factories of file writers make a device. The others take the
    plugins they use by reference.

    :param writer_cls:
        Concrete `NDPluginBaseIO` subclass to instantiate, or `None` if the
        factory makes no plugin.
    :param writer_suffix: PV suffix appended to *prefix* to form the writer's PV prefix.
    :param writer_name:
        Attribute name under which the writer device is stored on the
        `AreaDetector` instance.  Each static method defaults to its own name
        (``"hdf"``, ``"jpeg"``, ``"tiff"``); override when passing multiple
        factories so each writer gets a distinct name. Unused without a
        `writer_cls`.
    :param datakey_suffix: Suffix appended to the datakey name in stream resources.
    :param array_description:
        Override the array shape/type description built from the driver.
        May be an `NDArrayDescription` instance or a callable
        ``(driver) → NDArrayDescription``; use a callable when the description
        depends on signals that are only available once the driver has been
        constructed (e.g. inside a detector subclass that creates its own driver).
    :param data_logic_factory:
        Callable ``(writer, array_description, driver, plugins) → data logic``
        that builds the data logic given the already-constructed writer.
    :param ndattribute_sources:
        The plugins whose NDAttributes the data logic writes. `AreaDetector`
        checks that they are in its device tree and that no two factories list
        the same one.
    :param plugin:
        The plugin the data logic reads, when the factory makes none and the
        logic is built from it. `AreaDetector` checks it is in its device tree.
    """

    writer_cls: type[NDPluginBaseIOT] | None
    writer_suffix: str
    writer_name: str
    datakey_suffix: str
    array_description: (
        NDArrayDescription | Callable[[ADBaseIO], NDArrayDescription] | None
    )
    data_logic_factory: Callable[
        [NDPluginBaseIOT, NDArrayDescription, ADBaseIO, Sequence[NDPluginBaseIO]],
        BoundedDataLogic | UnboundedDataLogic,
    ]
    ndattribute_sources: Sequence[NDAttributeSource] = ()
    plugin: NDPluginBaseIOT | None = None

    def __call__(
        self,
        prefix: str,
        driver: ADBaseIO,
        plugins: Sequence[NDPluginBaseIO],
    ) -> tuple[NDPluginBaseIOT | None, BoundedDataLogic | UnboundedDataLogic]:
        """Instantiate the writer plugin, if any, and build the data logic.

        :param prefix: EPICS PV prefix for the detector (same as `AreaDetector.prefix`).
        :param driver: The detector driver, used to read array shape/type metadata.
        :param plugins: Additional plugins whose NDAttribute XML files should be
            included in HDF5 metadata.
        :return: ``(writer, data_logic)`` tuple ready to attach to the detector,
            where ``writer`` is `None` if the factory makes no plugin.
        """
        writer = (
            self.writer_cls(prefix + self.writer_suffix) if self.writer_cls else None
        )
        used = self.plugin if writer is None else writer
        if used is None:
            raise ValueError("A data factory needs a writer_cls or a plugin")
        if callable(self.array_description):
            array_description = self.array_description(driver)
        elif self.array_description is not None:
            array_description = self.array_description
        else:
            array_description = NDArrayDescription(
                shape_signals=[
                    driver.array_size_z,
                    driver.array_size_y,
                    driver.array_size_x,
                ],
                data_type_signal=driver.data_type,
                color_mode_signal=driver.color_mode,
            )
        data_logic = self.data_logic_factory(used, array_description, driver, plugins)
        return writer, data_logic

    @staticmethod
    def hdf(
        path_provider: PathProvider,
        writer_suffix: str = "HDF1:",
        writer_name: str = "hdf",
        datakey_suffix: str = "",
        array_description: NDArrayDescription
        | Callable[[ADBaseIO], NDArrayDescription]
        | None = None,
        hinted: bool = True,
        always_enable: bool = True,
        ndattribute_sources: Sequence[NDAttributeSource] = (),
    ) -> "ADDataFactory[NDFileHDF5IO]":
        """Create a factory for an HDF5 file writer.

        :param path_provider: Provides file path information for each acquisition.
        :param writer_suffix: PV suffix for the NDFileHDF5 plugin, defaults to
            ``HDF1:``.
        :param writer_name:
            Attribute name for the writer on the detector, defaults to
            ``"hdf"``.
        :param datakey_suffix: Suffix appended to the datakey name, defaults to ``""``.
        :param array_description:
            Override the array shape/type description built from the driver.
            Pass an `NDArrayDescription` or a callable ``(driver) → NDArrayDescription``
            when the shape/type comes from a plugin rather than the main driver
            (e.g. an ROI plugin).
        :param hinted: Include the image in hints. Set False for BestEffortCallback.
        :param always_enable: True (the default) switches the plugin on at
            stage, so it serves every scan. False sits out the scans of a stage
            if the plugin's EnableCallbacks PV is off at stage, so another data
            logic can serve them.
        :param ndattribute_sources: The plugins whose NDAttributes this writer
            writes into the file, as datasets named after what each publishes,
            e.g. ``det-stats-total``. Each must be in the detector's device tree
            and be listed by one writer only. The NDAttributes of the driver and
            of the detector's `plugins` that are not listed are parsed from
            their ``nd_attributes_file`` as the user set it.
        """
        return ADDataFactory(
            writer_cls=NDFileHDF5IO,
            writer_suffix=writer_suffix,
            writer_name=writer_name,
            datakey_suffix=datakey_suffix,
            array_description=array_description,
            data_logic_factory=lambda writer, desc, driver, plugins: ADHDFDataLogic(
                array_description=desc,
                path_provider=path_provider,
                writer=writer,
                driver=driver,
                plugins=list(plugins),
                ndattribute_sources=ndattribute_sources,
                datakey_suffix=datakey_suffix,
                hinted=hinted,
                always_enable=always_enable,
            ),
            ndattribute_sources=ndattribute_sources,
        )

    @staticmethod
    def jpeg(
        path_provider: PathProvider,
        writer_suffix: str = "JPEG1:",
        writer_name: str = "jpeg",
        datakey_suffix: str = "",
        array_description: NDArrayDescription
        | Callable[[ADBaseIO], NDArrayDescription]
        | None = None,
        hinted: bool = True,
        always_enable: bool = True,
    ) -> "ADDataFactory[NDPluginFileIO]":
        """Create a factory for a JPEG file writer.

        :param path_provider: Provides file path information for each acquisition.
        :param writer_suffix: PV suffix for the NDPluginFile plugin, defaults to
            ``JPEG1:``.
        :param writer_name:
            Attribute name for the writer on the detector, defaults to
            ``"jpeg"``.
        :param datakey_suffix: Suffix appended to the datakey name, defaults to ``""``.
        :param array_description:
            Override the array shape/type description built from the driver.
            Pass an `NDArrayDescription` or a callable ``(driver) → NDArrayDescription``
            when the shape/type comes from a plugin.
        :param hinted: Include the image in hints. Set False for BestEffortCallback.
        :param always_enable: True (the default) switches the plugin on at
            stage, so it serves every scan. False sits out the scans of a stage
            if the plugin's EnableCallbacks PV is off at stage, so another data
            logic can serve them.
        """
        return ADDataFactory(
            writer_cls=NDPluginFileIO,
            writer_suffix=writer_suffix,
            writer_name=writer_name,
            datakey_suffix=datakey_suffix,
            array_description=array_description,
            data_logic_factory=lambda writer, desc, driver, plugins: (
                ADMultipartDataLogic(
                    array_description=desc,
                    path_provider=path_provider,
                    writer=writer,
                    extension=".jpg",
                    mimetype="multipart/related;type=image/jpeg",
                    datakey_suffix=datakey_suffix,
                    hinted=hinted,
                    always_enable=always_enable,
                )
            ),
        )

    @staticmethod
    def tiff(
        path_provider: PathProvider,
        writer_suffix: str = "TIFF1:",
        writer_name: str = "tiff",
        datakey_suffix: str = "",
        array_description: NDArrayDescription
        | Callable[[ADBaseIO], NDArrayDescription]
        | None = None,
        hinted: bool = True,
        always_enable: bool = True,
    ) -> "ADDataFactory[NDPluginFileIO]":
        """Create a factory for a TIFF file writer.

        :param path_provider: Provides file path information for each acquisition.
        :param writer_suffix: PV suffix for the NDPluginFile plugin, defaults to
            ``TIFF1:``.
        :param writer_name:
            Attribute name for the writer on the detector, defaults to
            ``"tiff"``.
        :param datakey_suffix: Suffix appended to the datakey name, defaults to ``""``.
        :param array_description:
            Override the array shape/type description built from the driver.
            Pass an `NDArrayDescription` or a callable ``(driver) → NDArrayDescription``
            when the shape/type comes from a plugin.
        :param hinted: Include the image in hints. Set False for BestEffortCallback.
        :param always_enable: True (the default) switches the plugin on at
            stage, so it serves every scan. False sits out the scans of a stage
            if the plugin's EnableCallbacks PV is off at stage, so another data
            logic can serve them.
        """
        return ADDataFactory(
            writer_cls=NDPluginFileIO,
            writer_suffix=writer_suffix,
            writer_name=writer_name,
            datakey_suffix=datakey_suffix,
            array_description=array_description,
            data_logic_factory=lambda writer, desc, driver, plugins: (
                ADMultipartDataLogic(
                    array_description=desc,
                    path_provider=path_provider,
                    writer=writer,
                    extension=".tiff",
                    mimetype="multipart/related;type=image/tiff",
                    datakey_suffix=datakey_suffix,
                    hinted=hinted,
                    always_enable=always_enable,
                )
            ),
        )

    @staticmethod
    def stats_time_series(stats: NDStatsIO) -> "ADDataFactory[NDStatsIO]":
        """Create a factory for an NDPluginStats time series.

        Unlike `hdf`, `jpeg` and `tiff`, this makes no plugin and writes no file:
        it reads the time series of a stats plugin you give it, a fixed-length
        buffer read back as event pages, for detectors that cannot write assets
        (see `StatsTimeSeriesDataLogic`). Each series is named after its scalar
        signal, e.g. ``det-stats-total``, and the plugin's EnableCallbacks and
        compute switches decide which are published.

        :param stats: The stats plugin, which must be in the detector's device
            tree, e.g. in its `plugins`. `AreaDetector` raises if it is not.
            The same plugin can be listed in ``hdf(ndattribute_sources=)``.
        """
        return ADDataFactory(
            writer_cls=None,
            writer_suffix="",
            writer_name="",
            datakey_suffix="",
            array_description=None,
            data_logic_factory=lambda used, desc, driver, plugins: (
                StatsTimeSeriesDataLogic(stats=used)
            ),
            plugin=stats,
        )


class ADWriterFactory(ADDataFactory[NDPluginBaseIOT]):
    """Deprecated name of `ADDataFactory`."""

    def __post_init__(self) -> None:
        _warn_writer_factory()

    @staticmethod
    def hdf(*args, **kwargs) -> ADDataFactory[NDFileHDF5IO]:
        _warn_writer_factory()
        return ADDataFactory.hdf(*args, **kwargs)

    @staticmethod
    def jpeg(*args, **kwargs) -> ADDataFactory[NDPluginFileIO]:
        _warn_writer_factory()
        return ADDataFactory.jpeg(*args, **kwargs)

    @staticmethod
    def tiff(*args, **kwargs) -> ADDataFactory[NDPluginFileIO]:
        _warn_writer_factory()
        return ADDataFactory.tiff(*args, **kwargs)


def _warn_writer_factory() -> None:
    warnings.warn(
        DeprecationWarning(
            "'ADWriterFactory' is deprecated, use 'ADDataFactory' instead"
        ),
        stacklevel=3,
    )
