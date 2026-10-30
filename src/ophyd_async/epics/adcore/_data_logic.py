import asyncio
import math
import warnings
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    Array1D,
    BoundedDataLogic,
    Device,
    EnableDisable,
    EventPageDataProvider,
    PathInfo,
    PathProvider,
    SignalR,
    StandardReadable,
    StreamResourceDataProvider,
    StreamResourceInfo,
    UnboundedDataLogic,
    set_and_wait_for_value,
    wait_for_value,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import stop_busy_record

from ._io import (
    ADBaseColorMode,
    ADBaseDataType,
    ADBaseIO,
    ADFileWriteMode,
    NDArrayBaseIO,
    NDFileHDF5IO,
    NDParam,
    NDPluginBaseIO,
    NDPluginFileIO,
    NDROIStatIO,
    NDROIStatTSControl,
    NDStatsIO,
    NDStatsTSAcquireMode,
)
from ._ndattribute import (
    NDAttributeDataType,
    NDAttributeParam,
    NDAttributePvDbrType,
    ndattributes_to_xml,
)


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
    sources: Sequence[NDPluginBaseIO] = (),
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


# The formats that put a signal in read(), rather than in configuration
_READ_FORMATS = (
    Format.HINTED_SIGNAL,
    Format.UNCACHED_SIGNAL,
    Format.HINTED_UNCACHED_SIGNAL,
)


def _read_signals(device: Device) -> Iterator[SignalR]:
    """The signals a device registers to be read, down its readable children."""
    if isinstance(device, StandardReadable):
        for entry in device.get_readable_formats():
            match entry:
                case (child, Format.CHILD):
                    yield from _read_signals(child)
                case (signal, format) if format in _READ_FORMATS:
                    yield signal
                case _:
                    pass


async def published_params(plugin: NDPluginBaseIO) -> list[NDParam]:
    """The enabled params of an enabled plugin whose signal it registers to be read.

    These are what `ADHDFDataLogic` writes as NDAttributes and the time series
    data logic publishes.
    """
    enabled, params = await asyncio.gather(
        plugin_is_enabled(plugin), plugin.get_enabled_params()
    )
    read = set(_read_signals(plugin))
    return [param for param in params or () if enabled and param.signal in read]


async def plugin_is_enabled(plugin: NDPluginBaseIO) -> bool:
    """Whether a plugin is switched on, so is receiving NDArrays.

    The nearest thing to "is this plugin in the chain": a plugin with callbacks
    disabled certainly is not, though an enabled one may still have its
    `nd_array_port` pointed somewhere else.
    """
    return await plugin.enable_callbacks.get_value() is EnableDisable.ENABLE


async def _write_ndattribute_dtype_source(
    source: NDPluginBaseIO,
) -> dict[str, tuple[str, str]]:
    ndattributes = [
        NDAttributeParam(
            name=param.signal.name,
            param=param.param,
            datatype=NDAttributeDataType.DOUBLE,
            addr=param.addr,
        )
        for param in await published_params(source)
    ]
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
    """Stop the writer, and switch it on if `always_enable`. Called on stage()."""
    await stop_busy_record(writer.capture)
    if always_enable:
        await writer.enable_callbacks.set(EnableDisable.ENABLE)


# The largest HDF chunk a flush period may size. TODO: check with reviewers that
# 4 MiB suits GPFS, and whether it should be a parameter
_MAX_CHUNK_BYTES = 4 * 1024 * 1024


def _frames_per_flush(
    flush_period: float, collection_period: float, frame: StreamResourceInfo
) -> int:
    """The frames a flush period holds, capped so a chunk is `_MAX_CHUNK_BYTES`."""
    # Made with one frame per chunk, so the chunk shape is the frame's
    frame_bytes = math.prod(frame.chunk_shape) * np.dtype(frame.dtype_numpy).itemsize
    wanted = round(flush_period / collection_period)
    return max(1, min(wanted, _MAX_CHUNK_BYTES // frame_bytes))


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
    ndattribute_sources: Sequence[NDPluginBaseIO] = ()
    datakey_suffix: str = ""
    hinted: bool = True
    #: False sits out the scans of a stage if the plugin is off at the first
    #: prepare() of the stage. True switches the plugin on (EnableCallbacks) on
    #: stage(), so it serves every scan.
    always_enable: bool = False

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
            ),
            get_ndattribute_dtype_source(parsed, self.ndattribute_sources),
        )
        # Size the HDF chunk to hold a flush period of collections, e.g. 400 Hz
        # (2.5 ms) with a 0.5 s flush period gives 200 frames per chunk (#1309).
        # Otherwise use the chunk size set on the IOC, forcing a fresh-startup 0
        # to 1.
        if flush_period > 0 and collection_period > 0:
            frames_per_chunk = _frames_per_flush(
                flush_period, collection_period, main_dataset
            )
        else:
            frames_per_chunk = await self.writer.num_frames_chunks.get_value() or 1
        main_dataset.chunk_shape = (frames_per_chunk, *main_dataset.chunk_shape[1:])
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
            collections_written=self.writer.num_captured,
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
        False sits out the scans of a stage if the plugin is off at the first
        prepare() of the stage. True switches the plugin on (EnableCallbacks) on
        stage(), so it serves every scan.
    """

    array_description: NDArrayDescription
    path_provider: PathProvider
    writer: NDPluginFileIO
    extension: str
    mimetype: str
    datakey_suffix: str = ""
    hinted: bool = True
    always_enable: bool = False

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
            uri=path_info.directory_uri,
            resources=[main_dataset],
            mimetype=self.mimetype,
            collections_written=self.writer.num_captured,
            collections_per_event=collections_per_event,
        )

    async def ensure_stopped(self) -> None:
        await stop_busy_record(self.writer.capture)

    def get_hinted_fields(self, name: str) -> Sequence[str]:
        return [name + self.datakey_suffix] if self.hinted else []


async def _check_capacity(capacity: SignalR[int], num_collections: int) -> None:
    """Raise if the time series arrays hold fewer points than the scan needs."""
    points = await capacity.get_value()
    if points < num_collections:
        raise ValueError(
            f"{capacity.source}: the scan needs {num_collections} points, "
            f"but the plugin's arrays hold {points}"
        )


class TimeSeriesDataLogic(BoundedDataLogic[EventPageDataProvider]):
    """Data logic publishing the time series of a stats or ROI stat plugin.

    For detectors that write no file: the plugin holds each statistic in a
    fixed-length buffer that must be sized before acquisition, so this logic
    sizes it from the scan's number of collections. The plugin has one time-series
    control shared across many arrays, so one logic covers every array.

    It publishes the time series of each of the plugin's `published_params`
    under the name of its scalar signal, so the plugin's switches and read
    formats decide what is published, and this logic never writes them.
    It hints nothing: a hinted scalar of a plugin registered as a readable stays
    in the hints while its series replaces it.

    The buffer is sized and armed in `start_provider`, which the detector calls
    again at each step point after the first. Arming erases the arrays and resets
    the current point before the detector's acquire logic drives the camera; its
    frames then feed the time series via NDArray callbacks. `start_provider`
    raises `ValueError` if the plugin's arrays hold fewer points than the scan
    needs. Before each page the plugin is asked to post its arrays up to the
    current point. A detector that writes a file should pull stats into the file
    as NDAttributes instead (see `ADHDFDataLogic`).

    For an `NDStatsIO`, arming writes the fixed-length mode and then 1 to
    `ts_acquire`, and `ts_read` posts the arrays. For an `NDROIStatIO` it
    writes ERASE_START to `ts_control`, the plugin stops acquiring by itself
    when the buffer is full, and READ posts the arrays. Each ROI channel has its
    own timestamp array, and that of the first channel published is used.

    :param plugin: the stats or ROI stat plugin whose time series to read
    """

    def __init__(self, plugin: NDStatsIO | NDROIStatIO) -> None:
        self.plugin = plugin
        match plugin:
            case NDStatsIO():
                self._logic = _StatsTimeSeriesLogic(plugin)
            case NDROIStatIO():
                self._logic = _ROIStatTimeSeriesLogic(plugin)

    async def start_provider(
        self,
        name: str,
        existing_datakeys: Sequence[str],
        num_collections: int,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> EventPageDataProvider | None:
        # The buffer is sized by count, not rate, so neither period is used
        params = await published_params(self.plugin)
        arrays = {
            param.signal.name: param.ts_signal
            for param in params
            if param.ts_signal is not None
            and param.signal.name not in existing_datakeys
        }
        if not arrays:
            return None
        timestamp, capacity = self._logic.timestamp_and_capacity(params)
        await _check_capacity(capacity, num_collections)
        await self._logic.arm(num_collections)
        # The IOC resets the count within the put, but its monitor may arrive
        # after the put's readback, and the detector reads the count next
        await wait_for_value(self.plugin.ts_current_point, 0, timeout=DEFAULT_TIMEOUT)
        return EventPageDataProvider(
            arrays,
            timestamp,
            self.plugin.ts_current_point,
            collections_per_event,
            self._logic.post_arrays,
        )

    async def ensure_stopped(self) -> None:
        await self._logic.stop()


@dataclass(frozen=True)
class _StatsTimeSeriesLogic:
    """How an `NDStatsIO` arms, posts and stops its time series.

    Arming also enables the embedded time series plugin, which the IOC starts
    disabled.
    """

    stats: NDStatsIO

    def timestamp_and_capacity(
        self, params: Sequence[NDParam]
    ) -> tuple[SignalR[Array1D[np.float64]], SignalR[int]]:
        return self.stats.ts_timestamp, self.stats.ts_timestamp_nelm

    async def arm(self, num_collections: int) -> None:
        """Size the buffer, then erase the arrays and start filling them."""
        # Fixed-length mode before arming. Writing 1 to ts_acquire clears the
        # arrays and resets ts_current_point to 0. The put completes when the
        # buffer is full, so do not wait for it.
        # The embedded time series plugin ignores the stats until it is enabled
        await asyncio.gather(
            self.stats.ts_plugin.enable_callbacks.set(EnableDisable.ENABLE),
            self.stats.ts_num_points.set(num_collections),
            self.stats.ts_acquire_mode.set(NDStatsTSAcquireMode.FIXED_LENGTH),
        )
        await set_and_wait_for_value(
            self.stats.ts_acquire, True, wait_for_set_completion=False
        )

    async def post_arrays(self) -> None:
        """Ask the plugin to post its arrays, which it does about once a second."""
        await self.stats.ts_read.set(True)

    async def stop(self) -> None:
        await stop_busy_record(self.stats.ts_acquire)


@dataclass(frozen=True)
class _ROIStatTimeSeriesLogic:
    """How an `NDROIStatIO` arms, posts and stops its time series."""

    roistat: NDROIStatIO

    def timestamp_and_capacity(
        self, params: Sequence[NDParam]
    ) -> tuple[SignalR[Array1D[np.float64]], SignalR[int]]:
        # Each channel has its own timestamp array: use the first published
        first_channel = self.roistat.channels[params[0].addr + 1]
        return first_channel.ts_timestamp, first_channel.ts_timestamp_nelm

    async def arm(self, num_collections: int) -> None:
        """Size the buffer, then erase the arrays and start filling them."""
        # The plugin keeps the current point when resized, so size it first and
        # then erase and start, which resets the current point to 0
        await self.roistat.ts_num_points.set(num_collections)
        await self.roistat.ts_control.set(NDROIStatTSControl.ERASE_START)

    async def post_arrays(self) -> None:
        """Ask the plugin to post its arrays, which it does about once a second."""
        await self.roistat.ts_control.set(NDROIStatTSControl.READ)

    async def stop(self) -> None:
        await self.roistat.ts_control.set(NDROIStatTSControl.STOP)
        await wait_for_value(self.roistat.ts_acquiring, False, timeout=DEFAULT_TIMEOUT)


def _array_description(
    driver: ADBaseIO,
    override: NDArrayDescription | Callable[[ADBaseIO], NDArrayDescription] | None,
) -> NDArrayDescription:
    """The `override`, or the shape and type read from the `driver`."""
    if callable(override):
        description = override(driver)
    elif override is not None:
        description = override
    else:
        description = NDArrayDescription(
            shape_signals=[
                driver.array_size_z,
                driver.array_size_y,
                driver.array_size_x,
            ],
            data_type_signal=driver.data_type,
            color_mode_signal=driver.color_mode,
        )
    return description


@dataclass(kw_only=True)
class ADDataFactory:
    """Factory that builds a data logic, and the plugin it needs if any.

    Construct using the static methods `hdf`, `jpeg`, `tiff` or `time_series`,
    then pass one or more instances to `AreaDetector` as positional
    `*writer_factories` arguments, which calls `build` when it is initialised.

    :param build:
        Callable `(prefix, driver, plugins) → (named_plugin, data_logic)`,
        given the detector's PV prefix, its driver and its extra plugins, where
        `named_plugin` is the attribute name and plugin the logic needs, or
        `None` if the logic takes the plugins it uses by reference.
    """

    build: Callable[
        [str, ADBaseIO, Sequence[NDPluginBaseIO]],
        tuple[tuple[str, NDPluginBaseIO] | None, BoundedDataLogic | UnboundedDataLogic],
    ]

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
        always_enable: bool = False,
        ndattribute_sources: Sequence[NDPluginBaseIO] = (),
    ) -> "ADDataFactory":
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
        :param always_enable: False (the default) sits out the scans of a stage
            if the plugin's EnableCallbacks PV is off at the first prepare() of
            the stage, so another data logic can serve them. True switches the
            plugin on at stage, so it serves every scan.
        :param ndattribute_sources: The plugins whose NDAttributes this writer
            writes into the file, as datasets named after what each publishes,
            e.g. `det-stats-total`. The NDAttributes of the driver and
            of the detector's `plugins` that are not listed are parsed from
            their `nd_attributes_file` as the user set it.
        """

        def build(prefix, driver, plugins):
            writer = NDFileHDF5IO(prefix + writer_suffix)
            data_logic = ADHDFDataLogic(
                array_description=_array_description(driver, array_description),
                path_provider=path_provider,
                writer=writer,
                driver=driver,
                plugins=list(plugins),
                ndattribute_sources=ndattribute_sources,
                datakey_suffix=datakey_suffix,
                hinted=hinted,
                always_enable=always_enable,
            )
            return (writer_name, writer), data_logic

        return ADDataFactory(build=build)

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
        always_enable: bool = False,
    ) -> "ADDataFactory":
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
        :param always_enable: False (the default) sits out the scans of a stage
            if the plugin's EnableCallbacks PV is off at the first prepare() of
            the stage, so another data logic can serve them. True switches the
            plugin on at stage, so it serves every scan.
        """

        def build(prefix, driver, plugins):
            writer = NDPluginFileIO(prefix + writer_suffix)
            data_logic = ADMultipartDataLogic(
                array_description=_array_description(driver, array_description),
                path_provider=path_provider,
                writer=writer,
                extension=".jpg",
                mimetype="multipart/related;type=image/jpeg",
                datakey_suffix=datakey_suffix,
                hinted=hinted,
                always_enable=always_enable,
            )
            return (writer_name, writer), data_logic

        return ADDataFactory(build=build)

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
        always_enable: bool = False,
    ) -> "ADDataFactory":
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
        :param always_enable: False (the default) sits out the scans of a stage
            if the plugin's EnableCallbacks PV is off at the first prepare() of
            the stage, so another data logic can serve them. True switches the
            plugin on at stage, so it serves every scan.
        """

        def build(prefix, driver, plugins):
            writer = NDPluginFileIO(prefix + writer_suffix)
            data_logic = ADMultipartDataLogic(
                array_description=_array_description(driver, array_description),
                path_provider=path_provider,
                writer=writer,
                extension=".tiff",
                mimetype="multipart/related;type=image/tiff",
                datakey_suffix=datakey_suffix,
                hinted=hinted,
                always_enable=always_enable,
            )
            return (writer_name, writer), data_logic

        return ADDataFactory(build=build)

    @staticmethod
    def time_series(plugin: NDStatsIO | NDROIStatIO) -> "ADDataFactory":
        """Create a factory for the time series of a stats or ROI stat plugin.

        Unlike `hdf`, `jpeg` and `tiff`, this makes no plugin and writes no file:
        it reads the time series of the plugin you give it, a fixed-length buffer
        read back as event pages, for detectors that cannot write assets (see
        `TimeSeriesDataLogic`). Each series is named after its scalar signal, e.g.
        `det-stats-total` or `det-roistat-Blue-total`, and the plugin's
        switches decide which are published.

        :param plugin: The stats or ROI stat plugin, which must be in the
            detector's device tree, e.g. in `plugins`. The same plugin can be
            listed in `hdf(ndattribute_sources=)`.
        """
        return ADDataFactory(
            build=lambda prefix, driver, plugins: (
                None,
                TimeSeriesDataLogic(plugin),
            )
        )


class ADWriterFactory(ADDataFactory):
    """Deprecated name of `ADDataFactory`."""

    @staticmethod
    def hdf(*args, **kwargs) -> ADDataFactory:
        _warn_writer_factory()
        return ADDataFactory.hdf(*args, **kwargs)

    @staticmethod
    def jpeg(*args, **kwargs) -> ADDataFactory:
        _warn_writer_factory()
        return ADDataFactory.jpeg(*args, **kwargs)

    @staticmethod
    def tiff(*args, **kwargs) -> ADDataFactory:
        _warn_writer_factory()
        return ADDataFactory.tiff(*args, **kwargs)


def _warn_writer_factory() -> None:
    warnings.warn(
        DeprecationWarning(
            "'ADWriterFactory' is deprecated, use 'ADDataFactory' instead"
        ),
        stacklevel=3,
    )
