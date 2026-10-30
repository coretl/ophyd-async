import asyncio
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated as A
from typing import Any, TypeVar

import numpy as np

from ophyd_async.core import (
    Array1D,
    DeviceVector,
    EnableDisable,
    SignalR,
    SignalRW,
    StandardReadable,
    StrictEnum,
    SubsetEnum,
    SupersetEnum,
    TriggerableCommand,
    non_zero,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, EpicsOptions, PvSuffix

from ._ndattribute import NDAttributeDataType, NDAttributeParam

# Common classes for drivers and plugins


class ADBaseDataType(SupersetEnum):
    INT8 = "Int8"
    UINT8 = "UInt8"
    INT16 = "Int16"
    UINT16 = "UInt16"
    INT32 = "Int32"
    UINT32 = "UInt32"
    INT64 = "Int64"
    UINT64 = "UInt64"
    FLOAT32 = "Float32"
    FLOAT64 = "Float64"
    # Driver database override will blank the enum string if it doesn't
    # support a datatype
    UNDEFINED = ""
    # The NDProcess plugin adds an automatic datatype option,
    # which will set the data type to whatever is necessary to hold the data
    # after processing.
    AUTOMATIC = "Automatic"


class ADBaseColorMode(SupersetEnum):
    MONO = "Mono"
    BAYER = "Bayer"
    RGB1 = "RGB1"
    RGB2 = "RGB2"
    RGB3 = "RGB3"
    YUV444 = "YUV444"
    YUV422 = "YUV422"
    YUV421 = "YUV421"


class NDArrayBaseIO(EpicsDevice):
    """Class responsible for passing detector data from drivers to plugins.

    This mirrors the interface provided by ADCore/db/NDArrayBase.template.
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/NDArray.html
    """

    port_name: A[SignalR[str], PvSuffix("PortName_RBV")]
    unique_id: A[SignalR[int], PvSuffix("UniqueId_RBV")]
    nd_attributes_file: A[SignalRW[str], PvSuffix("NDAttributesFile")]

    # The array_size_x/y/z signals are used when outputting an NDArray
    # (i.e. from a driver)
    array_size_x: A[SignalR[int], PvSuffix("ArraySizeX_RBV")]
    array_size_y: A[SignalR[int], PvSuffix("ArraySizeY_RBV")]
    array_size_z: A[SignalR[int], PvSuffix("ArraySizeZ_RBV")]

    # The array_size0/1/2 signals are used when receiving an NDArray
    # (i.e. in a plugin)
    array_size0: A[SignalR[int], PvSuffix("ArraySize0_RBV")]
    array_size1: A[SignalR[int], PvSuffix("ArraySize1_RBV")]
    array_size2: A[SignalR[int], PvSuffix("ArraySize2_RBV")]

    color_mode: A[SignalR[ADBaseColorMode], PvSuffix("ColorMode_RBV")]
    data_type: A[SignalR[ADBaseDataType], PvSuffix("DataType_RBV")]
    array_counter: A[SignalRW[int], PvSuffix.rbv("ArrayCounter")]

    # Version information may be useful to include in metadata
    ad_core_version: A[SignalR[str], PvSuffix("ADCoreVersion_RBV")]
    driver_version: A[SignalR[str], PvSuffix("DriverVersion_RBV")]


# Classes for drivers


class ADImageMode(SubsetEnum):
    SINGLE = "Single"
    MULTIPLE = "Multiple"
    CONTINUOUS = "Continuous"


class ADState(StrictEnum):
    """Default set of states of an AreaDetector driver.

    See definition in ADApp/ADSrc/ADDriver.h in https://github.com/areaDetector/ADCore.
    """

    IDLE = "Idle"
    ACQUIRE = "Acquire"
    READOUT = "Readout"
    CORRECT = "Correct"
    SAVING = "Saving"
    ABORTING = "Aborting"
    ERROR = "Error"
    WAITING = "Waiting"
    INITIALIZING = "Initializing"
    DISCONNECTED = "Disconnected"
    ABORTED = "Aborted"


#: TypeVar for any `ADBaseIO` subclass
ADBaseIOT = TypeVar("ADBaseIOT", bound="ADBaseIO")


class ADBaseIO(StandardReadable, NDArrayBaseIO):
    """Base class from which areaDetector drivers are derived.

    This mirrors the interface provided by ADCore/db/ADBase.template.
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/ADDriver.html

    The interface deviates from the most up to date version of ADBase as it contains
    some signals that are now in NDArrayBaseIO. This is to maintain backwards
    compatibility with changes made in https://github.com/areaDetector/ADCore/blob/master/RELEASE.md#asynndarraydriver-addriver
    """

    acquire_time: A[SignalRW[float], PvSuffix.rbv("AcquireTime"), Format.CONFIG_SIGNAL]
    acquire_period: A[
        SignalRW[float], PvSuffix.rbv("AcquirePeriod"), Format.CONFIG_SIGNAL
    ]
    num_images: A[SignalRW[int], PvSuffix.rbv("NumImages"), Format.CONFIG_SIGNAL]
    image_mode: A[
        SignalRW[ADImageMode], PvSuffix.rbv("ImageMode"), Format.CONFIG_SIGNAL
    ]
    detector_state: A[SignalR[ADState], PvSuffix("DetectorState_RBV")]

    # The following signals have been moved from NDArrayBaseIO for backwards
    # compatibility
    acquire: A[SignalRW[bool], PvSuffix.rbv("Acquire"), EpicsOptions(wait=non_zero)]

    # There is no _RBV for this one
    wait_for_plugins: A[SignalRW[bool], PvSuffix("WaitForPlugins")]

    manufacturer: A[SignalR[str], PvSuffix("Manufacturer_RBV"), Format.CONFIG_SIGNAL]
    model: A[SignalR[str], PvSuffix("Model_RBV"), Format.CONFIG_SIGNAL]
    serial_number: A[SignalR[str], PvSuffix("SerialNumber_RBV"), Format.CONFIG_SIGNAL]
    sdk_version: A[SignalR[str], PvSuffix("SDKVersion_RBV"), Format.CONFIG_SIGNAL]
    firmware_version: A[
        SignalR[str], PvSuffix("FirmwareVersion_RBV"), Format.CONFIG_SIGNAL
    ]


# Classes for plugins
#: TypeVar for any `NDPluginBaseIO` subclass
NDPluginBaseIOT = TypeVar("NDPluginBaseIOT", bound="NDPluginBaseIO")


class NDPluginBaseIO(NDArrayBaseIO):
    """Base class from which plugins are derived.

    This mirrors the interface provided by ADCore/db/NDPluginBase.template.
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/NDPluginDriver.html
    """

    nd_array_port: A[SignalRW[str], PvSuffix.rbv("NDArrayPort")]
    enable_callbacks: A[SignalRW[EnableDisable], PvSuffix.rbv("EnableCallbacks")]
    nd_array_address: A[SignalRW[int], PvSuffix.rbv("NDArrayAddress")]
    queue_size: A[SignalRW[int], PvSuffix.rbv("QueueSize")]

    @cached_property
    def _snapshot(self) -> dict[SignalRW[Any], Any]:
        return {}

    def get_switches(self) -> Sequence[SignalRW[Any]]:
        """The PVs that switch what this plugin produces on and off."""
        return [self.enable_callbacks]

    def get_snapshot_signals(self) -> Sequence[SignalRW[Any]]:
        """The PVs `snapshot()` records: the switches and the NDArrayPort.

        Only this plugin's own NDArrayPort is recorded. The chain of plugins that
        feeds it, which may cross IOCs, is not walked: the chain as wired at
        `stage()` is the user's responsibility.
        """
        return [*self.get_switches(), self.nd_array_port]

    async def snapshot(self) -> None:
        """Record the value of each of `get_snapshot_signals()`.

        `AreaDetector` calls this on `stage()`, once. The values stay until
        `forget_snapshot()`.
        """
        signals = self.get_snapshot_signals()
        values = await asyncio.gather(*(signal.get_value() for signal in signals))
        self._snapshot.clear()
        self._snapshot.update(zip(signals, values, strict=True))

    async def get_changed_signals(self) -> list[SignalRW[Any]]:
        """The recorded signals whose PV has a different value now.

        `AreaDetector` calls this on each `prepare()`.
        """
        recorded = list(self._snapshot.items())
        current = await asyncio.gather(*(signal.get_value() for signal, _ in recorded))
        return [
            signal
            for (signal, value), now in zip(recorded, current, strict=True)
            if now != value
        ]

    def forget_snapshot(self) -> None:
        """Drop the recorded values. `AreaDetector` calls this on `unstage()`."""
        self._snapshot.clear()

    def switch_is_on(self, switch: SignalRW[Any]) -> bool:
        """Whether `switch` was on when recorded, True if it was not recorded."""
        value = self._snapshot.get(switch, True)
        return value is True or value is EnableDisable.ENABLE

    def get_switched_off_readables(self) -> set[SignalR]:
        """The readable signals the recorded switches turn off."""
        return set()


async def plugin_is_enabled(plugin: NDPluginBaseIO) -> bool:
    """Whether a plugin is switched on, so is receiving NDArrays.

    The nearest thing to "is this plugin in the chain": a plugin with callbacks
    disabled certainly is not, though an enabled one may still have its
    `nd_array_port` pointed somewhere else. Parsing the whole plugin graph was
    rejected in ADR 0025 as too large.
    """
    return await plugin.enable_callbacks.get_value() is EnableDisable.ENABLE


class NDROIIO(StandardReadable, NDPluginBaseIO):
    """Plugin for taking a region of an NDArray.

    This mirrors the interface provided by ADCore/db/NDROI.template.
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/NDPluginROI.html
    """

    min_x: A[SignalRW[int], PvSuffix.rbv("MinX"), Format.CONFIG_SIGNAL]
    min_y: A[SignalRW[int], PvSuffix.rbv("MinY"), Format.CONFIG_SIGNAL]
    size_x: A[SignalRW[int], PvSuffix.rbv("SizeX"), Format.CONFIG_SIGNAL]
    size_y: A[SignalRW[int], PvSuffix.rbv("SizeY"), Format.CONFIG_SIGNAL]
    size_z: A[SignalRW[int], PvSuffix.rbv("SizeZ")]

    bin_x: A[SignalRW[int], PvSuffix.rbv("BinX")]
    bin_y: A[SignalRW[int], PvSuffix.rbv("BinY")]
    bin_z: A[SignalRW[int], PvSuffix.rbv("BinZ")]


class NDStatsTSAcquireMode(StrictEnum):
    """Whether an NDPluginStats time series stops when full or wraps around.

    Mirrors the ``TSAcquireMode`` record in
    ADCore/db/NDPluginTimeSeries.template.
    """

    FIXED_LENGTH = "Fixed length"
    CIRCULAR_BUFFER = "Circ. buffer"


# The formats that put a signal in read(), rather than in configuration
_READ_FORMATS = (
    Format.HINTED_SIGNAL,
    Format.UNCACHED_SIGNAL,
    Format.HINTED_UNCACHED_SIGNAL,
)


@dataclass(frozen=True)
class NDStatistic:
    """One statistic of an `NDStatsIO`, in each form the plugin produces it.

    :param scalar: the value for the last frame
    :param param: the asyn parameter, the source of an NDAttribute
    :param time_series: the value for each frame of the time series
    :param compute: the switch for the group of statistics this belongs to
    """

    scalar: SignalR[float]
    param: str
    time_series: SignalR[Array1D[np.float64]]
    compute: SignalRW[bool]


class NDStatsIO(StandardReadable, NDPluginBaseIO):
    """Plugin for computing statistics from an image or ROI within an image.

    This mirrors the interface provided by ADCore/db/NDStats.template.
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/NDPluginStats.html
    """

    # Basic statistics
    compute_statistics: A[
        SignalRW[bool], PvSuffix.rbv("ComputeStatistics"), Format.CONFIG_SIGNAL
    ]
    bgd_width: A[SignalRW[int], PvSuffix.rbv("BgdWidth"), Format.CONFIG_SIGNAL]
    min_value: A[
        SignalR[float], PvSuffix("MinValue_RBV"), Format.HINTED_UNCACHED_SIGNAL
    ]
    min_x: A[SignalR[float], PvSuffix("MinX_RBV"), Format.UNCACHED_SIGNAL]
    min_y: A[SignalR[float], PvSuffix("MinY_RBV"), Format.UNCACHED_SIGNAL]
    max_value: A[
        SignalR[float], PvSuffix("MaxValue_RBV"), Format.HINTED_UNCACHED_SIGNAL
    ]
    max_x: A[SignalR[float], PvSuffix("MaxX_RBV"), Format.UNCACHED_SIGNAL]
    max_y: A[SignalR[float], PvSuffix("MaxY_RBV"), Format.UNCACHED_SIGNAL]
    mean_value: A[
        SignalR[float], PvSuffix("MeanValue_RBV"), Format.HINTED_UNCACHED_SIGNAL
    ]
    sigma: A[SignalR[float], PvSuffix("Sigma_RBV")]
    total: A[SignalR[float], PvSuffix("Total_RBV"), Format.HINTED_UNCACHED_SIGNAL]
    net: A[SignalR[float], PvSuffix("Net_RBV")]
    # Time series. NDStats delegates its time series to an embedded
    # NDPluginTimeSeries instance, so the controls live under an inner "TS:"
    # prefix while the per-statistic arrays are named by NDStats itself at the
    # plugin's own prefix. ts_num_points sizes the buffer, ts_acquire erases and
    # starts it (writing 1 clears the arrays and resets ts_current_point to 0),
    # ts_current_point reports progress, each ts_<statistic> holds the series of
    # that statistic and ts_timestamp holds the per-point acquisition times.
    # These are not registered as readables: the scalars serve read(), and the
    # arrays are consumed by StatsTimeSeriesDataLogic as event pages.
    ts_acquire: A[SignalRW[bool], PvSuffix("TS:TSAcquire"), EpicsOptions(wait=non_zero)]
    ts_num_points: A[SignalRW[int], PvSuffix("TS:TSNumPoints")]
    ts_current_point: A[SignalR[int], PvSuffix("TS:TSCurrentPoint")]
    ts_acquiring: A[SignalR[bool], PvSuffix("TS:TSAcquiring")]
    ts_acquire_mode: A[SignalRW[NDStatsTSAcquireMode], PvSuffix.rbv("TS:TSAcquireMode")]
    ts_min_value: A[SignalR[Array1D[np.float64]], PvSuffix("TSMinValue")]
    ts_min_x: A[SignalR[Array1D[np.float64]], PvSuffix("TSMinX")]
    ts_min_y: A[SignalR[Array1D[np.float64]], PvSuffix("TSMinY")]
    ts_max_value: A[SignalR[Array1D[np.float64]], PvSuffix("TSMaxValue")]
    ts_max_x: A[SignalR[Array1D[np.float64]], PvSuffix("TSMaxX")]
    ts_max_y: A[SignalR[Array1D[np.float64]], PvSuffix("TSMaxY")]
    ts_mean_value: A[SignalR[Array1D[np.float64]], PvSuffix("TSMeanValue")]
    ts_sigma: A[SignalR[Array1D[np.float64]], PvSuffix("TSSigma")]
    ts_total: A[SignalR[Array1D[np.float64]], PvSuffix("TSTotal")]
    ts_net: A[SignalR[Array1D[np.float64]], PvSuffix("TSNet")]
    ts_centroid_x: A[SignalR[Array1D[np.float64]], PvSuffix("TSCentroidX")]
    ts_centroid_y: A[SignalR[Array1D[np.float64]], PvSuffix("TSCentroidY")]
    ts_sigma_x: A[SignalR[Array1D[np.float64]], PvSuffix("TSSigmaX")]
    ts_sigma_y: A[SignalR[Array1D[np.float64]], PvSuffix("TSSigmaY")]
    ts_sigma_xy: A[SignalR[Array1D[np.float64]], PvSuffix("TSSigmaXY")]
    ts_timestamp: A[SignalR[Array1D[np.float64]], PvSuffix("TSTimestamp")]
    # Centroid statistics
    compute_centroid: A[
        SignalRW[bool], PvSuffix.rbv("ComputeCentroid"), Format.CONFIG_SIGNAL
    ]
    centroid_threshold: A[
        SignalRW[float], PvSuffix.rbv("CentroidThreshold"), Format.CONFIG_SIGNAL
    ]
    centroid_x: A[SignalR[float], PvSuffix("CentroidX_RBV")]
    centroid_y: A[SignalR[float], PvSuffix("CentroidY_RBV")]
    sigma_x: A[SignalR[float], PvSuffix("SigmaX_RBV")]
    sigma_y: A[SignalR[float], PvSuffix("SigmaY_RBV")]
    sigma_xy: A[SignalR[float], PvSuffix("SigmaXY_RBV")]
    # X and Y Profiles
    compute_profiles: A[
        SignalRW[bool], PvSuffix.rbv("ComputeProfiles"), Format.CONFIG_SIGNAL
    ]
    profile_size_x: A[SignalR[int], PvSuffix("ProfileSizeX_RBV")]
    profile_size_y: A[SignalR[int], PvSuffix("ProfileSizeY_RBV")]
    cursor_x: A[SignalRW[int], PvSuffix.rbv("CursorX"), Format.CONFIG_SIGNAL]
    cursor_y: A[SignalRW[int], PvSuffix.rbv("CursorY"), Format.CONFIG_SIGNAL]
    # Array Histogram
    compute_histogram: A[
        SignalRW[bool], PvSuffix.rbv("ComputeHistogram"), Format.CONFIG_SIGNAL
    ]
    hist_size: A[SignalRW[int], PvSuffix.rbv("HistSize"), Format.CONFIG_SIGNAL]
    hist_min: A[SignalRW[float], PvSuffix.rbv("HistMin"), Format.CONFIG_SIGNAL]
    hist_max: A[SignalRW[float], PvSuffix.rbv("HistMax"), Format.CONFIG_SIGNAL]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._suppressed_datakeys_funcs += (self._get_switched_off_datakeys,)

    @cached_property
    def statistics(self) -> tuple[NDStatistic, ...]:
        """Each statistic, as ADCore's NDStats.template defines it."""
        basic = self.compute_statistics
        centroid = self.compute_centroid
        return (
            NDStatistic(self.min_value, "MIN_VALUE", self.ts_min_value, basic),
            NDStatistic(self.min_x, "MIN_X", self.ts_min_x, basic),
            NDStatistic(self.min_y, "MIN_Y", self.ts_min_y, basic),
            NDStatistic(self.max_value, "MAX_VALUE", self.ts_max_value, basic),
            NDStatistic(self.max_x, "MAX_X", self.ts_max_x, basic),
            NDStatistic(self.max_y, "MAX_Y", self.ts_max_y, basic),
            NDStatistic(self.mean_value, "MEAN_VALUE", self.ts_mean_value, basic),
            NDStatistic(self.sigma, "SIGMA_VALUE", self.ts_sigma, basic),
            NDStatistic(self.total, "TOTAL", self.ts_total, basic),
            NDStatistic(self.net, "NET", self.ts_net, basic),
            NDStatistic(
                self.centroid_x, "CENTROIDX_VALUE", self.ts_centroid_x, centroid
            ),
            NDStatistic(
                self.centroid_y, "CENTROIDY_VALUE", self.ts_centroid_y, centroid
            ),
            NDStatistic(self.sigma_x, "SIGMAX_VALUE", self.ts_sigma_x, centroid),
            NDStatistic(self.sigma_y, "SIGMAY_VALUE", self.ts_sigma_y, centroid),
            NDStatistic(self.sigma_xy, "SIGMAXY_VALUE", self.ts_sigma_xy, centroid),
        )

    def get_switches(self) -> Sequence[SignalRW[Any]]:
        return [*super().get_switches(), self.compute_statistics, self.compute_centroid]

    def _is_computed(self, statistic: NDStatistic) -> bool:
        return self.switch_is_on(self.enable_callbacks) and self.switch_is_on(
            statistic.compute
        )

    def get_switched_off_readables(self) -> set[SignalR]:
        return {stat.scalar for stat in self.statistics if not self._is_computed(stat)}

    def _get_switched_off_datakeys(self) -> dict[str, None]:
        return {signal.name: None for signal in self.get_switched_off_readables()}

    def get_published_statistics(self) -> list[NDStatistic]:
        """The statistics that are computed and have a read format on this plugin."""
        formats = self.get_readable_formats()
        return [
            stat
            for stat in self.statistics
            if self._is_computed(stat) and formats.get(stat.scalar) in _READ_FORMATS
        ]

    def get_ndattributes(self) -> list[NDAttributeParam]:
        """Each published statistic, named after its scalar signal."""
        return [
            NDAttributeParam(
                name=stat.scalar.name,
                param=stat.param,
                datatype=NDAttributeDataType.DOUBLE,
            )
            for stat in self.get_published_statistics()
        ]


class NDROIStatNIO(StandardReadable, EpicsDevice):
    """Defines the parameters for a single ROI used for statistics calculation.

    This mirrors the interface provided by ADCore/db/NDROIStatN.template.
    See definition in ADApp/pluginSrc/NDPluginROIStat.h in https://github.com/areaDetector/ADCore.
    """

    name_: A[SignalRW[str], PvSuffix("Name")]
    use: A[SignalRW[bool], PvSuffix.rbv("Use"), Format.CONFIG_SIGNAL]
    min_x: A[SignalRW[int], PvSuffix.rbv("MinX"), Format.CONFIG_SIGNAL]
    min_y: A[SignalRW[int], PvSuffix.rbv("MinY"), Format.CONFIG_SIGNAL]
    size_x: A[SignalRW[int], PvSuffix.rbv("SizeX"), Format.CONFIG_SIGNAL]
    size_y: A[SignalRW[int], PvSuffix.rbv("SizeY"), Format.CONFIG_SIGNAL]
    min_value: A[
        SignalR[float], PvSuffix("MinValue_RBV"), Format.HINTED_UNCACHED_SIGNAL
    ]
    max_value: A[
        SignalR[float], PvSuffix("MaxValue_RBV"), Format.HINTED_UNCACHED_SIGNAL
    ]
    mean_value: A[
        SignalR[float], PvSuffix("MeanValue_RBV"), Format.HINTED_UNCACHED_SIGNAL
    ]
    total: A[SignalR[float], PvSuffix("Total_RBV"), Format.HINTED_UNCACHED_SIGNAL]

    def get_ndattributes(self, addr: int) -> list[NDAttributeParam]:
        """Each scalar with a read format, named after it, at asyn address `addr`."""
        formats = self.get_readable_formats()
        return [
            NDAttributeParam(
                name=scalar.name,
                param=param,
                datatype=NDAttributeDataType.DOUBLE,
                addr=addr,
            )
            for scalar, param in self.params.items()
            if formats.get(scalar) in _READ_FORMATS
        ]

    @cached_property
    def params(self) -> dict[SignalR[float], str]:
        """The asyn parameter each scalar signal of this ROI is read from."""
        return {
            self.min_value: "ROISTAT_MIN_VALUE",
            self.max_value: "ROISTAT_MAX_VALUE",
            self.mean_value: "ROISTAT_MEAN_VALUE",
            self.total: "ROISTAT_TOTAL",
        }


class NDROIStatIO(StandardReadable, NDPluginBaseIO):
    """Plugin for calculating basic statistics for multiple ROIs.

    Each ROI is implemented as an instance of NDROIStatNIO,
    and the collection of ROIs is held as a DeviceVector. Each channel is
    registered as a [](#StandardReadableFormat.CHILD) readable, so it reads the
    scalars of its own formats.

    When an `AreaDetector` stages, each channel is renamed to
    `<plugin name>-<Name PV>`, with characters other than letters, digits and
    underscores in the PV replaced by underscores, or the channel number if the
    PV is empty. A `Name` PV that changes after that makes `prepare()` raise.

    This mirrors the interface provided by ADCore/db/NDROIStat.template.
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/NDPluginROIStat.html
    """

    def __init__(self, prefix, num_channels=8, with_pvi=False, name=""):
        with self.add_children_as_readables():
            self.channels = DeviceVector(
                {i: NDROIStatNIO(f"{prefix}{i}:") for i in range(1, num_channels + 1)}
            )
        super().__init__(prefix, with_pvi, name)

    def get_switches(self) -> Sequence[SignalRW[Any]]:
        channel_switches = [channel.use for channel in self.channels.values()]
        return [*super().get_switches(), *channel_switches]

    def get_snapshot_signals(self) -> Sequence[SignalRW[Any]]:
        channel_names = [channel.name_ for channel in self.channels.values()]
        return [*super().get_snapshot_signals(), *channel_names]

    async def snapshot(self) -> None:
        """Record the signals, and name each channel after its Name PV."""
        await super().snapshot()
        self._name_channels()

    def _name_channels(self) -> None:
        """Name each channel after its recorded Name PV, raising on a clash."""
        names = {
            number: self._channel_name(number, channel)
            for number, channel in self.channels.items()
        }
        counts = Counter(names.values())
        clashes = sorted(name for name, count in counts.items() if count > 1)
        if clashes:
            raise ValueError(
                f"{self.name}: more than one channel is named {clashes}, "
                "so change the Name PV of all but one"
            )
        if self.name:
            for number, name in names.items():
                self.channels[number].set_name(
                    f"{self.name}{self._child_name_separator}{name}",
                    child_name_separator=self._child_name_separator,
                )

    def _channel_name(self, number: int, channel: NDROIStatNIO) -> str:
        pv_name = self._snapshot[channel.name_]
        return re.sub(r"[^A-Za-z0-9_]", "_", pv_name) if pv_name else str(number)

    def _is_computed(self, channel: NDROIStatNIO) -> bool:
        return self.switch_is_on(self.enable_callbacks) and self.switch_is_on(
            channel.use
        )

    def get_switched_off_readables(self) -> set[SignalR]:
        return {
            scalar
            for channel in self.channels.values()
            if not self._is_computed(channel)
            for scalar in channel.params
        }

    def get_ndattributes(self) -> list[NDAttributeParam]:
        """The NDAttributes of each channel that is computed, by asyn address."""
        return [
            ndattribute
            for number, channel in self.channels.items()
            if self._is_computed(channel)
            for ndattribute in channel.get_ndattributes(addr=number - 1)
        ]


class NDCBFlushOnSoftTrgMode(StrictEnum):
    ON_NEW_IMAGE = "OnNewImage"
    IMMEDIATELY = "Immediately"


class NDCircularBuffIO(NDPluginBaseIO):
    """Plugin for flow control of arrays based on NDAttributes.

    This mirrors the interface provided by ADCore/db/NDCircularBuff.template.
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/NDPluginCircularBuff.html
    """

    pre_count: A[SignalRW[int], PvSuffix.rbv("PreCount")]
    post_count: A[SignalRW[int], PvSuffix.rbv("PostCount")]
    preset_trigger_count: A[SignalRW[int], PvSuffix.rbv("PresetTriggerCount")]
    # Disable put completion for trigger, since the busy record reset is broken
    # upstream. See https://github.com/areaDetector/ADCore/issues/604
    trigger_: A[SignalRW[bool], PvSuffix.rbv("Trigger"), EpicsOptions(wait=False)]
    capture: A[SignalRW[bool], PvSuffix.rbv("Capture"), EpicsOptions(wait=non_zero)]
    flush_on_soft_trg: A[
        SignalRW[NDCBFlushOnSoftTrgMode], PvSuffix.rbv("FlushOnSoftTrg")
    ]


class NDProcessFilterCallbacks(StrictEnum):
    EVERY_ARRAY = "Every array"
    ARRAY_N_ONLY = "Array N only"


class NDProcessFilterType(StrictEnum):
    RECURSIVE_AVG = "RecursiveAve"
    AVERAGE = "Average"
    SUM = "Sum"
    DIFFERENCE = "Difference"
    RECURSIVE_AVG_DIFFERENCE = "RecursiveAveDiff"
    COPY_TO_FILTER = "CopyToFilter"


class NDProcessIO(NDPluginBaseIO):
    """Plugin for processing NDArray data.

    This mirrors the interface provided by ADCore/db/NDProcess.template.
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/NDPluginProcess.html
    """

    data_type_out: A[SignalRW[ADBaseDataType], PvSuffix.rbv("DataTypeOut")]
    save_background: A[SignalRW[bool], PvSuffix.rbv("SaveBackground")]
    enable_background: A[SignalRW[bool], PvSuffix.rbv("EnableBackground")]
    valid_background: A[SignalR[bool], PvSuffix("ValidBackground_RBV")]
    save_flat_field: A[SignalRW[bool], PvSuffix.rbv("SaveFlatField")]
    enable_flat_field: A[SignalRW[bool], PvSuffix.rbv("EnableFlatField")]
    valid_flat_field: A[SignalR[bool], PvSuffix("ValidFlatField_RBV")]
    scale_flat_field: A[SignalRW[float], PvSuffix.rbv("ScaleFlatField")]
    enable_offset_scale: A[SignalRW[bool], PvSuffix.rbv("EnableOffsetScale")]
    auto_offset_scale: A[SignalRW[bool], PvSuffix("AutoOffsetScale")]
    offset: A[SignalRW[float], PvSuffix.rbv("Offset")]
    scale: A[SignalRW[float], PvSuffix.rbv("Scale")]
    enable_low_clip: A[SignalRW[bool], PvSuffix.rbv("EnableLowClip")]
    enable_high_clip: A[SignalRW[bool], PvSuffix.rbv("EnableHighClip")]

    # The below signals were renamed in ADCore 3-13. Since
    # This is a relatively recent release, we are omitting them from the
    # IO class for now.
    # low_clip_thresh: A[SignalRW[float], PvSuffix.rbv("LowClipThresh")]
    # low_clip_value: A[SignalRW[float], PvSuffix.rbv("LowClipValue")]
    # high_clip_thresh: A[SignalRW[float], PvSuffix.rbv("HighClipThresh")]
    # high_clip_value: A[SignalRW[float], PvSuffix.rbv("HighClipValue")]

    enable_filter: A[SignalRW[bool], PvSuffix.rbv("EnableFilter")]
    reset_filter: A[TriggerableCommand, PvSuffix.rbv("ResetFilter")]
    auto_reset_filter: A[SignalRW[bool], PvSuffix.rbv("AutoResetFilter")]
    filter_type: A[SignalRW[NDProcessFilterType], PvSuffix("FilterType")]
    filter_callbacks: A[
        SignalRW[NDProcessFilterCallbacks], PvSuffix.rbv("FilterCallbacks")
    ]
    num_filter: A[SignalRW[int], PvSuffix.rbv("NumFilter")]
    num_filter_recip: A[SignalR[float], PvSuffix("NumFilterRecip")]
    num_filtered: A[SignalR[int], PvSuffix("NumFiltered_RBV")]
    o_offset: A[SignalRW[float], PvSuffix.rbv("OOffset")]
    o_scale: A[SignalRW[float], PvSuffix.rbv("OScale")]
    f_offset: A[SignalRW[float], PvSuffix.rbv("FOffset")]
    f_scale: A[SignalRW[float], PvSuffix.rbv("FScale")]
    r_offset: A[SignalRW[float], PvSuffix.rbv("ROffset")]

    def __init__(self, prefix: str, name: str = ""):
        super().__init__(prefix, name=name)


# Codec/compression classes


class ADBloscCompressor(StrictEnum):
    BLOSCLZ = "BloscLZ"
    LZ4 = "LZ4"
    LZ4HC = "LZ4HC"
    SNAPPY = "SNAPPY"
    ZLIB = "ZLIB"
    ZSTD = "ZSTD"


class ADBloscShuffle(StrictEnum):
    NONE = "None"
    BYTE = "Byte"
    BIT = "Bit"


class ADCompressMode(StrictEnum):
    COMPRESS = "Compress"
    DECOMPRESS = "Decompress"


# Compressor is superset enum, because LZ4HDF5 and ZLIB were added only in ADCore R3-15
class ADCompressor(SupersetEnum):
    NONE = "None"
    JPEG = "JPEG"
    ZLIB = "ZLIB"
    BLOSC = "Blosc"
    LZ4 = "LZ4"
    LZ4HDF5 = "LZ4HDF5"
    BSLZ4 = "BSLZ4"


class NDCodecStatus(StrictEnum):
    SUCCESS = "Success"
    WARNING = "Warning"
    ERROR = "Error"


class NDCodecIO(NDPluginBaseIO):
    """Plugin for compressing and decompressing NDArray data.

    This mirrors the interface provided by ADCore/db/NDCodec.template.
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/NDPluginCodec.html
    """

    compressor: A[SignalRW[ADCompressor], PvSuffix.rbv("Compressor")]
    mode: A[SignalRW[ADCompressMode], PvSuffix.rbv("Mode")]
    comp_factor: A[SignalR[float], PvSuffix("CompFactor_RBV")]
    jpeg_quality: A[SignalRW[int], PvSuffix.rbv("JPEGQuality")]
    zlib_c_level: A[SignalRW[int], PvSuffix.rbv("ZlibCLevel")]
    blosc_compressor: A[SignalRW[ADBloscCompressor], PvSuffix.rbv("BloscCompressor")]
    blosc_c_level: A[SignalRW[int], PvSuffix.rbv("BloscCLevel")]
    blosc_num_threads: A[SignalRW[int], PvSuffix.rbv("BloscNumThreads")]
    lz4_hdf5_block_size: A[SignalRW[int], PvSuffix.rbv("LZ4HDF5BlockSize")]
    codec_status: A[SignalR[NDCodecStatus], PvSuffix("CodecStatus")]
    codec_error: A[SignalR[str], PvSuffix("CodecError")]


# Classes for filewriters


class ADFileWriteMode(StrictEnum):
    SINGLE = "Single"
    CAPTURE = "Capture"
    STREAM = "Stream"


class NDFileIO(NDArrayBaseIO):
    """Base class from which file writing drivers are derived.

    This mirrors the interface provided by ADCore/ADApp/Db/NDFile.template.
    It does not include any plugin-related fields, for that see NDFilePluginIO.
    """

    file_path: A[SignalRW[str], PvSuffix.rbv("FilePath")]
    file_name: A[SignalRW[str], PvSuffix.rbv("FileName")]
    file_path_exists: A[SignalR[bool], PvSuffix("FilePathExists_RBV")]
    file_template: A[SignalRW[str], PvSuffix.rbv("FileTemplate")]
    full_file_name: A[SignalR[str], PvSuffix("FullFileName_RBV")]
    file_number: A[SignalRW[int], PvSuffix("FileNumber")]
    auto_increment: A[SignalRW[bool], PvSuffix("AutoIncrement")]
    file_write_mode: A[SignalRW[ADFileWriteMode], PvSuffix.rbv("FileWriteMode")]
    num_capture: A[SignalRW[int], PvSuffix.rbv("NumCapture")]
    num_captured: A[SignalR[int], PvSuffix("NumCaptured_RBV")]
    capture: A[SignalRW[bool], PvSuffix.rbv("Capture"), EpicsOptions(wait=non_zero)]
    array_size0: A[SignalR[int], PvSuffix("ArraySize0")]
    array_size1: A[SignalR[int], PvSuffix("ArraySize1")]
    create_directory: A[SignalRW[int], PvSuffix("CreateDirectory")]
    lazy_open: A[SignalRW[bool], PvSuffix.rbv("LazyOpen")]


NDPluginFileIOT = TypeVar("NDPluginFileIOT", bound="NDPluginFileIO")


class NDPluginFileIO(NDPluginBaseIO, NDFileIO):
    """Base class from which file plugins are derived.

    This mirrors the interface provided by ADCore/db/NDFile.template when added to
    NDPluginBase.template
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/NDPluginFile.html
    """

    ...


class NDFileHDF5Compression(StrictEnum):
    NONE = "None"
    NBIT = "N-bit"
    SZIP = "szip"
    ZLIB = "zlib"
    BLOSC = "Blosc"
    BSLZ4 = "BSLZ4"
    LZ4 = "LZ4"
    JPEG = "JPEG"


class NDFileHDF5IO(NDPluginFileIO):
    """Plugin for storing data in HDF5 file format.

    This mirrors the interface provided by ADCore/db/NDFileHDF5.template.
    See HTML docs at https://areadetector.github.io/areaDetector/ADCore/NDFileHDF5.html
    """

    position_mode: A[SignalRW[bool], PvSuffix.rbv("PositionMode")]
    compression: A[SignalRW[NDFileHDF5Compression], PvSuffix.rbv("Compression")]
    num_extra_dims: A[SignalRW[int], PvSuffix.rbv("NumExtraDims")]
    swmr_mode: A[SignalRW[bool], PvSuffix.rbv("SWMRMode")]
    flush_now: A[SignalRW[bool], PvSuffix("FlushNow")]
    num_frames_flush: A[SignalRW[int], PvSuffix.rbv("NumFramesFlush")]
    xml_file_name: A[SignalRW[str], PvSuffix.rbv("XMLFileName")]
    num_frames_chunks: A[SignalRW[int], PvSuffix.rbv("NumFramesChunks")]
    chunk_size_auto: A[SignalRW[bool], PvSuffix.rbv("ChunkSizeAuto")]

    # Compression options
    s_zip_num_pixels: A[SignalRW[int], PvSuffix.rbv("SZipNumPixels")]
    z_level: A[SignalRW[int], PvSuffix.rbv("ZLevel")]
    blosc_shuffle: A[SignalRW[ADBloscShuffle], PvSuffix.rbv("BloscShuffle")]
    blosc_compressor: A[SignalRW[ADBloscCompressor], PvSuffix.rbv("BloscCompressor")]
    blosc_level: A[SignalRW[int], PvSuffix.rbv("BloscLevel")]
    jpeg_quality: A[SignalRW[int], PvSuffix.rbv("JPEGQuality")]
