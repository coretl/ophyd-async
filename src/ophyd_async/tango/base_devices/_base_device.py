from __future__ import annotations

from typing import (
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    Device,
    DeviceVector,
    Signal,
)
from ophyd_async.tango.signal import (
    make_backend,
    tango_signal_auto,
)
from tango import DeviceProxy as SyncDeviceProxy
from tango.asyncio import DeviceProxy as AsyncDeviceProxy

T = TypeVar("T")


class TangoDevice(Device):
    """
    General class for TangoDevices. Extends Device to provide attributes for Tango
    devices.

    Parameters
    ----------
    trl: str
        Tango resource locator, typically of the device server.
    device_proxy: Optional[Union[AsyncDeviceProxy, SyncDeviceProxy]]
        Asynchronous or synchronous DeviceProxy object for the device. If not provided,
        an asynchronous DeviceProxy object will be created using the trl and awaited
        when the device is connected.
    """

    trl: str = ""
    proxy: Optional[Union[AsyncDeviceProxy, SyncDeviceProxy]] = None
    _polling: Tuple[bool, float, float, float] = (False, 0.1, None, 0.1)
    _signal_polling: Dict[str, Tuple[bool, float, float, float]] = {}
    _poll_only_annotated_signals: bool = True

    def __init__(
        self,
        trl: Optional[str] = None,
        device_proxy: Optional[Union[AsyncDeviceProxy, SyncDeviceProxy]] = None,
        name: str = "",
    ) -> None:
        if not trl and not device_proxy:
            raise ValueError("Either 'trl' or 'device_proxy' must be provided.")

        self.trl = trl if trl else ""
        self.proxy = device_proxy
        tango_create_children_from_annotations(self)
        super().__init__(name=name)

    async def connect(
        self,
        mock: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        force_reconnect: bool = False,
    ):
        async def closure():
            try:
                if self.proxy is None:
                    self.proxy = await AsyncDeviceProxy(self.trl)
            except Exception as e:
                raise RuntimeError("Could not connect to device proxy") from e
            return self

        if self.trl in ["", None]:
            self.trl = self.proxy.name()

        await closure()
        self.register_signals()
        await fill_proxy_entries(self)

        # set_name should be called again to propagate the new signal names
        self.set_name(self.name)

        # Set the polling configuration
        if self._polling[0]:
            for child in self.children():
                if issubclass(type(child[1]), Signal):
                    child[1]._backend.set_polling(*self._polling)  # noqa: SLF001
                    child[1]._backend.allow_events(False)  # noqa: SLF001
        if self._signal_polling:
            for signal_name, polling in self._signal_polling.items():
                if hasattr(self, signal_name):
                    attr = getattr(self, signal_name)
                    attr._backend.set_polling(*polling)  # noqa: SLF001
                    attr._backend.allow_events(False)  # noqa: SLF001

        await super().connect(mock=mock, timeout=timeout)

    # Users can override this method to register new signals
    def register_signals(self):
        pass


def tango_polling(
    polling: Optional[
        Union[Tuple[float, float, float], Dict[str, Tuple[float, float, float]]]
    ] = None,
    signal_polling: Optional[Dict[str, Tuple[float, float, float]]] = None,
):
    """
    Class decorator to configure polling for Tango devices.

    This decorator allows for the configuration of both device-level and signal-level
    polling for Tango devices. Polling is useful for device servers that do not support
    event-driven updates.

    Parameters
    ----------
    polling : Optional[Union[Tuple[float, float, float],
        Dict[str, Tuple[float, float, float]]]], optional
        Device-level polling configuration as a tuple of three floats representing the
        polling interval, polling timeout, and polling delay. Alternatively,
        a dictionary can be provided to specify signal-level polling configurations
        directly.
    signal_polling : Optional[Dict[str, Tuple[float, float, float]]], optional
        Signal-level polling configuration as a dictionary where keys are signal names
        and values are tuples of three floats representing the polling interval, polling
        timeout, and polling delay.

    Returns
    -------
    Callable
        A class decorator that sets the `_polling` and `_signal_polling` attributes on
        the decorated class.

    Example
    -------
    Device-level and signal-level polling:
    @tango_polling(
        polling=(0.5, 1.0, 0.1),
        signal_polling={
            'signal1': (0.5, 1.0, 0.1),
            'signal2': (1.0, 2.0, 0.2),
        }
    )
    class MyTangoDevice(TangoDevice):
        signal1: Signal
        signal2: Signal
    """
    if isinstance(polling, dict):
        signal_polling = polling
        polling = None

    def decorator(cls):
        if polling is not None:
            cls._polling = (True, *polling)
        if signal_polling is not None:
            cls._signal_polling = {k: (True, *v) for k, v in signal_polling.items()}
        return cls

    return decorator


def tango_create_children_from_annotations(
    device: TangoDevice,
    included_optional_fields: Tuple[str, ...] = (),
    device_vectors: Optional[Dict[str, int]] = None,
):
    """Initialize blocks at __init__ of `device`."""
    for name, device_type in get_type_hints(type(device)).items():
        if name in ("_name", "parent"):
            continue

        device_type, is_optional = _strip_union(device_type)
        if is_optional and name not in included_optional_fields:
            continue

        is_device_vector, device_type = _strip_device_vector(device_type)
        if is_device_vector:
            kwargs = "_" + name + "_kwargs"
            kwargs = getattr(device, kwargs, {})
            prefix = kwargs["prefix"]
            count = kwargs["count"]
            n_device_vector = DeviceVector(
                {i: device_type(f"{prefix}{i}") for i in range(1, count + 1)}
            )
            setattr(device, name, n_device_vector)

        else:
            origin = get_origin(device_type)
            origin = origin if origin else device_type

            if issubclass(origin, Signal):
                datatype = None
                tango_name = name.lstrip("_")
                read_trl = f"{device.trl}/{tango_name}"
                type_args = get_args(device_type)
                if type_args:
                    datatype = type_args[0]
                backend = make_backend(
                    datatype=datatype,
                    read_trl=read_trl,
                    write_trl=read_trl,
                    device_proxy=device.proxy,
                )
                setattr(device, name, origin(name=name, backend=backend))

            elif issubclass(origin, Device) or isinstance(origin, Device):
                kwargs = "_" + name + "_kwargs"
                kwargs = getattr(device, kwargs, "")
                setattr(device, name, origin(**kwargs))


async def fill_proxy_entries(device: TangoDevice):
    proxy_trl = device.trl
    children = [name.lstrip("_") for name, _ in device.children()]
    proxy_attributes = list(device.proxy.get_attribute_list())
    proxy_commands = list(device.proxy.get_command_list())
    combined = proxy_attributes + proxy_commands

    for name in combined:
        if name not in children:
            full_trl = f"{proxy_trl}/{name}"
            try:
                auto_signal = await tango_signal_auto(
                    trl=full_trl, device_proxy=device.proxy
                )
                setattr(device, name, auto_signal)
            except RuntimeError as e:
                if "Commands with different in and out dtypes" in str(e):
                    print(
                        f"Skipping {name}. Commands with different in and out dtypes"
                        f" are not supported."
                    )
                    continue
                raise e


def _strip_union(field: Union[Union[T], T]) -> Tuple[T, bool]:
    if get_origin(field) is Union:
        args = get_args(field)
        is_optional = type(None) in args
        for arg in args:
            if arg is not type(None):
                return arg, is_optional
    return field, False


def _strip_device_vector(field: Union[Type[Device]]) -> Tuple[bool, Type[Device]]:
    if get_origin(field) is DeviceVector:
        return True, get_args(field)[0]
    return False, field
