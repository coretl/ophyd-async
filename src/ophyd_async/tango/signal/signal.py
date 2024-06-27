"""Tango Signals over Pytango"""

from __future__ import annotations

from typing import Optional, Type

from ophyd_async.core import SignalR, SignalRW, SignalW, SignalX, T
from ophyd_async.tango._backend import (
    TangoSignalBackend,
    TangoTransport,
)
from ophyd_async.core.utils import DEFAULT_TIMEOUT
from tango.asyncio import DeviceProxy

__all__ = ("tango_signal_rw", "tango_signal_r", "tango_signal_w", "tango_signal_x")


def _make_backend(
    datatype: Optional[Type[T]],
    read_trl: str,
    write_trl: str,
    device_proxy: Optional[DeviceProxy] = None,
) -> TangoSignalBackend:
    return TangoTransport(datatype, read_trl, write_trl, device_proxy)


# --------------------------------------------------------------------
def tango_signal_rw(
    datatype: Type[T],
    read_trl: str,
    write_trl: Optional[str] = None,
    device_proxy: Optional[DeviceProxy] = None,
    timeout: float = DEFAULT_TIMEOUT,
    name: str = "",
) -> SignalRW[T]:
    """Create a `SignalRW` backed by 1 or 2 Tango Attribute/Command

    Parameters
    ----------
    datatype:
        Check that the Attribute/Command is of this type
    read_trl:
        The Attribute/Command to read and monitor
    write_trl:
        If given, use this Attribute/Command to write to, otherwise use read_trl
    device_proxy:
        If given, this DeviceProxy will be used
    timeout:
        The timeout for the read and write operations
    name:
        The name of the Signal
    """
    backend = _make_backend(datatype, read_trl, write_trl or read_trl, device_proxy)
    return SignalRW(backend, timeout=timeout, name=name)


# --------------------------------------------------------------------
def tango_signal_r(
    datatype: Type[T],
    read_trl: str,
    device_proxy: Optional[DeviceProxy] = None,
    timeout: float = DEFAULT_TIMEOUT,
    name: str = "",
) -> SignalR[T]:
    """Create a `SignalR` backed by 1 Tango Attribute/Command

    Parameters
    ----------
    datatype:
        Check that the Attribute/Command is of this type
    read_trl:
        The Attribute/Command to read and monitor
    device_proxy:
        If given, this DeviceProxy will be used
    timeout:
        The timeout for the read operation
    name:
        The name of the Signal
    """
    backend = _make_backend(datatype, read_trl, read_trl, device_proxy)
    return SignalR(backend, timeout=timeout, name=name)


# --------------------------------------------------------------------
def tango_signal_w(
    datatype: Type[T],
    write_trl: str,
    device_proxy: Optional[DeviceProxy] = None,
    timeout: float = DEFAULT_TIMEOUT,
    name: str = "",
) -> SignalW[T]:
    """Create a `TangoSignalW` backed by 1 Tango Attribute/Command

    Parameters
    ----------
    datatype:
        Check that the Attribute/Command is of this type
    write_trl:
        The Attribute/Command to write to
    device_proxy:
        If given, this DeviceProxy will be used
    timeout:
        The timeout for the write operation
    name:
        The name of the Signal
    """
    backend = _make_backend(datatype, write_trl, write_trl, device_proxy)
    return SignalW(backend, timeout=timeout, name=name)


# --------------------------------------------------------------------
def tango_signal_x(
    write_trl: str,
    device_proxy: Optional[DeviceProxy] = None,
    timeout: float = DEFAULT_TIMEOUT,
    name: str = "",
) -> SignalX:
    """Create a `SignalX` backed by 1 Tango Attribute/Command

    Parameters
    ----------
    write_trl:
        The Attribute/Command to write its initial value to on execute
    device_proxy:
        If given, this DeviceProxy will be used
    timeout:
        The timeout for the command operation
    name:
        The name of the Signal
    """
    backend = _make_backend(None, write_trl, write_trl, device_proxy)
    return SignalX(backend, timeout=timeout, name=name)


# --------------------------------------------------------------------
# def tango_signal_auto(
#     datatype: Type[T], full_trl: str, device_proxy: Optional[DeviceProxy] = None
# ) -> Union[TangoSignalW, TangoSignalX, TangoSignalR, TangoSignalRW]:
#     backend: TangoSignalBackend = TangoTransport(
#         datatype, full_trl, full_trl, device_proxy
#     )

#     device_trl, tr_name = full_trl.rsplit("/", 1)
#     device_proxy = SyncDeviceProxy(device_trl)
#     if tr_name in device_proxy.get_attribute_list():
#         config = device_proxy.get_attribute_config(tr_name)
#         if config.writable in [AttrWriteType.READ_WRITE,
#             AttrWriteType.READ_WITH_WRITE]:
#             return TangoSignalRW(backend)
#         elif config.writable == AttrWriteType.READ:
#             return TangoSignalR(backend)
#         else:
#             return TangoSignalW(backend)

#     if tr_name in device_proxy.get_command_list():
#         config = device_proxy.get_command_config(tr_name)
#         if config.in_type == CmdArgType.DevVoid:
#             return TangoSignalX(backend)
#         elif config.out_type != CmdArgType.DevVoid:
#             return TangoSignalRW(backend)
#         else:
#             return TangoSignalX(backend)

#     if tr_name in device_proxy.get_pipe_list():
#         raise NotImplementedError("Pipes are not supported")

#     raise RuntimeError(f"Cannot find {tr_name} in {device_trl}")
