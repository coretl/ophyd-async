from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable, Iterator, Mapping, MutableMapping
from functools import cached_property
from logging import LoggerAdapter, getLogger
from typing import Any, Generic, TypeVar
from unittest.mock import Mock

from bluesky.protocols import HasName
from bluesky.run_engine import call_in_bluesky_event_loop, in_bluesky_event_loop

from ._utils import (
    DEFAULT_TIMEOUT,
    NotConnectedError,
    error_if_none,
    wait_for_connection,
)

DeviceT = TypeVar("DeviceT", bound="Device")

DEVICE_RESERVED_ATTRS = {
    "name",
    "collect_asset_docs",
    "get_index",
    "read_configuration",
    "describe_configuration",
    "trigger",
    "prepare",
    "read",
    "describe",
    "describe_collect",
    "collect",
    "collect_pages",
    "set",
    "locate",
    "kickoff",
    "complete",
    "stage",
    "unstage",
    "pause",
    "resume",
    "stop",
    "subscribe",
    "clear_sub",
    "check_value",
    "hints",
}


def _reserved_attrs_allowed() -> bool:
    # Opt-out for the reserved-name check, mainly so downstream test suites that
    # mock protocol methods (e.g. `device.set = AsyncMock()`) can be flipped back
    # on without editing every call site. Read from the environment each time so a
    # test can toggle it; only reached when a reserved name is actually set, so it
    # never touches the hot path. Prefer set_mock_attr() for a single override.
    return os.environ.get("OPHYD_ASYNC_ALLOW_RESERVED_ATTRS", "NO").upper() == "YES"


class DeviceMock(Generic[DeviceT]):
    """A lazily created Mock to be used when connecting in mock mode.

    Creating Mocks is reasonably expensive when each Device (and Signal)
    requires its own, and the tree is only used when ``Signal.set()`` is
    called. This class allows a tree of lazily connected Mocks to be
    constructed so that when the leaf is created, so are its parents.
    Any calls to the child are then accessible from the parent mock.

    Subclasses can override the `connect()` method to inject custom logic
    when mock devices are connected.

    ```python
    >>> parent = DeviceMock()
    >>> child = DeviceMock("child", parent)
    >>> child_mock = child()
    >>> child_mock()  # doctest: +ELLIPSIS
    <Mock name='mock.child()' id='...'>
    >>> parent_mock = parent()
    >>> parent_mock.mock_calls
    [call.child()]

    ```
    """

    def __init__(self, name: str = "", parent: DeviceMock | None = None) -> None:
        self.name = name
        self.parent = parent
        self._mock: Mock | None = None
        # The per-Device-type mock class overrides in force for this connect, if
        # any (see `Device.connect`'s `mock` param). A public attribute (not
        # `_overrides`) so it can be read back by `Device.connect` when it builds
        # a mock for a child, without a private-attribute access; inherited from
        # `parent` so it reaches arbitrarily deep descendants for free.
        self.overrides: dict[type[Device], type[DeviceMock]] = (
            parent.overrides if parent is not None else {}
        )

    def __call__(self) -> Mock:
        if self._mock is None:
            self._mock = Mock(spec=object)
            if self.parent is not None:
                self.parent().attach_mock(self._mock, self.name)
        return self._mock

    async def connect(self, device: DeviceT) -> None:
        """Will be called when the device is connected in mock mode.

        This allows mock values to be set and callbacks to be added
        to the mock device so it behaves more like the real device.
        """


# Keep LazyMock as an alias for backwards compatibility
# Remove for ophyd-async 1.0
LazyMock = DeviceMock


def get_mock(device: Device | None) -> DeviceMock | None:
    """Return the `DeviceMock` `device` was last connected with, if any.

    `None` if `device` is `None` or hasn't been connected in mock mode.
    """
    return device._mock if device is not None else None  # noqa: SLF001


def _select_mock_class(
    device: Device,
    overrides: Mapping[type[Device], type[DeviceMock]],
    default: type[DeviceMock],
) -> type[DeviceMock]:
    """Pick the `DeviceMock` subclass to use for `device`.

    Walks `overrides` in insertion order and returns the first value whose key
    `device` is an instance of, falling back to `default` if none match.
    """
    for device_cls, mock_cls in overrides.items():
        if isinstance(device, device_cls):
            return mock_cls
    return default


def _local_mock_name(device: Device) -> str:
    """The attribute name `device` is registered under on its parent.

    Used as a `DeviceMock`'s own `name`, so mocks attach under the same key
    their Device does, e.g. `parent_mock.x` for a device assigned as `self.x`.
    Root devices (no parent) have no such key, so their mock name is unused.
    """
    if device.parent is None:
        return ""
    return next(
        (name for name, child in device.parent.children() if child is device),
        device.name,
    )


class DeviceConnector:
    """Defines how a `Device` should be connected and type hints processed."""

    def create_children_from_annotations(self, device: Device):
        """Use when children can be created from introspecting the hardware.

        Some control systems allow introspection of a device to determine what
        children it has. To allow this to work nicely with typing we add these
        hints to the Device like so::

            my_signal: SignalRW[int]
            my_device: MyDevice

        This method will be run during `Device.__init__`, and is responsible
        for turning all of those type hints into real Signal and Device instances.

        Subsequent runs of this function should do nothing, to allow it to be
        called early in Devices that need to pass references to their children
        during `__init__`.
        """

    async def connect_mock(self, device: Device, mock: DeviceMock):
        """Use during [](#Device.connect) with `mock=True`.

        This is called when there is no cached connect done in `mock=True`
        mode. It connects the Device and all its children in mock mode.
        """
        # Connect serially, no errors to gather up as in mock mode. Pass down
        # exactly the mock we were given: each child's own `connect()` builds
        # its own mock from it (see `Device.connect`), so this never changes.
        exceptions: dict[str, Exception] = {}
        for name, child_device in device.children():
            try:
                await child_device.connect(mock=mock)
            except Exception as exc:
                exceptions[name] = exc
        if exceptions:
            raise NotConnectedError.with_other_exceptions_logged(exceptions)

        # Call the DeviceMock's connect method to inject custom logic
        await mock.connect(device)

    async def connect_real(self, device: Device, timeout: float, force_reconnect: bool):
        """Use during [](#Device.connect) with `mock=False`.

        This is called when there is no cached connect done in `mock=False`
        mode. It connects the Device and all its children in real mode in parallel.
        """
        # Connect in parallel, gathering up NotConnectedErrors
        coros = {
            name: child_device.connect(timeout=timeout, force_reconnect=force_reconnect)
            for name, child_device in device.children()
        }
        await wait_for_connection(**coros)


def _fail_if_overwriting_parent(self: Device, name: str, value: Any):
    if self.parent not in (value, None):
        raise TypeError(
            f"Cannot set the parent of {self} to be {value}: "
            f"it is already a child of {self.parent}"
        )
    object.__setattr__(self, name, value)


def _set_device_child(self: Device, name: str, value: Device | None):
    if value is None:
        # Remove optional devices that have resolved to None
        self._child_devices.pop(name, None)
    else:
        value.parent = self
        self._child_devices[name] = value
        # And if the name is set, then set the name of all children,
        # including the child
        if self._name:
            self.set_name(self._name)
    object.__setattr__(self, name, value)


class Device(HasName):
    """Common base class for all Ophyd Async Devices.

    :param name: Optional name of the Device
    :param connector: Optional DeviceConnector instance to use at connect()
    """

    parent: Device | None = None
    """The parent Device if it exists"""
    _name: str = ""
    # None if connect hasn't started, a Task if it has
    _connect_task: asyncio.Task | None = None
    # The mock class to be used if we connect in mock mode
    _mock_class: type[DeviceMock] = DeviceMock
    # The mock if we have connected in mock mode
    _mock: DeviceMock | None = None
    # The separator to use when making child names
    _child_name_separator: str = "-"
    # Methods to call on setattr
    _setattr_methods: dict[str, Callable[[Device, str, Any], None]]

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        # These are guaranteed not to be devices, so don't check them
        setattr_methods = dict.fromkeys(_not_device_attrs, object.__setattr__) | {
            # parent needs special handling
            "parent": _fail_if_overwriting_parent,
        }
        # Assign _setattr_methods in __new__ instead of __init__,
        # as this is called before any __setattr__ calls are made
        object.__setattr__(self, "_setattr_methods", setattr_methods)
        return self

    def __init__(
        self, name: str = "", connector: DeviceConnector | None = None
    ) -> None:
        self._connector = connector or DeviceConnector()
        self._connector.create_children_from_annotations(self)
        if name:
            self.set_name(name)

    @property
    def name(self) -> str:
        """Return the name of the Device."""
        return self._name

    @cached_property
    def _child_devices(self) -> dict[str, Device]:
        return {}

    def children(self) -> Iterator[tuple[str, Device]]:
        """For each attribute that is a Device, yield the name and Device.

        :yields: `(attr_name, attr)` for each child attribute that is a Device.
        """
        yield from self._child_devices.items()

    @cached_property
    def log(self) -> LoggerAdapter:
        """Return a logger configured with the device name."""
        return LoggerAdapter(
            getLogger("ophyd_async.devices"), {"ophyd_async_device_name": self.name}
        )

    def set_name(self, name: str, *, child_name_separator: str | None = None) -> None:
        """Set `self.name=name` and each `self.child.name=name+"-child"`.

        :param name: New name to set.
        :param child_name_separator:
            Use this as a separator instead of "-". Use "_" instead to make the
            same names as the equivalent ophyd sync device.
        """
        self._name = name
        if child_name_separator:
            self._child_name_separator = child_name_separator
        # Ensure logger is recreated after a name change
        if "log" in self.__dict__:
            del self.log
        for attr_name, child in self.children():
            child_name = (
                f"{self.name}{self._child_name_separator}{attr_name}"
                if self.name
                else ""
            )
            child.set_name(child_name, child_name_separator=self._child_name_separator)

    def __setattr__(self, name: str, value: Any) -> None:
        # Bear in mind that this function is called *a lot*, so
        # we need to make sure nothing expensive happens in it, hence the
        # dictionary of setattr functions
        func = self._setattr_methods.get(name, None)
        if func is None:
            if name in DEVICE_RESERVED_ATTRS and not _reserved_attrs_allowed():
                raise NameError(
                    f"`{name}` is used in one of the bluesky protocols. "
                    f"Please use `{name}_` instead. To override this attribute in a "
                    f"test (e.g. with a mock) use ophyd_async.testing.set_mock_attr, "
                    f"or set OPHYD_ASYNC_ALLOW_RESERVED_ATTRS=YES to disable this "
                    f"check entirely."
                )
            # First encounter, so assign correct
            # __setattr__ method depending on `value` type
            if isinstance(value, Device):
                func = _set_device_child
            else:
                func = object.__setattr__
            self._setattr_methods[name] = func
        # Dispatch the correct __setattr__ method
        func(self, name, value)

    async def connect(
        self,
        mock: bool
        | type[DeviceMock]
        | dict[type[Device], type[DeviceMock]]
        | DeviceMock = False,
        timeout: float = DEFAULT_TIMEOUT,
        force_reconnect: bool = False,
    ) -> None:
        """Connect the device and all child devices.

        Successful connects will be cached so subsequent calls will return
        immediately. Contains a timeout that gets propagated to child.connect
        methods.

        :param mock:
            If `False` then connect for real. If `True` then use
            [](#MockSignalBackend) for all Signals, creating this Device's mock
            from its registered default (or a plain [](#DeviceMock) if none is
            registered). If passed a [](#DeviceMock) instance then use it as-is.
            If passed a `DeviceMock` subclass then use that class for this
            Device instead of its registered default; descendants keep using
            their own registered defaults. If passed a `dict` mapping Device
            types to `DeviceMock` subclasses then apply it to every Device in
            the tree (this one included): for each, use the value of the first
            key it is an `isinstance` of, falling back to that Device's
            registered default if none match.
        :param timeout: Time to wait before failing with a TimeoutError.
        :param force_reconnect:
            If True, force a reconnect even if the last connect succeeded.
        """
        connector: DeviceConnector = error_if_none(
            getattr(self, "_connector", None),
            f"{self}: doesn't have attribute `_connector`,"
            f" did you call `super().__init__` in your `__init__` method?",
        )
        if mock is not False:
            # Always connect in mock mode serially. `connector.connect_mock`
            # passes each child exactly the `mock` its parent was given (see
            # `DeviceConnector.connect_mock`), so a `DeviceMock` instance here
            # is either a mock a caller supplied directly for this device, or
            # one this device's own parent was just resolved to: only the
            # former should be adopted as-is, the latter is a signal to build
            # a fresh mock parented off it instead.
            parent_mock = get_mock(self.parent)
            if isinstance(mock, DeviceMock) and mock is not parent_mock:
                # Use the caller-supplied mock for this device directly
                self._mock = mock
            elif not self._mock:
                # Resolve the override map in force: an explicit dict here, or
                # else whatever my parent's own mock is carrying (if any)
                overrides = (
                    mock
                    if isinstance(mock, dict)
                    else parent_mock.overrides
                    if parent_mock is not None
                    else {}
                )
                mock_cls = (
                    mock
                    if isinstance(mock, type)
                    else _select_mock_class(self, overrides, self._mock_class)
                )
                new_mock = mock_cls(_local_mock_name(self), parent_mock)
                new_mock.overrides = overrides
                self._mock = new_mock
            await connector.connect_mock(self, self._mock)
        else:
            # Try to cache the connect in real mode
            can_use_previous_connect = (
                self._mock is None
                and self._connect_task
                and not (self._connect_task.done() and self._connect_task.exception())
            )
            if force_reconnect or not can_use_previous_connect:
                self._mock = None
                coro = connector.connect_real(self, timeout, force_reconnect)
                self._connect_task = asyncio.create_task(coro)
            connect_task = error_if_none(
                self._connect_task, "Connect task not created, this shouldn't happen"
            )
            # Wait for it to complete
            await connect_task

    def __repr__(self) -> str:
        if self.name == "":
            return super().__repr__()
        return f'{type(self).__name__}(name="{self.name}")'

    def __str__(self) -> str:
        return repr(self)


_not_device_attrs = {
    "_name",
    "_children",
    "_connector",
    "_timeout",
    "_mock",
    "_connect_task",
    "_child_name_separator",
    "_attempts",
}


class DeviceVector(MutableMapping[int, DeviceT], Device):
    """Defines a dictionary of Device children with arbitrary integer keys.

    :see-also: [](#implementing-devices) for examples of how to use this class.
    """

    def __init__(
        self,
        children: Mapping[int, DeviceT] | None = None,
        name: str = "",
        connector: DeviceConnector | None = None,
    ) -> None:
        self._children: dict[int, DeviceT] = {}
        self.update(children or {})
        super().__init__(name=name, connector=connector)

    def __getitem__(self, key: int) -> DeviceT:
        return self._children[key]

    def __setitem__(self, key: int, value: DeviceT) -> None:
        # Check the types on entry to dict to make sure we can't accidentally
        # make a non-integer named child
        if not isinstance(key, int):
            msg = f"Expected int, got {key}"
            raise TypeError(msg)
        if not isinstance(value, Device):
            msg = f"Expected Device, got {value}"
            raise TypeError(msg)
        self._children[key] = value
        value.parent = self

    def __delitem__(self, key: int) -> None:
        del self._children[key]

    def __iter__(self) -> Iterator[int]:
        yield from self._children

    def __len__(self) -> int:
        return len(self._children)

    def children(self) -> Iterator[tuple[str, Device]]:
        for key, child in self._children.items():
            yield str(key), child
        yield from super().children()

    def __hash__(self):  # to allow DeviceVector to be used as dict keys and in sets
        return hash(id(self))


class DeviceMap(MutableMapping[str, DeviceT], Device):
    """Defines a dictionary of Device children with arbitrary string keys.

    Like [](#DeviceVector) but indexed by `str` rather than `int`, for when
    sub-devices are more naturally addressed by name than by number.

    :see-also: [](#implementing-devices) for examples of how to use this class.
    """

    def __init__(
        self,
        children: Mapping[str, DeviceT] | None = None,
        name: str = "",
        connector: DeviceConnector | None = None,
    ) -> None:
        self._children: dict[str, DeviceT] = {}
        self.update(children or {})
        super().__init__(name=name, connector=connector)

    def __getitem__(self, key: str) -> DeviceT:
        return self._children[key]

    def __setitem__(self, key: str, value: DeviceT) -> None:
        # Check the types on entry to dict to make sure we can't accidentally
        # make a non-string named child
        if not isinstance(key, str):
            msg = f"Expected str, got {key}"
            raise TypeError(msg)
        if not isinstance(value, Device):
            msg = f"Expected Device, got {value}"
            raise TypeError(msg)
        self._children[key] = value
        value.parent = self

    def __setattr__(self, name: str, child: Any) -> None:
        # Child Devices must be set via `device_map[key] = child` so they get a
        # string key; setting them as attributes would give them no key.
        if name != "parent" and isinstance(child, Device):
            raise AttributeError(
                "DeviceMap can only have string named children, "
                "set via device_map[key] = child"
            )
        super().__setattr__(name, child)

    def __delitem__(self, key: str) -> None:
        del self._children[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._children

    def __len__(self) -> int:
        return len(self._children)

    def children(self) -> Iterator[tuple[str, Device]]:
        # Keys are already str, so yield them directly (no str() needed)
        yield from self._children.items()
        yield from super().children()

    def __hash__(self):  # to allow DeviceMap to be used as dict keys and in sets
        return hash(id(self))


class DeviceProcessor:
    """Sync/Async Context Manager that finds all the Devices declared within it.

    Used in `init_devices`
    """

    def __init__(self, process_devices: Callable[[dict[str, Device]], Awaitable[None]]):
        self._process_devices = process_devices
        self._locals_on_enter: dict[str, Any] = {}
        self._locals_on_exit: dict[str, Any] = {}

    def _caller_locals(self) -> dict[str, Any]:
        """Walk up until we find a stack frame that doesn't have us as self."""
        try:
            raise ValueError
        except ValueError:
            _, _, tb = sys.exc_info()
            tb = error_if_none(tb, "Can't get traceback, this shouldn't happen")

            caller_frame = tb.tb_frame
            while caller_frame.f_locals.get("self", None) is self:
                caller_frame = caller_frame.f_back
                if not caller_frame:
                    msg = (
                        "No previous frame to the one with self in it, "
                        "this shouldn't happen"
                    )
                    raise RuntimeError(  # noqa: B904
                        msg
                    )
            return caller_frame.f_locals.copy()

    def __enter__(self) -> DeviceProcessor:
        # Stash the names that were defined before we were called
        self._locals_on_enter = self._caller_locals()
        return self

    async def __aenter__(self) -> DeviceProcessor:
        return self.__enter__()

    async def __aexit__(self, type, value, traceback):
        self._locals_on_exit = self._caller_locals()
        await self._on_exit()

    def __exit__(self, type_, value, traceback):
        if in_bluesky_event_loop():
            raise RuntimeError(
                "Cannot use DeviceConnector inside a plan, instead use "
                "`yield from ophyd_async.plan_stubs.ensure_connected(device)`"
            )
        self._locals_on_exit = self._caller_locals()
        try:
            fut = call_in_bluesky_event_loop(self._on_exit())
        except RuntimeError as exc:
            raise NotConnectedError(
                "Could not connect devices. Is the bluesky event loop running? See "
                "https://blueskyproject.io/ophyd-async/main/"
                "user/explanations/event-loop-choice.html for more info."
            ) from exc
        return fut

    async def _on_exit(self) -> None:
        # Find all the devices
        devices = {
            name: obj
            for name, obj in self._locals_on_exit.items()
            if isinstance(obj, Device) and self._locals_on_enter.get(name) is not obj
        }
        # Call the provided process function on them
        await self._process_devices(devices)


def init_devices(
    set_name: bool = True,
    child_name_separator: str = "-",
    connect: bool = True,
    mock: bool
    | type[DeviceMock]
    | dict[type[Device], type[DeviceMock]]
    | DeviceMock = False,
    timeout: float = 10.0,
):
    """Auto initialize top level Device instances: to be used as a context manager.

    :param set_name:
        If True, call `device.set_name(variable_name)` on all Devices created
        within the context manager that have an empty `name`.
    :param child_name_separator: Separator for child names if `set_name` is True.
    :param connect:
        If True, call `device.connect(mock, timeout)` in parallel on all Devices
        created within the context manager.
    :param mock:
        Passed straight through to [](#Device.connect) for every Device created
        within the context manager; see its `mock` parameter for the accepted
        forms. Note that a `DeviceMock` instance (unlike the other forms) would
        then be shared across all of them.
    :param timeout: How long to wait for connect before logging an exception.
    :raises RuntimeError: If used inside a plan, use [](#ensure_connected) instead.
    :raises NotConnectedError: If devices could not be connected.

    For example, to connect and name 2 motors in parallel:
    ```python
    [async] with init_devices():
        t1x = motor.Motor("BLxxI-MO-TABLE-01:X")
        t1y = motor.Motor("pva://BLxxI-MO-TABLE-01:Y")
        # Names and connects devices here
    assert t1x.name == "t1x"
    ```
    """

    async def process_devices(devices: dict[str, Device]):
        if set_name:
            for name, device in devices.items():
                if not device.name:
                    device.set_name(name, child_name_separator=child_name_separator)
        if connect:
            coros = {
                name: device.connect(mock, timeout) for name, device in devices.items()
            }
            await wait_for_connection(**coros)

    return DeviceProcessor(process_devices)


def default_mock_class(
    mock_cls: type[DeviceMock],
) -> Callable[[type[DeviceT]], type[DeviceT]]:
    """Register a DeviceMock subclass as the default mock for a Device class.

    This decorator allows automatic injection of mock logic when devices are
    connected in mock mode. The decorated DeviceMock class should override
    the `connect()` method to define custom mock behavior.

    :param mock_cls: A DeviceMock subclass to register.
    :returns: A decorator that registers the mock class for a Device subclass.
    """

    def wrapper(device_cls: type[DeviceT]) -> type[DeviceT]:
        device_cls._mock_class = mock_cls  # noqa: SLF001
        return device_cls

    return wrapper
