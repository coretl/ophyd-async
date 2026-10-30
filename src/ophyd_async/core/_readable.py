import asyncio
import itertools
import warnings
from collections.abc import (
    Awaitable,
    Callable,
    Generator,
    Iterator,
    MutableMapping,
    Sequence,
)
from contextlib import contextmanager
from enum import Enum
from functools import cached_property
from typing import Literal

from bluesky.protocols import HasHints, Hints, Reading
from event_model import DataKey

from ._device import Device, DeviceMap, DeviceVector
from ._protocol import AsyncConfigurable, AsyncReadable, AsyncStageable
from ._signal import SignalR, walk_devices
from ._standard_base import _StandardBase
from ._utils import merge_gathered_dicts


class StandardReadableFormat(Enum):
    """Declare how a `Device` should contribute to the `StandardReadable` verbs."""

    CHILD = "CHILD"
    """Detect which verbs the child supports and contribute to:

    - `read()`, `describe()` if it is [](#bluesky.protocols.Readable)
    - `read_configuration()`, `describe_configuration()` if it is
      [](#bluesky.protocols.Configurable)
    - `stage()`, `unstage()` if it is [](#bluesky.protocols.Stageable)
    - `hints` if it [](#bluesky.protocols.HasHints)
    """

    CONFIG_SIGNAL = "CONFIG_SIGNAL"
    """Contribute the `Signal` value to `read_configuration()` and
    `describe_configuration()`
    """

    HINTED_SIGNAL = "HINTED_SIGNAL"
    """Contribute the monitored `Signal` value to `read()` and `describe()` and
    put the signal name in `hints`
    """

    UNCACHED_SIGNAL = "UNCACHED_SIGNAL"
    """Contribute the uncached `Signal` value to `read()` and `describe()`
    """

    HINTED_UNCACHED_SIGNAL = "HINTED_UNCACHED_SIGNAL"
    """Contribute the uncached `Signal` value to `read()` and `describe()` and
    put the signal name in `hints`
    """

    def __call__(self, parent: Device, child: Device):
        if not isinstance(parent, StandardReadable):
            raise TypeError(f"Expected parent to be StandardReadable, got {parent}")
        parent.set_readable_format(child, self)


# Short alias for the case statements below
Format = StandardReadableFormat

SignalFormat = Literal[
    Format.CONFIG_SIGNAL,
    Format.HINTED_SIGNAL,
    Format.UNCACHED_SIGNAL,
    Format.HINTED_UNCACHED_SIGNAL,
]

#: A registered child and its format. Only a `SignalR` has a signal format.
ReadableEntry = tuple[Device, Literal[Format.CHILD]] | tuple[SignalR, SignalFormat]


class _Readables(MutableMapping[Device, StandardReadableFormat]):
    """The registered children and their formats, in registration order.

    Only a `SignalR` may be given a signal format.
    """

    def __init__(self) -> None:
        self._entries: dict[Device, ReadableEntry] = {}

    def __getitem__(self, device: Device) -> StandardReadableFormat:
        return self._entries[device][1]

    def __setitem__(self, device: Device, format: StandardReadableFormat) -> None:
        if not isinstance(format, StandardReadableFormat):
            raise TypeError(f"{format} is not a StandardReadableFormat")
        match format:
            case Format.CHILD:
                self._entries[device] = (device, format)
            case (
                Format.CONFIG_SIGNAL
                | Format.HINTED_SIGNAL
                | Format.UNCACHED_SIGNAL
                | Format.HINTED_UNCACHED_SIGNAL
            ):
                if not isinstance(device, SignalR):
                    raise TypeError(f"{device} is not a SignalR")
                self._entries[device] = (device, format)

    def __delitem__(self, device: Device) -> None:
        del self._entries[device]

    def __iter__(self) -> Iterator[Device]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[ReadableEntry]:
        """A snapshot of each child and its format, in registration order."""
        return list(self._entries.values())


class _SealDefaultFormatsMeta(type(_StandardBase)):  # type: ignore[misc]
    """Record the formats a `StandardReadable` was constructed with.

    The snapshot is taken after the whole `__init__` chain has returned, so it
    holds the formats from annotations and from registration before or after
    `super().__init__()`.
    """

    def __call__(cls, *args, **kwargs):
        self = super().__call__(*args, **kwargs)
        self._default_readables = dict(self._readables)
        return self


class StandardReadable(
    _StandardBase,
    AsyncReadable,
    AsyncConfigurable,
    HasHints,
    metaclass=_SealDefaultFormatsMeta,
):
    """Device that provides selected child Device values in `read()`.

    Provides the ability for children to be registered to:
    - Participate in `stage()` and `unstage()`
    - Provide their value in `read()` and `describe()
    - Provide their value in `read_configuration()` and `describe_configuration()
    - Select a value to appear in `hints`

    The behavior is customized with a [](#StandardReadableFormat), which can be
    changed at runtime with [](#StandardReadable.set_readable_format).
    """

    @cached_property
    def _readables(self) -> _Readables:
        """The registered children and their formats, in registration order.

        A `cached_property`, so `add_children_as_readables` and
        `set_readable_format` can be called before `super().__init__()`.
        """
        return _Readables()

    # The formats declared during construction, sealed by the metaclass once
    # __init__ has fully returned. Only ever replaced, never mutated in place,
    # so the shared class level default is safe.
    _default_readables: dict[Device, StandardReadableFormat] = {}

    # What contributes to each verb, gathered in parallel by the verb methods.
    # Immutable defaults to avoid accidental sharing between instances. A
    # subclass producing data from somewhere other than a registered child adds
    # to these rather than overriding the verbs, so registered children keep
    # working; `StandardDetector` does it for its data logics. Each registry
    # is named after the method that gathers it, and each hook added to it is
    # `__on_<method>`, name-mangled so mix-ins in one MRO never collide.
    _read_funcs: tuple[Callable[[], Awaitable[dict[str, Reading]]], ...] = ()
    _describe_funcs: tuple[Callable[[], Awaitable[dict[str, DataKey]]], ...] = ()
    _read_configuration_funcs: tuple[
        Callable[[], Awaitable[dict[str, Reading]]], ...
    ] = ()
    _describe_configuration_funcs: tuple[
        Callable[[], Awaitable[dict[str, DataKey]]], ...
    ] = ()
    _hints_funcs: tuple[Callable[[], Iterator[Hints]], ...] = ()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Each of these derives from _readables on every call rather than being
        # registered up-front, so a format changed between runs is picked up on
        # the next stage() or the next descriptor.
        self._stage_funcs += (self.__on_stage,)
        self._unstage_funcs += (self.__on_unstage,)
        self._read_funcs += (self.__on_read,)
        self._describe_funcs += (self.__on_describe,)
        self._read_configuration_funcs += (self.__on_read_configuration,)
        self._describe_configuration_funcs += (self.__on_describe_configuration,)
        self._hints_funcs += (self.__on_hints,)

    def _signals_to_stage(self) -> Iterator[AsyncStageable]:
        # HINTED_SIGNAL stages so that caching is set up and read() is fast; a
        # CHILD stages because it may have staging of its own. The uncached and
        # config formats deliberately do not.
        for entry in self._readables.entries():
            match entry:
                case (device, Format.CHILD):
                    if isinstance(device, AsyncStageable):
                        yield device
                case (signal, Format.HINTED_SIGNAL):
                    yield signal
                case (
                    _,
                    Format.CONFIG_SIGNAL
                    | Format.UNCACHED_SIGNAL
                    | Format.HINTED_UNCACHED_SIGNAL,
                ):
                    pass

    async def __on_stage(self) -> None:
        await asyncio.gather(*(sig.stage().task for sig in self._signals_to_stage()))

    async def __on_unstage(self) -> None:
        await asyncio.gather(*(sig.unstage().task for sig in self._signals_to_stage()))

    async def __on_describe_configuration(self) -> dict[str, DataKey]:
        """What the registered children contribute to `describe_configuration()`."""
        coros = []
        for entry in self._readables.entries():
            match entry:
                case (device, Format.CHILD):
                    if isinstance(device, AsyncConfigurable):
                        coros.append(device.describe_configuration())
                case (signal, Format.CONFIG_SIGNAL):
                    coros.append(signal.describe())
                case (
                    _,
                    Format.HINTED_SIGNAL
                    | Format.UNCACHED_SIGNAL
                    | Format.HINTED_UNCACHED_SIGNAL,
                ):
                    pass
        return await merge_gathered_dicts(coros)

    async def __on_read_configuration(self) -> dict[str, Reading]:
        """What the registered children contribute to `read_configuration()`."""
        coros = []
        for entry in self._readables.entries():
            match entry:
                case (device, Format.CHILD):
                    if isinstance(device, AsyncConfigurable):
                        coros.append(device.read_configuration())
                case (signal, Format.CONFIG_SIGNAL):
                    coros.append(signal.read())
                case (
                    _,
                    Format.HINTED_SIGNAL
                    | Format.UNCACHED_SIGNAL
                    | Format.HINTED_UNCACHED_SIGNAL,
                ):
                    pass
        return await merge_gathered_dicts(coros)

    async def __on_describe(self) -> dict[str, DataKey]:
        """What the registered children contribute to `describe()`."""
        coros = []
        for entry in self._readables.entries():
            match entry:
                case (device, Format.CHILD):
                    if isinstance(device, AsyncReadable):
                        coros.append(device.describe())
                case (
                    signal,
                    Format.HINTED_SIGNAL
                    | Format.UNCACHED_SIGNAL
                    | Format.HINTED_UNCACHED_SIGNAL,
                ):
                    coros.append(signal.describe())
                case (_, Format.CONFIG_SIGNAL):
                    pass
        return await merge_gathered_dicts(coros)

    async def __on_read(self) -> dict[str, Reading]:
        """What the registered children contribute to `read()`."""
        coros = []
        for entry in self._readables.entries():
            match entry:
                case (device, Format.CHILD):
                    if isinstance(device, AsyncReadable):
                        coros.append(device.read())
                case (signal, Format.HINTED_SIGNAL):
                    coros.append(signal.read())
                case (
                    signal,
                    Format.UNCACHED_SIGNAL | Format.HINTED_UNCACHED_SIGNAL,
                ):
                    coros.append(signal.read(cached=False))
                case (_, Format.CONFIG_SIGNAL):
                    pass
        return await merge_gathered_dicts(coros)

    def __on_hints(self) -> Iterator[Hints]:
        """What the registered children contribute to `hints`."""
        for entry in self._readables.entries():
            match entry:
                case (device, Format.CHILD):
                    if isinstance(device, HasHints):
                        yield device.hints
                case (
                    signal,
                    Format.HINTED_SIGNAL | Format.HINTED_UNCACHED_SIGNAL,
                ):
                    # An unnamed signal has nothing to put in `fields`, but is
                    # still hinted, so it contributes the key and no name
                    yield {"fields": [signal.name] if signal.name else []}
                case (
                    _,
                    Format.CONFIG_SIGNAL | Format.UNCACHED_SIGNAL,
                ):
                    pass

    async def describe_configuration(self) -> dict[str, DataKey]:
        return await merge_gathered_dicts(
            [func() for func in self._describe_configuration_funcs]
        )

    async def read_configuration(self) -> dict[str, Reading]:
        return await merge_gathered_dicts(
            [func() for func in self._read_configuration_funcs]
        )

    async def describe(self) -> dict[str, DataKey]:
        return await merge_gathered_dicts([func() for func in self._describe_funcs])

    async def read(self) -> dict[str, Reading]:
        return await merge_gathered_dicts([func() for func in self._read_funcs])

    @property
    def hints(self) -> Hints:
        # fields and dimensions keys from sub devices are flattened
        # into the hints.
        flattened_fields = []
        flattened_dimensions = []

        hints: Hints = {}
        for new_hint in itertools.chain.from_iterable(
            func() for func in self._hints_funcs
        ):
            # If a hints key in a subdevice isn't given in bluesky protocols, then it
            # doesn't carry over to the device holding it.
            for key, value in new_hint.items():
                if key in ("fields", "dimensions"):
                    if isinstance(value, str) or not isinstance(value, Sequence):
                        raise RuntimeError(
                            f"{self.name}: '{key}' key must be a Sequence, "
                            f"received {type(value)}"
                        )
                    already_parsed = (
                        flattened_fields if key == "fields" else flattened_dimensions
                    )
                    if common := [v for v in value if v in already_parsed]:
                        raise RuntimeError(
                            f"Hint '{key}' {list(common)} override existing hint."
                        )

                    already_parsed += list(value)
                elif key == "gridding":
                    if (already_set := hints.get(key, None)) and already_set != value:
                        raise RuntimeError("Hint key 'gridding' may not be overridden.")
                    hints["gridding"] = value  # type: ignore

        if flattened_fields:
            hints["fields"] = flattened_fields
        if flattened_dimensions:
            hints["dimensions"] = flattened_dimensions

        return hints

    @contextmanager
    def add_children_as_readables(
        self,
        format: StandardReadableFormat = StandardReadableFormat.CHILD,
    ) -> Generator[None, None, None]:
        """Register child Devices added within with the given format.

        Scans `self.children()` on entry and exit to context manager, and calls
        [](#StandardReadable.set_readable_format) on any that are added with the
        provided `StandardReadableFormat`.
        """
        dict_copy = dict(self.children())

        yield

        new_dict = dict(self.children())
        for key, value in new_dict.items():
            # Check if key already exists in dict_copy and if the value has changed.
            if key in dict_copy and value != dict_copy[key]:
                error_msg = (
                    f"Duplicate readable device found: '{key}' in {value.parent}. "
                    "Derived class must not redefine a readable. "
                    "See: https://github.com/bluesky/ophyd-async/issues/848. "
                    "If this functionality is required, please raise an issue: "
                    "https://github.com/bluesky/ophyd-async"
                )
                raise KeyError(error_msg)

        # Newly added children, in the order they were declared
        new_values = [value for key, value in new_dict.items() if key not in dict_copy]

        flattened_values = []
        for value in new_values:
            # DeviceVector and DeviceMap case.
            if isinstance(value, (DeviceVector, DeviceMap)):
                flattened_values.extend(value.values())
            else:
                flattened_values.append(value)

        for device in flattened_values:
            if isinstance(device, Device):
                self.set_readable_format(device, format)

    def set_readable_format(
        self, device: Device, format: StandardReadableFormat | None
    ) -> None:
        """Set how a child Device contributes to this Device's bluesky verbs.

        This is the runtime equivalent of the `Kind` attribute in ophyd v1: it
        can be called after construction to move a child between configuration,
        hinted and uncached reads, or to drop it from the verbs entirely.

        Call it **between runs**. The descriptor for a run is emitted at its
        start, and [](#StandardReadableFormat.HINTED_SIGNAL) sets up caching in
        `stage()`, so changing a format part way through a run would make
        `describe()` and `read()` disagree.

        :param device: The Device to set the format of, normally a child of this one
        :param format:
            The format to give it, or `None` to stop it contributing at all.
            Replaces any format the device already has, rather than adding a
            second contribution.
        :raises TypeError: If a signal-only format is given a non-`SignalR`
        """
        if format is None:
            self._readables.pop(device, None)
        else:
            # Re-formatting an already registered child keeps its position, so
            # changing a format does not reorder `hints`
            self._readables[device] = format

    def reset_readable_formats(self) -> None:
        """Undo every runtime format change, back to how the class declared them.

        The baseline is everything registered by the time construction
        finished: annotations, `add_children_as_readables`, and any
        [](#StandardReadable.set_readable_format) an `__init__` made. Anything
        done after that is undone: switching a child between config and hinted,
        dropping one with a format of `None`, or registering a new one.

        Intended for a Device that is retuned per technique, so a scan can put
        it back without knowing what it changed.

        This resets **this Device only**, not its children. To reset a whole
        tree use `apply_readable_formats(device, {})`.

        The baseline is the *class declaration*, not the last stored settings
        file, so this discards a technique loaded by [](#apply_settings) too.
        """
        # Mutated rather than replaced, so the cached_property stays the same
        # object and nothing holding a reference sees a stale registry
        self._readables.clear()
        self._readables.update(self._default_readables)

    def get_readable_formats(self) -> Iterator[ReadableEntry]:
        """Yield each registered child and its format, in registration order.

        A Device that does not contribute is not yielded. It is safe to call
        [](#StandardReadable.set_readable_format) while iterating.
        """
        return iter(self._readables.entries())

    def get_readable_format_changes(
        self,
    ) -> Iterator[tuple[Device, StandardReadableFormat | None]]:
        """Yield the formats that differ from those declared at construction.

        A child registered at construction but since dropped has a format of `None`.
        It is safe to call [](#StandardReadable.set_readable_format) while iterating.
        """
        changes: list[tuple[Device, StandardReadableFormat | None]] = [
            (child, format)
            for child, format in self._readables.items()
            if self._default_readables.get(child) is not format
        ]
        changes.extend(
            (child, None)
            for child in self._default_readables
            if child not in self._readables
        )
        return iter(changes)

    # Back compat - delete before 1.0
    def add_readables(
        self,
        devices: Sequence[Device],
        format: StandardReadableFormat = StandardReadableFormat.CHILD,
    ) -> None:
        """Add devices to contribute to various bluesky verbs.

        :param devices: The devices to be added
        :param format:
            Determines which of the devices functions are added to which verb as
            per the [](#StandardReadableFormat) documentation
        """
        warnings.warn(
            DeprecationWarning(
                "Use `set_readable_format(device, format)` on each device "
                "instead of `add_readables(devices, format)`"
            ),
            stacklevel=2,
        )
        for device in devices:
            self.set_readable_format(device, format)


def _config_signals(device: Device) -> set[SignalR]:
    """Return every `Signal` contributing to a Device's `read_configuration()`.

    Recurses into children registered as [](#StandardReadableFormat.CHILD), so
    a Device that registers a sub-Device whole picks up whatever that
    sub-Device declares as configuration. A Device that is not a
    `StandardReadable` contributes nothing.

    Unlike `walk_config_signals` this reads the registry rather than calling
    `read_configuration()`, so it does no I/O and includes read-only `SignalR`s
    as well as `SignalRW`s (`ADBaseIO.model` is a `SignalR`, and it is exactly
    the signal an areaDetector trigger logic needs for deadtime).
    """
    signals: set[SignalR] = set()
    if not isinstance(device, StandardReadable):
        return signals
    for entry in device.get_readable_formats():
        match entry:
            case (signal, Format.CONFIG_SIGNAL):
                signals.add(signal)
            case (child, Format.CHILD):
                signals |= _config_signals(child)
            case _:
                pass
    return signals


#: Reserved key under which [](#store_settings) writes readable formats. Chosen
#: so that it cannot be a dotted attribute path: `<` cannot start a Python
#: identifier, so no attribute assignment can produce a colliding key, and
#: unlike `*FORMATS*` it needs no quoting in yaml.
READABLE_FORMATS_KEY = "<READABLE_FORMATS>"

#: The key standing in for the root Device itself in a [](#ReadableFormats),
#: where every other key is a dotted attribute path. Not a valid Python
#: identifier, so no attribute assignment can produce a colliding path, and
#: unlike `""` it does not make yaml fall back to its hard-to-read explicit
#: key syntax.
ROOT_DEVICE_KEY = "<ROOT_DEVICE>"

#: The changes to the readable formats of a Device tree since construction, in
#: a form that can be stored and retrieved:
#: `{path of the StandardReadable: {path of the child: format}}`, where a format
#: of `None` means "stop this child contributing".
#: The owner path is a dotted attribute path from the root Device, as produced
#: by [](#walk_devices), with `ROOT_DEVICE_KEY` meaning the root Device
#: itself. The child path is a dotted attribute path from the owner.
ReadableFormats = dict[str, dict[str, StandardReadableFormat | None]]


def _paths_to_devices(device: Device) -> dict[str, Device]:
    return {ROOT_DEVICE_KEY: device, **walk_devices(device)}


def walk_readable_formats(device: Device) -> ReadableFormats | None:
    """Retrieve how the readable formats in a Device tree differ from construction.

    Used as part of saving a Device, so that which signals are hinted or
    config can be restored later. Only the differences from what each
    `StandardReadable` registered during construction are returned, with a
    child it has since dropped given a format of `None`.

    :param device: The root Device to walk.
    :return:
        A [](#ReadableFormats) suitable for storing, or `None` if there is no
        `StandardReadable` in the tree.
    """
    paths_to_devices = _paths_to_devices(device)
    readables = {
        path: dev
        for path, dev in paths_to_devices.items()
        if isinstance(dev, StandardReadable)
    }
    if not readables:
        return None
    formats: ReadableFormats = {}
    for path, dev in readables.items():
        device_to_path = {child: p for p, child in walk_devices(dev).items()}
        entries: dict[str, StandardReadableFormat | None] = {}
        for child, format in dev.get_readable_format_changes():
            child_path = device_to_path.get(child)
            if child_path is None:
                # Not below its owner, so it has no stable path to store it
                # against. See https://github.com/bluesky/ophyd-async/issues/1402
                warnings.warn(
                    f"{dev.name or dev}: {child.name or child} is not within "
                    f"{dev.name or dev}, so the change to its readable format "
                    "will not be stored",
                    stacklevel=2,
                )
            else:
                entries[child_path] = format
        if entries:
            formats[path] = entries
    return formats


def apply_readable_formats(device: Device, formats: ReadableFormats) -> None:
    """Reset the readable formats in a Device tree, then apply the given changes.

    Every `StandardReadable` in the tree is first put back to what it
    registered during construction, as
    [](#StandardReadable.reset_readable_formats) does, then the changes in
    `formats` are applied on top. So an empty `formats` restores the class
    defaults, and a child added to a Device since `formats` was stored keeps
    the format its class gives it.

    To stop a child contributing, give it a format of `None`, which in a stored
    file is a null:

    ```yaml
    <READABLE_FORMATS>:
      <ROOT_DEVICE>:
        energy: HINTED_SIGNAL
        temperature: null
    ```

    :param device: The root Device the owner paths in `formats` are relative to.
    :param formats: A [](#ReadableFormats), e.g. from [](#walk_readable_formats).
    :raises KeyError: If a path does not name a Device in the tree.
    :raises TypeError: If an owner path does not name a `StandardReadable`.
    """
    devices = _paths_to_devices(device)
    resolved: list[
        tuple[StandardReadable, list[tuple[Device, StandardReadableFormat | None]]]
    ] = []
    # Resolve everything before changing anything, so a bad path cannot leave
    # the tree half applied
    for owner_path, entries in formats.items():
        owner = devices[owner_path]
        if not isinstance(owner, StandardReadable):
            raise TypeError(f"{owner_path!r} is not a StandardReadable, got {owner}")
        children = walk_devices(owner)
        readables: list[tuple[Device, StandardReadableFormat | None]] = []
        for child_path, format in entries.items():
            readables.append(
                (
                    children[child_path],
                    None if format is None else StandardReadableFormat(format),
                )
            )
        resolved.append((owner, readables))
    for dev in devices.values():
        if isinstance(dev, StandardReadable):
            dev.reset_readable_formats()
    for owner, readables in resolved:
        for child, format in readables:
            owner.set_readable_format(child, format)
