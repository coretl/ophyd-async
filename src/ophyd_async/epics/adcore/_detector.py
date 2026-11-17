import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Generic

from ophyd_async.core import (
    BoundedDataLogic,
    DetectorAcquireLogic,
    DetectorLogic,
    DetectorSuppressLogic,
    DetectorTriggerLogic,
    ReadableEntry,
    SignalR,
    StandardDetector,
    StandardReadable,
    UnboundedDataLogic,
)
from ophyd_async.core import StandardReadableFormat as Format

from ._acquire_logic import ADContAcqAcquireLogic
from ._data_logic import ADDataFactory, _read_signals, plugin_is_enabled
from ._io import (
    ADBaseIO,
    ADBaseIOT,
    NDCircularBuffIO,
    NDPluginBaseIO,
    NDPluginBaseIOT,
    NDProcessIO,
)
from ._trigger_logic import ADContAcqTriggerLogic


def _readable_plugins(
    entries: Iterator[ReadableEntry],
) -> Iterator[NDPluginBaseIO]:
    """The plugins registered as CHILD readables, down the readable tree."""
    for entry in entries:
        match entry:
            case (StandardReadable() as child, Format.CHILD):
                if isinstance(child, NDPluginBaseIO):
                    yield child
                yield from _readable_plugins(child.get_readable_formats())
            case _:
                pass


@dataclass(frozen=True)
class _PluginSuppressLogic(DetectorSuppressLogic):
    """Hides the read signals of each readable plugin that its switches have off.

    :param get_readable_formats: the detector's, the top of the readable tree
        that holds the plugins to ask
    """

    get_readable_formats: Callable[[], Iterator[ReadableEntry]]

    async def hidden_datakeys(self) -> set[str]:
        plugins = set(_readable_plugins(self.get_readable_formats()))
        hidden = await asyncio.gather(*(_hidden(plugin) for plugin in plugins))
        return set().union(*hidden)


async def _hidden(plugin: NDPluginBaseIO) -> set[str]:
    """All the read signals of a disabled plugin, else those not enabled."""
    enabled, params = await asyncio.gather(
        plugin_is_enabled(plugin), plugin.get_enabled_params()
    )
    read = {signal.name for signal in _read_signals(plugin)}
    if not enabled:
        hidden = read
    elif params is None:
        hidden = set()
    else:
        hidden = read - {param.signal.name for param in params}
    return hidden


class AreaDetector(StandardDetector, Generic[ADBaseIOT]):
    """An areaDetector driver, with its plugins and the data logics that serve it.

    Every plugin of the detector is an attribute and registered as a `CHILD`
    readable, as is each writer a factory makes. The factories are given all the
    plugins, to parse their NDAttributes.

    :param plugins: All the detector's plugins by attribute name, those the
        caller passes and those the detector creates, such as its process plugin.
    """

    def __init__(
        self,
        driver: ADBaseIOT,
        prefix: str | None = None,
        *writer_factories: ADDataFactory,
        acquire_logic: DetectorAcquireLogic | None = None,
        trigger_logic: DetectorTriggerLogic | None = None,
        plugins: Mapping[str, NDPluginBaseIO] | None = None,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ) -> None:
        self.driver = driver
        if plugins is not None:
            for plugin_name, plugin in plugins.items():
                self._set_unique(plugin_name, plugin)
        self._trigger_logic = trigger_logic
        self._acquire_logic = acquire_logic
        plugin_list = list(plugins.values()) if plugins else []
        writers: list[NDPluginBaseIO] = []
        data_logics: list[BoundedDataLogic | UnboundedDataLogic] = []
        if writer_factories:
            if prefix is None:
                raise ValueError("prefix is required when writer_factories are given")
            made = [
                factory.build(prefix, driver, plugin_list)
                for factory in writer_factories
            ]
            named_writers = [named for named, _ in made if named is not None]
            names = [attr for attr, _ in named_writers]
            if len(names) != len(set(names)):
                duplicates = sorted({n for n in names if names.count(n) > 1})
                raise ValueError(
                    f"Duplicate writer_name(s) in writer_factories: {duplicates}"
                )
            for attr, writer in named_writers:
                self._set_unique(attr, writer)
                writers.append(writer)
            data_logics = [data_logic for _, data_logic in made]
        self._data_logics = tuple(data_logics)
        # The driver is always wired up, so register it whole and let ADBaseIO
        # declare which of its signals are configuration. So is each of the
        # plugins and writers, whose own formats declare what they read.
        self.set_readable_format(self.driver, Format.CHILD)
        for plugin in (*plugin_list, *writers):
            self.set_readable_format(plugin, Format.CHILD)
        for signal in config_sigs:
            self.set_readable_format(signal, Format.CONFIG_SIGNAL)
        super().__init__(name=name)

    def _set_unique(self, attr: str, value: NDPluginBaseIO) -> None:
        if attr in self.__dict__ or hasattr(type(self), attr):
            raise ValueError(
                f"Cannot add {type(value).__name__} as attribute '{attr}': "
                f"{type(self).__name__} already has an attribute with that name"
            )
        setattr(self, attr, value)

    @cached_property
    def logic(self) -> DetectorLogic:
        return DetectorLogic(
            trigger=self._trigger_logic,
            acquire=self._acquire_logic,
            data=self._data_logics,
            suppress=_PluginSuppressLogic(self.get_readable_formats),
        )

    def get_plugin(
        self, name: str, plugin_type: type[NDPluginBaseIOT] = NDPluginBaseIO
    ) -> NDPluginBaseIOT:
        """Get a plugin by name and type.

        :param name: Name of the plugin attribute
        :param plugin_type: Expected type of the plugin
        :return: The plugin instance
        :raises AttributeError: If the plugin does not exist
        :raises TypeError: If the plugin is not of the expected type
        """
        plugin = getattr(self, name, None)
        if plugin is None:
            raise AttributeError(f"{self.name} has no plugin named '{name}'")
        elif not isinstance(plugin, plugin_type):
            raise TypeError(
                f"Expected {self.name}.{name} to be a {plugin_type.__name__}, "
                f"got {type(plugin).__name__}"
            )
        return plugin


def plugins_with_proc(
    plugins: Mapping[str, NDPluginBaseIO] | None,
    prefix: str,
    proc_suffix: str | None,
) -> tuple[dict[str, NDPluginBaseIO], NDProcessIO | None]:
    """Add a process plugin made from proc_suffix to plugins.

    :param plugins: The caller's plugins, which must not include "proc"
    :param prefix: The detector's PV prefix
    :param proc_suffix: The process plugin's PV suffix, if it has one
    :returns: The plugins with "proc" added, and the process plugin
    """
    plugins = dict(plugins or {})
    if "proc" in plugins:
        msg = "Pass proc_suffix to add a process plugin, not plugins['proc']"
        raise ValueError(msg)
    proc = None
    if proc_suffix:
        proc = NDProcessIO(prefix + proc_suffix)
        plugins["proc"] = proc
    return plugins, proc


class ContAcqDetector(AreaDetector[ADBaseIOT]):
    """Create an continuously acquiring AreaDetector instance.

    :param prefix: EPICS PV prefix for the detector
    :param writer_factories: Factories for file writer plugins and their data logics
    :param driver_cls: IO class for the driver, defaults to ADBaseIO
    :param driver_suffix: Suffix for the driver PV, defaults to "cam1:"
    :param cb_suffix: Suffix for the circular buffer plugin PV, defaults to "CB1:"
    :param cb_plugin_name: Attribute name of the circular buffer plugin
    :param proc_suffix: Suffix for the NDProcessIO plugin PV, if any, defaults to None
    :param num_frames_to_skip:
        Incoming frames to skip before each CB gate opening, defaults to 0
    :param plugins:
        areaDetector plugins to add, each registered as a readable. "proc" is
        reserved: use proc_suffix instead. So is the name cb_plugin_name.
    :param config_sigs: Additional signals to include in configuration
    :param name: Name for the detector device
    """

    def __init__(
        self,
        prefix: str,
        *writer_factories: ADDataFactory,
        driver_cls: type[ADBaseIOT] = ADBaseIO,
        driver_suffix: str = "cam1:",
        cb_suffix: str = "CB1:",
        cb_plugin_name: str = "cb",
        proc_suffix: str | None = None,
        num_frames_to_skip: int = 0,
        plugins: dict[str, NDPluginBaseIO] | None = None,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ) -> None:
        driver = driver_cls(prefix + driver_suffix)
        cb_plugin = NDCircularBuffIO(prefix + cb_suffix, name=cb_plugin_name)
        plugins, proc = plugins_with_proc(plugins, prefix, proc_suffix)
        if cb_plugin_name in plugins:
            msg = (
                "Pass cb_plugin_name to name the circular buffer plugin, "
                f"not plugins[{cb_plugin_name!r}]"
            )
            raise ValueError(msg)
        super().__init__(
            driver,
            prefix,
            *writer_factories,
            acquire_logic=ADContAcqAcquireLogic(driver, cb_plugin, num_frames_to_skip),
            trigger_logic=ADContAcqTriggerLogic(driver, cb_plugin, proc),
            plugins={**plugins, cb_plugin_name: cb_plugin},
            config_sigs=config_sigs,
            name=name,
        )
