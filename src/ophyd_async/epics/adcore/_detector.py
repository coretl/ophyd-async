import asyncio
import contextlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Generic

from ophyd_async.core import (
    BoundedDataLogic,
    DetectorAcquireLogic,
    DetectorLogic,
    DetectorTriggerLogic,
    Device,
    SignalR,
    StandardDetector,
    UnboundedDataLogic,
)
from ophyd_async.core import StandardReadableFormat as Format

from ._acquire_logic import ADContAcqAcquireLogic
from ._data_logic import ADDataFactory
from ._io import (
    ADBaseIO,
    ADBaseIOT,
    NDCircularBuffIO,
    NDPluginBaseIO,
    NDPluginBaseIOT,
    NDProcessIO,
)
from ._ndattribute import NDAttributeSource
from ._trigger_logic import ADContAcqTriggerLogic


def _descendants(device: Device) -> Iterator[Device]:
    """The device, then each of its children and theirs."""
    yield device
    for _, child in device.children():
        yield from _descendants(child)


def _plugins_held_by(logic: object | None) -> list[NDPluginBaseIO]:
    """The plugins a logic holds, which it switches on itself when it prepares."""
    held = vars(logic).values() if logic is not None else ()
    return [value for value in held if isinstance(value, NDPluginBaseIO)]


@dataclass(frozen=True, kw_only=True, eq=False)
class _ADDetectorLogic(DetectorLogic):
    """A `DetectorLogic` that holds its plugins' recording from stage to unstage.

    :param plugins: the plugins to record at stage, check at each prepare and
        forget at unstage. The recording is taken once, so a plugin the
        detector's own logics switch on at prepare is not one of these.
    """

    plugins: Sequence[NDPluginBaseIO] = ()

    async def ensure_ready(self) -> None:
        # After the data logics, which may switch their plugins on
        await super().ensure_ready()
        await asyncio.gather(*(plugin.snapshot() for plugin in self.plugins))

    # Parameter and return types are those of DetectorLogic.prepare_to_fly,
    # whose context type is not public
    async def prepare_to_fly(self, name, value, previous_ctx):
        await self._raise_if_recording_changed(name)
        steps = super().prepare_to_fly(name, value, previous_ctx)
        async with contextlib.aclosing(steps):
            async for step in steps:
                yield step

    async def _raise_if_recording_changed(self, name: str) -> None:
        changed_per_plugin = await asyncio.gather(
            *(plugin.get_changed_signals() for plugin in self.plugins)
        )
        sources = sorted(
            signal.source for changed in changed_per_plugin for signal in changed
        )
        if sources:
            raise RuntimeError(
                f"{name}: plugin switches changed since stage(): {sources}. "
                "unstage() and stage() to use the new values."
            )

    async def ensure_stopped(self) -> None:
        await super().ensure_stopped()
        for plugin in self.plugins:
            plugin.forget_snapshot()


class AreaDetector(StandardDetector, Generic[ADBaseIOT]):
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
            names = [f.writer_name for f in writer_factories if f.writer_cls]
            if len(names) != len(set(names)):
                duplicates = sorted({n for n in names if names.count(n) > 1})
                raise ValueError(
                    f"Duplicate writer_name(s) in writer_factories: {duplicates}"
                )
            for factory in writer_factories:
                writer, data_logic = factory(prefix, driver, plugin_list)
                if writer is not None:
                    self._set_unique(factory.writer_name, writer)
                    writers.append(writer)
                data_logics.append(data_logic)
        self._data_logics = tuple(data_logics)
        tree = list(_descendants(self))
        referenced = self._check_referenced_plugins(writer_factories, tree)
        sources = self._check_ndattribute_sources(writer_factories, tree)
        # The plugins the trigger logic holds are switched on by it at prepare, so
        # they are not the user's switches to record
        held = _plugins_held_by(trigger_logic)
        self._used_plugins = tuple(
            plugin
            for plugin in self._find_plugins(
                [*plugin_list, *writers, *sources, *referenced]
            )
            if not any(plugin is owned for owned in held)
        )
        # The driver is always wired up, so register it whole and let ADBaseIO
        # declare which of its signals are configuration. So is each of the
        # plugins, whose own formats declare what they read.
        self.set_readable_format(self.driver, Format.CHILD)
        for plugin in plugin_list:
            self.set_readable_format(plugin, Format.CHILD)
        for signal in config_sigs:
            self.set_readable_format(signal, Format.CONFIG_SIGNAL)
        super().__init__(name=name)
        self._suppressed_datakeys_funcs += (self.__on_suppressed_datakeys,)

    def __on_suppressed_datakeys(self) -> Mapping[str, None]:
        """The names of the readables the plugins' switches turned off at stage."""
        return {
            signal.name: None
            for plugin in self._used_plugins
            for signal in plugin.get_switched_off_readables()
        }

    def _check_referenced_plugins(
        self, writer_factories: Sequence[ADDataFactory], tree: Sequence[Device]
    ) -> list[NDPluginBaseIO]:
        """The plugins the factories take by reference, raising if one is missing."""
        referenced = [f.plugin for f in writer_factories if f.plugin is not None]
        for plugin in referenced:
            if not any(plugin is device for device in tree):
                raise ValueError(
                    f"{plugin!r} is used by a data factory but is not in the "
                    "device tree of the detector: pass it, or the plugin it "
                    "is a child of, in plugins="
                )
        return referenced

    def _check_ndattribute_sources(
        self, writer_factories: Sequence[ADDataFactory], tree: Sequence[Device]
    ) -> list[NDAttributeSource]:
        """The NDAttribute sources of the factories, raising if one is misplaced."""
        sources: list[NDAttributeSource] = []
        for factory in writer_factories:
            for source in factory.ndattribute_sources:
                if any(source is listed for listed in sources):
                    raise ValueError(
                        f"{source!r} is listed more than once in ndattribute_sources"
                    )
                if not any(source is device for device in tree):
                    raise ValueError(
                        f"{source!r} is in ndattribute_sources but is not in the "
                        "device tree of the detector: pass it, or the plugin it "
                        "is a child of, in plugins="
                    )
                sources.append(source)
        return sources

    def _find_plugins(self, roots: Sequence[object]) -> tuple[NDPluginBaseIO, ...]:
        """Each plugin that is a root or a descendant of one, once."""
        found = {
            device: None
            for root in roots
            if isinstance(root, Device)
            for device in _descendants(root)
            if isinstance(device, NDPluginBaseIO)
        }
        return tuple(found)

    def _set_unique(self, attr: str, value: NDPluginBaseIO) -> None:
        if attr in self.__dict__ or hasattr(type(self), attr):
            raise ValueError(
                f"Cannot add {type(value).__name__} as attribute '{attr}': "
                f"{type(self).__name__} already has an attribute with that name"
            )
        setattr(self, attr, value)

    @cached_property
    def logic(self) -> DetectorLogic:
        return _ADDetectorLogic(
            trigger=self._trigger_logic,
            acquire=self._acquire_logic,
            data=self._data_logics,
            plugins=self._used_plugins,
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


class ContAcqDetector(AreaDetector[ADBaseIOT]):
    """Create an continuously acquiring AreaDetector instance.

    :param prefix: EPICS PV prefix for the detector
    :param writer_factories: Factories for file writer plugins and their data logics
    :param driver_cls: IO class for the driver, defaults to ADBaseIO
    :param driver_suffix: Suffix for the driver PV, defaults to "cam1:"
    :param cb_suffix: Suffix for the circular buffer plugin PV, defaults to "CB1:"
    :param cb_plugin_name: Attribute name of the circular buffer plugin
    :param proc_suffix: Suffix for the NDProcessIO plugin PV, if any, defaults to None
    :param plugins: Additional areaDetector plugins to include
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
        plugins: dict[str, NDPluginBaseIO] | None = None,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ) -> None:
        driver = driver_cls(prefix + driver_suffix)
        cb_plugin = NDCircularBuffIO(prefix + cb_suffix, name=cb_plugin_name)
        proc_plugin = NDProcessIO(prefix + proc_suffix) if proc_suffix else None
        super().__init__(
            driver,
            prefix,
            *writer_factories,
            acquire_logic=ADContAcqAcquireLogic(driver, cb_plugin),
            trigger_logic=ADContAcqTriggerLogic(driver, cb_plugin, proc_plugin),
            plugins=(plugins or {})
            | {cb_plugin.name: cb_plugin}
            | ({"proc": proc_plugin} if proc_plugin else {}),
            config_sigs=config_sigs,
            name=name,
        )
