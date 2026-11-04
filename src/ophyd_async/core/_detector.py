"""Module which defines abstract classes to work with detectors."""

import asyncio
import os
import time
import warnings
from abc import abstractmethod
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Collection,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Any, Generic, TypeVar, cast

from bluesky.protocols import (
    Collectable,
    HasHints,
    Hints,
    Reading,
    Stageable,
    Status,
    StreamAsset,
    Triggerable,
)
from event_model import DataKey
from event_model.documents import PartialEventPage
from pydantic import Field, PositiveInt, computed_field

from ._data_providers import (
    DataProvider,
    PageableDataProvider,
    StreamableDataProvider,
)
from ._device import Device, name_added_to_errors
from ._flyable import KICKED_OFF, FlyableLogic, StandardFlyable
from ._readable import StandardReadable, StandardReadableFormat
from ._settings import Settings
from ._signal import SignalDict, SignalR, SignalRW, observe_signals_value
from ._standard_base import _State, _StateMachine
from ._status import WatchableAsyncStatus
from ._utils import (
    DEFAULT_TIMEOUT,
    ConfinedModel,
    WatcherUpdate,
    abstract_cached_property,
    error_if_none,
    gather_dict,
    get_unique,
    merge_dicts,
    merge_gathered_dicts,
)


class DetectorTrigger(Enum):
    """Type of mechanism for triggering a detector to take exposures."""

    INTERNAL = "INTERNAL"
    """On arm generate internally timed exposures"""

    EXTERNAL_EDGE = "EXTERNAL_EDGE"
    """On every (normally rising) edge of an external input generate an internally
    timed exposure"""

    EXTERNAL_LEVEL = "EXTERNAL_LEVEL"
    """On a rising edge of an external input start an exposure, ending on the falling
    edge"""


class TriggerInfo(ConfinedModel):
    """Information required to setup `trigger` or `kickoff` on a `StandardDetector`."""

    trigger: DetectorTrigger = Field(default=DetectorTrigger.INTERNAL)
    """What sort of triggering should the detector be set for."""

    livetime: float = Field(default=0.0, ge=0.0)
    """For INTERNAL or EXTERNAL_EDGE triggering, how long should each exposure be.
    0 means whatever is currently set."""

    deadtime: float = Field(default=0.0, ge=0.0)
    """For INTERNAL triggering, how long should be left between each exposure.
    0 means use the minimum the detector supports."""

    exposures_per_collection: PositiveInt = Field(default=1)
    """An exposure corresponds to a single trigger sent to the detector.
    If many exposures are averaged together on the detector or in a processing
    chain to make a single collection that is exposed to bluesky as data then
    this number should be set to the number of exposures to be processed into a
    single collection."""

    collections_per_event: PositiveInt = Field(default=1)
    """A collection is exposed to bluesky as data, but different detectors can
    be set to have a different number of collections per event so that multiple
    collections from a faster detector can be zipped with a single collection
    from a slower detector. E.g. if number_of_events=10 and
    collections_per_event=5 then the detector will take 50 exposures, but
    publish 10 StreamDatum indices, and describe() will show a shape of (5, h,
    w) for each.
    """

    number_of_events: PositiveInt = Field(default=1)
    """Number of bluesky events that will be emitted."""

    exposure_timeout: float = Field(
        default_factory=lambda d: d["livetime"] + d["deadtime"] + DEFAULT_TIMEOUT,
        gt=0,
    )
    """The longest wait for progress before giving up: the most time allowed
    between one collection being written and the next, not a total for the
    scan."""

    flush_period: float = Field(default=0.0, ge=0.0)
    """How many seconds between flushes of the data being written.

    A data logic that buffers before it writes sizes that buffer from this and
    the exposure period, so data appears at roughly this rate: an HDF chunk of
    `flush_period / period` frames, say. Set it to the rate the plan collects at
    so a fly scan sees each batch as it lands. 0 means leave whatever the
    hardware is set to."""

    @computed_field
    @cached_property
    def number_of_collections(self) -> int:
        return self.number_of_events * self.collections_per_event

    @computed_field
    @cached_property
    def number_of_exposures(self) -> int:
        return self.number_of_collections * self.exposures_per_collection


class DetectorTriggerLogic:
    """Logic for configuring detector triggering modes.

    This class defines the interface for detector trigger configuration, handling
    both internal and external triggering modes. Implementations should provide
    detector-specific logic for preparing the detector to operate in different
    trigger modes and manage exposure parameters.

    The class manages:
    - Configuration signals that should appear in detector metadata
    - Deadtime calculations based on detector configuration
    - Preparation for internal (self-triggered) exposures
    - Preparation for external edge-triggered exposures
    - Preparation for external level-triggered exposures
    - Multi-exposure collection batching

    Subclasses must implement the appropriate `prepare_*` method for any trigger
    mode the detector supports, and `get_deadtime` if it supports external
    triggering.
    """

    def get_deadtime(self, config_values: SignalDict) -> float:
        """Return the deadtime in seconds for the detector.

        :param config_values:
            The value of every `Signal` the detector reports as configuration,
            i.e. everything registered with
            [](#StandardReadableFormat.CONFIG_SIGNAL), including those declared
            on children registered as [](#StandardReadableFormat.CHILD). A
            logic that needs a signal must make sure it is declared as
            configuration, rather than nominating it separately: the values
            that determine deadtime are exactly the ones that ought to be
            recorded for the scan anyway.
        """
        raise NotImplementedError(self)

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        """Prepare the detector to take internally triggered exposures.

        :param num: the number of exposures to take
        :param livetime: how long the exposure should be, 0 means what is currently set
        :param deadtime: how long between exposures, 0 means the shortest possible
        """
        raise NotImplementedError(self)

    async def prepare_edge(self, num: int, livetime: float):
        """Prepare the detector to take external edge triggered exposures.

        :param num: the number of exposures to take
        :param livetime: how long the exposure should be, 0 means what is currently set
        """
        raise NotImplementedError(self)

    async def prepare_level(self, num: int):
        """Prepare the detector to take external level triggered exposures.

        :param num: the number of exposures to take
        """
        raise NotImplementedError(self)

    async def prepare_exposures_per_collection(self, exposures_per_collection: int):
        """Prepare processing of multiple exposures into a single collection.

        :param exposures_per_collection:
            number of exposures to process into each collection
        """
        raise NotImplementedError(self)

    async def default_trigger_info(self) -> TriggerInfo:
        """Fallback for the default TriggerInfo in plans without prepare."""
        raise NotImplementedError(self)


def _trigger_logic_supported(method) -> bool:
    # If the function that is bound in a subclass is the same as the function
    # attached to DetectorTriggerLogic, then the subclass has not overridden it,
    # so this method is not supported by the subclass.
    return method.__func__ is not getattr(DetectorTriggerLogic, method.__name__)


def _get_supported_triggers(
    trigger_logic: DetectorTriggerLogic,
) -> set[DetectorTrigger]:
    supported_triggers = set()
    if _trigger_logic_supported(trigger_logic.prepare_internal):
        supported_triggers.add(DetectorTrigger.INTERNAL)
    if _trigger_logic_supported(trigger_logic.prepare_edge):
        supported_triggers.add(DetectorTrigger.EXTERNAL_EDGE)
    if _trigger_logic_supported(trigger_logic.prepare_level):
        supported_triggers.add(DetectorTrigger.EXTERNAL_LEVEL)
    return supported_triggers


class DetectorAcquireLogic:
    """Base class for detector acquisition lifecycle hooks.

    Subclasses implement these hooks, called at defined points in the
    [](#StandardDetector) lifecycle:

    - `ensure_ready` — called from `stage()` to put the detector into a known
      idle state before a scan begins. Defaults to `ensure_stopped`.
    - `acquire` — an async generator for one acquisition: start the detector
      acquiring, yield once it is armed, then wait for it to return to idle.
    - `ensure_stopped` — called from `unstage()` to stop the detector and
      perform any end-of-scan cleanup.
    """

    async def ensure_ready(self):
        """Ensure the detector is idle before a scan.

        Called from `stage()`. Defaults to `ensure_stopped`; override it when
        stage needs a different action, such as arming once for many
        kickoff/complete cycles.
        """
        await self.ensure_stopped()

    @abstractmethod
    def acquire(self) -> AsyncGenerator[None, None]:
        """Start acquiring, yield once armed, then wait until idle.

        Write it as an async generator that yields once. It is entered in
        `prepare()` for external triggering, and in `kickoff()` or `trigger()`
        for internal triggering, then run to the end once the data is written.
        Acquisition must stop by itself once the data is written: a free-running
        detector is not supported.
        If the scan is cancelled it is closed at its `yield`, so own whatever it
        starts: hold a status with `async with`, or cancel a task in `finally`.
        For example:

        ```python
        async def acquire(self):
            async with self.driver.acquire.set(True) as status:
                await wait_for_value(self.driver.armed, True, timeout=10)
                yield  # armed
                await status
        ```
        """

    @abstractmethod
    async def ensure_stopped(self):
        """Stop the detector and perform end-of-scan cleanup.

        Called from `unstage()`.
        """


#: The provider a data logic returns from `start_provider`
ProviderT = TypeVar("ProviderT", bound=DataProvider)


class DetectorDataLogic(Generic[ProviderT]):
    """Base class for detector data logic: do not inherit from it directly.

    Inherit from [](#BoundedDataLogic) or [](#UnboundedDataLogic), which say
    whether the data source needs to know how many collections to take.

    At the first `prepare()` after `stage()` the detector starts every
    unbounded data logic, then every bounded one, handing each bounded logic the
    datakeys the unbounded providers already make. The logics that return a
    provider serve the stage, and are started again at each later `prepare()`.
    The provider describes and publishes what is being written.

    Whether the data is published as stream assets or event pages is the type
    of the provider that is returned, not a declaration on the logic, so one
    logic may produce either depending on the scan.

    A source that produces a single value per event, like a plugin scalar, needs
    no data logic at all: register its signal with
    [](#StandardReadable.set_readable_format) instead.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "stop" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} defines stop, which DetectorDataLogic no longer "
                "calls: stop -> ensure_stopped"
            )

    #: Counts the collections written so far. `start_provider` resets it, and
    #: waits until it reads 0 before returning
    collections_written: SignalR[int]

    async def ensure_ready(self) -> None:
        """Ensure the data logic is idle before a scan.

        Called from `stage()`. Defaults to `ensure_stopped`.
        """
        await self.ensure_stopped()

    def get_hinted_fields(self, name: str) -> Sequence[str]:
        """Return the datakeys to hint, of those this logic may produce.

        The detector hints only those its providers actually make.

        :param name: the detector name, which prefixes every datakey
        """
        return []

    async def ensure_stopped(self) -> None:
        """Stop taking data."""
        return None


class BoundedDataLogic(DetectorDataLogic[ProviderT]):
    """A data logic whose source must be told how many collections to take.

    For a finite buffer, like an areaDetector stats time series. It is started
    after the unbounded logics, so it can leave out datakeys they already make.
    At each step point after the first the detector stops it and starts a new
    provider, so a buffer holding the last point's data is cleared.
    """

    @abstractmethod
    async def start_provider(
        self,
        name: str,
        existing_datakeys: Sequence[str],
        num_collections: int,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> ProviderT | None:
        """Start writing and return a new provider describing what is written.

        Return a [](#StreamableDataProvider) to publish stream assets, or a
        [](#PageableDataProvider) to publish event pages, made for
        `collections_per_event`. The provider must make none of
        `existing_datakeys`. Return `None`, without writing, if there is nothing
        left to make: this is not an error and is not warned about.

        :param name: the detector name, which prefixes every datakey
        :param existing_datakeys: the datakeys the unbounded providers make
        :param num_collections: how many collections the scan takes
        :param collections_per_event: how many collections make up one event
        :param collection_period: seconds per collection,
            `exposures_per_collection * (livetime + deadtime)`, for sizing
            chunks (0 means "use whatever is currently set on the hardware")
        :param flush_period: how often to flush
        """
        raise NotImplementedError(self)


class UnboundedDataLogic(DetectorDataLogic[ProviderT]):
    """A data logic whose source takes data for as long as it is told to.

    For a file writer, or anything else that does not need to know how many
    collections to take. Its provider carries over to the next `prepare()` of
    the same stage if that is for the same collections per event, collection period
    and flush period, so it may already have written collections when a scan
    starts.
    """

    @abstractmethod
    async def start_provider(
        self,
        name: str,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> ProviderT | None:
        """Start writing and return a new provider describing what is written.

        Return a [](#StreamableDataProvider) to publish stream assets, or a
        [](#PageableDataProvider) to publish event pages, made for
        `collections_per_event`. Return `None`, without writing, to sit out the
        scans of this stage, for instance a plugin that is switched off: this is
        not an error and is not warned about.

        :param name: the detector name, which prefixes every datakey
        :param collections_per_event: how many collections make up one event
        :param collection_period: seconds per collection,
            `exposures_per_collection * (livetime + deadtime)`, for sizing
            chunks (0 means "use whatever is currently set on the hardware")
        :param flush_period: how often to flush
        """
        raise NotImplementedError(self)


def _merge_pages(
    pages: Sequence[tuple[PageableDataProvider, PartialEventPage | None]],
) -> PartialEventPage | None:
    """Merge one page per pageable provider into a single page.

    `None` if every provider returned `None`. Raises if only some did, if the
    pages hold different numbers of events, or if a key appears twice.
    """
    if all(page is None for _, page in pages):
        return None
    # A None page has length None, so a mix of None and real pages differs too
    lengths = [
        (type(p).__name__, len(page["time"]) if page else None) for p, page in pages
    ]
    get_unique(lengths, "numbers of events")
    made = [page for _, page in pages if page]
    # Each event happened when the last of its providers' data arrived
    times = zip(*(page["time"] for page in made), strict=True)
    return {
        "data": merge_dicts(page["data"] for page in made),
        "timestamps": merge_dicts(page["timestamps"] for page in made),
        "time": [max(t) for t in times],
    }


@dataclass
class DetectorCtx:
    """What `prepare()` set up, held until `unstage()`."""

    #: The plan's TriggerInfo
    trigger_info: TriggerInfo
    #: The detector name, which prefixes every datakey
    name: str
    #: Seconds per collection: `exposures_per_collection` times the livetime
    #: plus deadtime, or what the hardware has set when the livetime is 0 and
    #: its state is preserved
    collection_period: float
    #: The bounded data logics of this stage: those that returned a provider
    #: when last started
    bounded_logics: Sequence[BoundedDataLogic[Any]] = ()
    #: The unbounded data logics of this stage: those that returned a provider
    #: when last started
    unbounded_logics: Sequence[UnboundedDataLogic[Any]] = ()
    #: The providers `bounded_logics` started for this prepare, one per logic.
    #: Restarted at each step point after the first
    bounded_providers: Sequence[DataProvider] = ()
    #: The providers `unbounded_logics` started, one per logic. Carried over from
    #: the previous prepare if made for the same collections per event,
    #: collection period and flush period
    unbounded_providers: Sequence[DataProvider] = ()
    #: The datakeys of every provider, the same at every prepare of the stage
    datakeys: dict[str, DataKey] = field(default_factory=dict)
    #: The collections every unbounded data logic had written at the end of
    #: prepare()
    unbounded_start_collections: int = 0
    #: How many trigger()s this prepare has served
    triggers: int = 0
    #: The readings of `DetectorLogic.get_readings` at the current step point,
    #: `None` until read
    readings: dict[str, Reading] | None = None
    #: The acquire logic's `acquire()`, run to its yield by prepare() for
    #: external triggering
    external_acquire_gen: AsyncGenerator[None, None] | None = None

    @property
    def providers(self) -> list[DataProvider]:
        """Every serving provider, the unbounded ones first."""
        return [*self.unbounded_providers, *self.bounded_providers]


async def _datakeys(providers: Sequence[DataProvider]) -> dict[str, DataKey]:
    """The datakeys of every provider, raising on one made by more than one."""
    return await merge_gathered_dicts(p.make_datakeys() for p in providers)


async def _stop_logics(logics: Sequence[DetectorDataLogic[Any]]) -> None:
    """Stop every logic in `logics`."""
    await asyncio.gather(*(dl.ensure_stopped() for dl in logics))


async def _start_unbounded(ctx: DetectorCtx) -> None:
    """Start `ctx.unbounded_logics`, keeping those that return a provider."""
    info = ctx.trigger_info
    providers = await asyncio.gather(
        *(
            dl.start_provider(
                ctx.name,
                info.collections_per_event,
                ctx.collection_period,
                info.flush_period,
            )
            for dl in ctx.unbounded_logics
        )
    )
    started = list(zip(ctx.unbounded_logics, providers, strict=True))
    ctx.unbounded_logics = [dl for dl, p in started if p]
    ctx.unbounded_providers = [p for _, p in started if p]


async def _start_bounded(ctx: DetectorCtx) -> None:
    """Start `ctx.bounded_logics`, keeping those that return a provider.

    Their providers leave out the datakeys the unbounded providers make.
    """
    info = ctx.trigger_info
    existing_datakeys = list(await _datakeys(ctx.unbounded_providers))
    providers = await asyncio.gather(
        *(
            dl.start_provider(
                ctx.name,
                existing_datakeys,
                info.number_of_collections,
                info.collections_per_event,
                ctx.collection_period,
                info.flush_period,
            )
            for dl in ctx.bounded_logics
        )
    )
    started = list(zip(ctx.bounded_logics, providers, strict=True))
    ctx.bounded_logics = [dl for dl, p in started if p]
    ctx.bounded_providers = [p for _, p in started if p]


async def _restart_bounded(ctx: DetectorCtx, when: str) -> None:
    """Stop and start `ctx.bounded_logics`, then check the datakeys are unchanged.

    :param when: when this happens, for the error message
    """
    await _stop_logics(ctx.bounded_logics)
    await _start_bounded(ctx)
    before = ctx.datakeys
    ctx.datakeys = await _datakeys(ctx.providers)
    _check_datakeys(ctx, before, when)


def _check_datakeys(ctx: DetectorCtx, before: Collection[str], when: str) -> None:
    """Raise unless the serving providers make the datakeys `before`."""
    if missing_or_extra := sorted(set(before) ^ ctx.datakeys.keys()):
        raise RuntimeError(
            f"The data logics made {sorted(ctx.datakeys)} at {when}, but "
            f"{sorted(before)} before it, which differ in {missing_or_extra}: "
            "a data logic that serves a stage must make the same datakeys for "
            "all of it, so unstage() and stage() to change what it makes"
        )


def _made_for(ctx: DetectorCtx) -> tuple[str, int, float, float]:
    """What an unbounded provider was made for: it carries over if unchanged."""
    info = ctx.trigger_info
    return (
        ctx.name,
        info.collections_per_event,
        ctx.collection_period,
        info.flush_period,
    )


def _bounded_origin(ctx: DetectorCtx) -> int:
    """The events written before the bounded providers were last started.

    The step points before the current one, as each point restarts them, and 0
    in a fly scan.
    """
    return max(ctx.triggers - 1, 0)


def _collections_offset(ctx: DetectorCtx) -> dict[SignalR[int], int]:
    """The offset from each data logic's `collections_written` to the scan count.

    Added to `collections_written` it gives the collections since `prepare()`:
    positive for a bounded data logic, which only counts from the current step
    point, and negative for an unbounded one, which counts from before
    `prepare()`.
    """
    cpe = ctx.trigger_info.collections_per_event
    bounded = _bounded_origin(ctx) * cpe
    unbounded = -ctx.unbounded_start_collections
    return {
        **{dl.collections_written: bounded for dl in ctx.bounded_logics},
        **{dl.collections_written: unbounded for dl in ctx.unbounded_logics},
    }


def _provider_ends(ctx: DetectorCtx, index: int) -> Iterator[tuple[DataProvider, int]]:
    """Each serving provider, with event `index` counted from its origin."""
    for provider in ctx.bounded_providers:
        yield provider, index - _bounded_origin(ctx)
    for provider in ctx.unbounded_providers:
        yield provider, index


async def _no_acquisition() -> AsyncGenerator[None, None]:
    """The acquisition of a detector with no acquire logic."""
    yield


def _preserve_detector_state() -> bool:
    # Opt-in: set OPHYD_ASYNC_PRESERVE_DETECTOR_STATE=YES to have the detector
    # read back current hardware state via default_trigger_info(). See ADR 0013.
    # TODO: flip default to YES and remove this guard in a future PR once
    # downstream code has had time to implement default_trigger_info().
    return os.environ.get("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "NO").upper() == "YES"


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectorLogic(FlyableLogic[TriggerInfo, DetectorCtx]):
    """Drive a detector through its trigger, acquire and data logics.

    The fields are configuration and never change; what a scan sets up is held
    in the context, so `dataclasses.replace` makes an independent copy.

    :param trigger: sets up exposures; without one only internal triggering
        with the hardware's current timing is supported
    :param acquire: starts and stops acquisition
    :param data: the data logics, each publishing one part of the data
    """

    trigger: DetectorTriggerLogic | None = None
    acquire: DetectorAcquireLogic | None = None
    data: Sequence[BoundedDataLogic[Any] | UnboundedDataLogic[Any]] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", tuple(self.data))
        for dl in self.data:
            if isinstance(dl, BoundedDataLogic) == isinstance(dl, UnboundedDataLogic):
                raise TypeError(
                    f"{dl!r} must inherit from exactly one of BoundedDataLogic and "
                    "UnboundedDataLogic"
                )

    @cached_property
    def supported_triggers(self) -> set[DetectorTrigger]:
        """The trigger types the trigger logic implements."""
        if self.trigger is None:
            return {DetectorTrigger.INTERNAL}
        return _get_supported_triggers(self.trigger)

    @cached_property
    def supports_deadtime(self) -> bool:
        """Whether the trigger logic can calculate a deadtime."""
        return self.trigger is not None and _trigger_logic_supported(
            self.trigger.get_deadtime
        )

    def with_device(self, name: str = "") -> "StandardDetector":
        """Wrap this logic in an ephemeral `StandardDetector` for use in a plan."""
        return _LogicDetector(self, name=name)

    def get_deadtime(self, config_values: SignalDict) -> float:
        """Return the deadtime the trigger logic calculates from `config_values`."""
        trigger = error_if_none(self.trigger, "No trigger logic")
        return trigger.get_deadtime(config_values)

    async def default_trigger_info(self) -> TriggerInfo:
        """The TriggerInfo to prepare with when a plan calls trigger() directly.

        The trigger logic's `default_trigger_info()` if
        [](#OPHYD_ASYNC_PRESERVE_DETECTOR_STATE) is `YES`, otherwise a bare
        [`TriggerInfo()`](#TriggerInfo).
        """
        info = TriggerInfo()
        if _preserve_detector_state() and self.trigger is not None:
            if not _trigger_logic_supported(self.trigger.default_trigger_info):
                raise RuntimeError(
                    "OPHYD_ASYNC_PRESERVE_DETECTOR_STATE=YES is set but the "
                    "trigger logic has no default_trigger_info(): implement it on "
                    f"{type(self.trigger).__name__} or unset the environment "
                    "variable"
                )
            info = await self.trigger.default_trigger_info()
        return info

    async def ensure_ready(self) -> None:
        """Make every data logic and the detector ready, in parallel."""
        await asyncio.gather(
            *(dl.ensure_ready() for dl in self.data),
            *([self.acquire.ensure_ready()] if self.acquire else []),
        )

    async def prepare_to_fly(
        self, name: str, value: TriggerInfo, previous_ctx: DetectorCtx | None
    ) -> AsyncGenerator[DetectorCtx | WatcherUpdate, None]:
        """Set the trigger logic up, yield the context, then start the data.

        The context holds `value` and the collection period, which is read back
        from the trigger logic when `value` leaves the livetime at 0. An
        externally triggered detector is armed last, after the data logics are
        started so their buffers are sized.
        """
        await self._prepare_trigger_logic(value)
        # After preparing the trigger logic, as the hardware then holds the
        # values read back
        collection_period = await self._resolve_collection_period(value)
        ctx = DetectorCtx(
            trigger_info=value, name=name, collection_period=collection_period
        )
        yield ctx
        await self._start_providers(ctx, previous_ctx)
        if value.trigger is not DetectorTrigger.INTERNAL:
            ctx.external_acquire_gen = await self._arm()

    async def _start_providers(
        self, ctx: DetectorCtx, previous_ctx: DetectorCtx | None
    ) -> None:
        """Start the data providers for this scan.

        At the first `prepare()` of a stage every unbounded data logic is
        started, then every bounded one, leaving out the datakeys the unbounded
        providers make. The logics that return a provider serve the rest of the
        stage. At a later `prepare()` the unbounded providers of `previous_ctx`
        carry over if they were made for the same name, collections per event,
        collection period and flush period, and are restarted otherwise; the
        bounded ones are always restarted. Every unbounded provider then starts
        new stream resources at `ctx.unbounded_start_collections`, so each run
        has its own.
        """
        if previous_ctx is None:
            ctx.unbounded_logics = [
                dl for dl in self.data if isinstance(dl, UnboundedDataLogic)
            ]
            ctx.bounded_logics = [
                dl for dl in self.data if isinstance(dl, BoundedDataLogic)
            ]
            await _start_unbounded(ctx)
            await _start_bounded(ctx)
            ctx.datakeys = await _datakeys(ctx.providers)
        else:
            ctx.unbounded_logics = previous_ctx.unbounded_logics
            ctx.bounded_logics = previous_ctx.bounded_logics
            ctx.datakeys = previous_ctx.datakeys
            if _made_for(previous_ctx) == _made_for(ctx):
                ctx.unbounded_providers = previous_ctx.unbounded_providers
            else:
                await _stop_logics(ctx.unbounded_logics)
                await _start_unbounded(ctx)
            await _restart_bounded(ctx, "this prepare()")
        # TODO: delete this check if the run bundler learns to collect both from
        # one device (coretl/bluesky#1), with the prepare() override. See ADR 0024.
        if all(self.produces(ctx)):
            pageable = await _datakeys(
                [p for p in ctx.providers if isinstance(p, PageableDataProvider)]
            )
            raise TypeError(
                f"The detector would produce {sorted(pageable)} as event pages and "
                "the rest of its data as stream assets; bluesky collects only one "
                "of them from a detector, so its stream assets must make these "
                "datakeys too, or a data logic must not serve this scan"
            )
        # Every unbounded data logic has written the same: they all started
        # together, after stage() stopped them all
        written = await asyncio.gather(
            *(dl.collections_written.get_value() for dl in ctx.unbounded_logics)
        )
        counts = [
            (type(dl).__name__, n)
            for dl, n in zip(ctx.unbounded_logics, written, strict=True)
        ]
        ctx.unbounded_start_collections = (
            get_unique(counts, "numbers of collections written") if counts else 0
        )
        for provider in ctx.unbounded_providers:
            provider.start_new_resources(ctx.unbounded_start_collections)

    async def fly(
        self, ctx: DetectorCtx
    ) -> AsyncGenerator[KICKED_OFF | WatcherUpdate, None]:
        """Start the fly scan, then report events written until it finishes."""
        info = ctx.trigger_info
        events = info.number_of_events
        async with self._acquiring(ctx):
            yield KICKED_OFF
            async for update in self._wait_for(
                ctx, 0, events, divisor=info.collections_per_event
            ):
                yield update

    async def step(self, ctx: DetectorCtx) -> AsyncGenerator[WatcherUpdate, None]:
        """Take one step point's exposures and wait for them to be written.

        At each point after the first the bounded providers are restarted, so a
        buffer holding the last point's data is cleared. Unbounded providers
        keep counting.
        """
        info = ctx.trigger_info
        if info.number_of_events != 1:
            raise ValueError(
                "trigger() is not supported for multiple events, the detector was "
                f"prepared with number_of_events={info.number_of_events}: prepare "
                "with number_of_events=1 for a step scan"
            )
        if ctx.triggers and info.trigger is not DetectorTrigger.INTERNAL:
            raise RuntimeError(
                "An externally triggered detector must be prepared again before "
                "each trigger()"
            )
        # prepare() started the bounded providers for the first point
        if ctx.triggers:
            await _restart_bounded(ctx, f"step point {ctx.triggers + 1}")
        ctx.triggers += 1
        ctx.readings = None
        async with self._acquiring(ctx):
            async for update in self._wait_for(
                ctx, ctx.triggers - 1, ctx.triggers, divisor=1
            ):
                yield update

    async def ensure_stopped(self) -> None:
        """Stop every data logic and the detector."""
        await asyncio.gather(
            *(dl.ensure_stopped() for dl in self.data),
            *([self.acquire.ensure_stopped()] if self.acquire else []),
        )

    def get_hinted_fields(self, name: str, ctx: DetectorCtx | None = None) -> list[str]:
        """The hinted fields of the data logics, each once.

        :param name: the detector name, which prefixes every datakey, used if
            `ctx` is `None`
        :param ctx: the prepared context, whose serving logics are asked for
            `ctx.name` and whose datakeys filter their fields; every data logic
            is asked if `None`
        """
        if ctx is None:
            return [field for dl in self.data for field in dl.get_hinted_fields(name)]
        fields = (
            field
            for dl in (*ctx.unbounded_logics, *ctx.bounded_logics)
            for field in dl.get_hinted_fields(ctx.name)
            if field in ctx.datakeys
        )
        # A bounded logic may hint a datakey an unbounded provider makes
        return list(dict.fromkeys(fields))

    def produces(self, ctx: DetectorCtx) -> tuple[bool, bool]:
        """Whether the serving providers make (stream assets, event pages)."""
        return (
            any(isinstance(p, StreamableDataProvider) for p in ctx.providers),
            any(isinstance(p, PageableDataProvider) for p in ctx.providers),
        )

    async def get_readings(self, ctx: DetectorCtx) -> dict[str, Reading]:
        """Read the current step point's event from each pageable provider.

        Makes the page up to the end of the point and reads its last event. The
        readings are kept in `ctx`, so a repeat read at the same point returns
        them again.
        """
        if ctx.readings is None:
            page = await self._make_page(ctx, ctx.triggers)
            ctx.readings = (
                {
                    key: Reading(
                        value=values[-1], timestamp=page["timestamps"][key][-1]
                    )
                    for key, values in page["data"].items()
                }
                if page
                else {}
            )
        return ctx.readings

    async def get_written_index(self, ctx: DetectorCtx) -> int:
        """The events every serving data logic has written since `prepare()`.

        In a step scan this counts every point, though the bounded data logics
        only hold the current one.
        """
        offsets = _collections_offset(ctx)
        written = await gather_dict({s: s.get_value() for s in offsets})
        collections = min((written[s] + o for s, o in offsets.items()), default=0)
        return collections // ctx.trigger_info.collections_per_event

    async def _collect_end(self, ctx: DetectorCtx, index: int | None) -> int:
        """The event a collect publishes up to, counted since `prepare()`.

        `index` if given, else in a step scan the step point `trigger()` waited
        for, so collect publishes exactly what trigger saw written and never
        reads the counter a second time, or `get_written_index` in a fly scan.
        """
        if index is None:
            index = ctx.triggers or await self.get_written_index(ctx)
        return index

    async def _make_page(self, ctx: DetectorCtx, index: int) -> PartialEventPage | None:
        """One page merging each pageable provider's events up to event `index`.

        `None` if no provider has new events.
        """
        providers = [
            (p, end)
            for p, end in _provider_ends(ctx, index)
            if isinstance(p, PageableDataProvider)
        ]
        pages = await asyncio.gather(*(p.make_page(end) for p, end in providers))
        return _merge_pages(
            [(p, page) for (p, _), page in zip(providers, pages, strict=True)]
        )

    async def asset_docs(
        self, ctx: DetectorCtx, index: int | None
    ) -> AsyncIterator[StreamAsset]:
        """The stream assets of each streamable provider up to event `index`.

        `None` collects a step scan to the end of its point, and a fly scan up to
        `get_written_index`.
        """
        for p, end in _provider_ends(ctx, await self._collect_end(ctx, index)):
            if isinstance(p, StreamableDataProvider):
                async for doc in p.make_stream_docs(end):
                    yield doc

    async def event_pages(
        self, ctx: DetectorCtx, index: int | None
    ) -> AsyncIterator[PartialEventPage]:
        """One page merging each pageable provider's events up to event `index`.

        `None` collects a step scan to the end of its point, and a fly scan up to
        `get_written_index`.
        """
        if page := await self._make_page(ctx, await self._collect_end(ctx, index)):
            yield page

    async def _wait_for(
        self, ctx: DetectorCtx, start: int, end: int, divisor: int
    ) -> AsyncIterator[WatcherUpdate[int]]:
        """Wait until every serving data logic has written up to event `end`.

        Yields the collections written since event `start`, divided by
        `divisor`. Events are counted since `prepare()`.
        """
        offsets = _collections_offset(ctx)
        # A detector with no data logic has nothing to wait for
        if not offsets:
            return
        cpe = ctx.trigger_info.collections_per_event
        target = (end - start) * cpe
        # Until each signal reports, count it as at `start`
        collections = dict.fromkeys(offsets, start * cpe)
        start_time = time.monotonic()
        async for signal, value in observe_signals_value(
            *offsets, timeout=ctx.trigger_info.exposure_timeout
        ):
            collections[signal] = value + offsets[signal]
            done = min(collections.values()) - start * cpe
            yield WatcherUpdate(
                name=ctx.name,
                current=done // divisor,
                initial=0,
                target=target // divisor,
                unit="",
                precision=0,
                time_elapsed=time.monotonic() - start_time,
            )
            if done >= target:
                break

    async def _arm(self) -> AsyncGenerator[None, None]:
        """Run the acquire logic's `acquire()` to its yield, when it is armed."""
        acquisition = self.acquire.acquire() if self.acquire else _no_acquisition()
        try:
            await anext(acquisition)
        except StopAsyncIteration:
            raise RuntimeError(
                f"{type(self.acquire).__name__}.acquire() finished without "
                "yielding: it must yield exactly once, when armed"
            ) from None
        return acquisition

    @asynccontextmanager
    async def _acquiring(self, ctx: DetectorCtx) -> AsyncGenerator[None, None]:
        """Take the acquisition while the data is waited for.

        Takes the one armed by `prepare()` for external triggering, else arms
        one now. Runs it to the end after a clean exit, and closes it otherwise.
        """
        acquisition = ctx.external_acquire_gen or await self._arm()
        # Taken once: a later step point arms afresh
        ctx.external_acquire_gen = None
        async with aclosing(acquisition):
            yield
            async for _ in acquisition:
                raise RuntimeError(
                    f"{type(self.acquire).__name__}.acquire() yielded more than "
                    "once: it must yield exactly once, when armed"
                )

    async def _prepare_trigger_logic(self, value: TriggerInfo) -> None:
        """Set the trigger logic up for `value`, raising if it cannot do it."""
        if value.trigger not in self.supported_triggers:
            supported = ", ".join(sorted(t.name for t in self.supported_triggers))
            raise ValueError(
                f"Trigger type {value.trigger} not supported, supported types are: "
                f"[{supported}]"
            )
        if not self.trigger:
            if value.livetime or value.deadtime or value.exposures_per_collection != 1:
                raise ValueError(
                    "No trigger logic, so cannot set livetime or deadtime "
                    "or exposures_per_collection"
                )
            return
        if _trigger_logic_supported(self.trigger.prepare_exposures_per_collection):
            # Set up even for 1, to clear a previous setting
            await self.trigger.prepare_exposures_per_collection(
                value.exposures_per_collection
            )
        elif value.exposures_per_collection != 1:
            raise ValueError(
                "Multiple exposures per collection not supported: the trigger "
                "logic must implement prepare_exposures_per_collection()"
            )
        match value.trigger:
            case DetectorTrigger.INTERNAL:
                await self.trigger.prepare_internal(
                    num=value.number_of_exposures,
                    livetime=value.livetime,
                    deadtime=value.deadtime,
                )
            case DetectorTrigger.EXTERNAL_EDGE:
                await self.trigger.prepare_edge(
                    num=value.number_of_exposures, livetime=value.livetime
                )
            case DetectorTrigger.EXTERNAL_LEVEL:
                await self.trigger.prepare_level(num=value.number_of_exposures)

    async def _resolve_collection_period(self, value: TriggerInfo) -> float:
        # A livetime of 0 means "use whatever the detector currently has set",
        # which a data logic sizing chunks or a buffer by the period needs to
        # know. Reading it back is reading hardware state, so is opt-in.
        period = value.livetime + value.deadtime
        if (
            value.livetime == 0.0
            and _preserve_detector_state()
            and self.trigger is not None
            and _trigger_logic_supported(self.trigger.default_trigger_info)
        ):
            current = await self.trigger.default_trigger_info()
            if current.livetime or current.deadtime:
                period = current.livetime + current.deadtime
        return value.exposures_per_collection * period


def _config_signals(device: Device) -> set[SignalR]:
    """Return every `Signal` contributing to a Device's `read_configuration()`.

    Recurses into children registered as [](#StandardReadableFormat.CHILD), so
    a Device that registers a sub-Device whole picks up whatever that
    sub-Device declares as configuration. A Device that is not a
    `StandardReadable` contributes nothing.

    A format changed at runtime cannot remove a signal: the formats the class
    declared count as well as the current ones, so dropping `driver.model` from
    configuration does not stop a trigger logic reading it for deadtime.

    Unlike `walk_config_signals` this reads the registry rather than calling
    `read_configuration()`, so it does no I/O and includes read-only `SignalR`s
    as well as `SignalRW`s -- `ADBaseIO.model` is a `SignalR`, and it is
    exactly the signal an areaDetector trigger logic needs for deadtime.

    Deliberately private and unexported: #1367 restructures detector logic, so
    what the public shape of this should be is not settled yet.
    """
    signals: set[SignalR] = set()
    if not isinstance(device, StandardReadable):
        return signals
    current = device.get_readable_formats()
    declared = device._default_readables  # noqa: SLF001
    for child, format in [*declared.items(), *current.items()]:
        match format:
            case StandardReadableFormat.CONFIG_SIGNAL:
                # set_readable_format only accepts a SignalR for a signal format
                signals.add(cast(SignalR, child))
            case StandardReadableFormat.CHILD:
                signals |= _config_signals(child)
            case (
                StandardReadableFormat.HINTED_SIGNAL
                | StandardReadableFormat.UNCACHED_SIGNAL
                | StandardReadableFormat.HINTED_UNCACHED_SIGNAL
            ):
                pass
    return signals


class StandardDetector(
    StandardReadable,
    StandardFlyable[TriggerInfo, DetectorCtx],
    Stageable,
    Triggerable,
    Collectable,
    HasHints,
):
    """Detector base class for step and fly scanning detectors.

    Aggregates trigger, acquire and data logic together in a
    [](#DetectorLogic), which a subclass builds in `logic`:

    ```python
    class MyDetector(StandardDetector):
        def __init__(self, prefix: str, name: str = "") -> None:
            self.driver = MyDriverIO(prefix)
            super().__init__(name=name)

        @cached_property
        def logic(self) -> DetectorLogic:
            return DetectorLogic(
                trigger=MyTriggerLogic(self.driver),
                acquire=MyAcquireLogic(self.driver),
            )
    ```

    For an ad-hoc detector, [](#DetectorLogic.with_device) does the same thing
    without a subclass.

    Signals read during a step scan are registered with
    [](#StandardReadable.set_readable_format), exactly as on any other
    `StandardReadable`; data produced by a `DetectorDataLogic` is added on top
    of those in `read()` and `describe()`, replacing a registered signal of the
    same name, shape and dtype.

    Every verb but `stage()`, `read_configuration()` and
    `describe_configuration()` is for after `stage()`. A step scan triggers and
    reads, and `trigger()` prepares implicitly if needed; a fly scan prepares,
    kicks off, completes and collects. `read()` is for after a `trigger()`, the
    collect verbs for after a `kickoff()`, and `describe()`,
    `describe_collect()` and `get_index()` for after a `prepare()`. An
    externally triggered detector is prepared again before each `trigger()`.
    After any verb raises, `unstage()` and `stage()` it before using it again.

    `WritesStreamAssets` (and its `collect_asset_docs`) and
    `EventPageCollectable` (and its `collect_pages`) are *not* inherited: which
    of them applies depends on what the data logics produce for a given scan, so
    whichever it is gets bound as an instance attribute by `prepare()`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # What the data logics produce, gathered alongside the registered
        # children, so the verb methods themselves need no overriding
        self._read_funcs += (self.__on_read,)
        self._describe_funcs += (self.__on_describe,)
        self._hints_funcs += (self.__on_hints,)
        self._suppressed_datakeys_funcs += (self.__on_suppressed_datakeys,)
        self._unstage_funcs += (self.__on_unstage,)

    @cached_property
    def _state(self) -> _StateMachine:
        return super()._state | {
            _State.PREPARED: {_State.TRIGGERING},
            _State.TRIGGERING: {_State.TRIGGERED, _State.STAGED},
            _State.TRIGGERED: {_State.TRIGGERING, _State.PREPARING},
        }

    @abstract_cached_property
    def logic(self) -> DetectorLogic:
        """The logic that drives this detector."""
        raise NotImplementedError

    async def __on_unstage(self) -> None:
        # Close an acquisition prepare() armed and nothing took. This does not
        # stop the detector, the acquire logic's ensure_stopped() does that: it
        # only ends the generator's tasks now, rather than when it is garbage
        # collected, which asyncio warns about
        if self._ctx and self._ctx.external_acquire_gen:
            await self._ctx.external_acquire_gen.aclose()

    def __on_suppressed_datakeys(self) -> Mapping[str, DataKey | None]:
        """The datakeys of the serving providers, which replace registered signals."""
        return {} if self._ctx is None else self._ctx.datakeys

    async def __on_describe(self) -> dict[str, DataKey]:
        """What the data logics contribute to `describe()`.

        Every serving provider describes its shape for a step scan, whichever
        way it goes on to produce the data.
        """
        return self._prepared_ctx("describe").datakeys

    async def __on_read(self) -> dict[str, Reading]:
        """What the data logics contribute to `read()`: pageable data only.

        Streamable providers produce their data through `collect_asset_docs`.
        """
        self._state.check_one_of(_State.TRIGGERED, verb="read")
        with name_added_to_errors(self):
            return await self.logic.get_readings(self._prepared_ctx("read"))

    def __on_hints(self) -> Iterator[Hints]:
        """The data logics' hinted fields, alongside the children's."""
        if fields := self.logic.get_hinted_fields(self.name, self._ctx):
            yield {"fields": fields}

    def prepare(self, value: TriggerInfo) -> WatchableAsyncStatus[Any]:
        """Set up the scan as `StandardFlyable.prepare`, then bind the collect verb.

        The collect verb matching what the serving providers produce is bound as
        a real instance attribute, so the bundler's isinstance checks
        (`WritesStreamAssets` vs `EventPageCollectable`) see exactly one of them.
        """
        # TODO: delete this override once the run bundler picks the collect verb
        # from what a device produced (coretl/bluesky#1). See ADR 0024.
        status = super().prepare(value)
        # Added first, so it runs before whoever waits on the status
        status.add_callback(self._bind_collect_verb)
        return status

    def _bind_collect_verb(self, status: Status) -> None:
        if not status.success:
            return
        # Those checks resolve with inspect.getattr_static on Python 3.12+, which
        # does not call __getattr__, so a dynamic hook would be invisible; both
        # names are reserved by Device, hence object.__setattr__.
        streamable, pageable = self.logic.produces(self._prepared_ctx("prepare"))
        for verb, method, wanted in (
            ("collect_asset_docs", self._collect_asset_docs, streamable),
            ("collect_pages", self._collect_pages, pageable),
        ):
            if wanted:
                object.__setattr__(self, verb, method)
            elif verb in self.__dict__:
                object.__delattr__(self, verb)

    @WatchableAsyncStatus.wrap
    async def trigger(self) -> AsyncIterator[WatcherUpdate[int]]:
        """Trigger a single exposure.

        If [`prepare()`](#StandardFlyable.prepare) has not been called since the
        last `stage()`, an implicit prepare is performed. When
        [](#OPHYD_ASYNC_PRESERVE_DETECTOR_STATE) is `YES`
        [](#DetectorTriggerLogic.default_trigger_info) is called to read current
        hardware state; otherwise a bare [`TriggerInfo()`](#TriggerInfo) is used.
        `unstage()` cancels a running trigger, and a failed one leaves the detector
        staged.
        """
        if self._state.state is _State.STAGED:
            with name_added_to_errors(self):
                info = await self.logic.default_trigger_info()
            await self.prepare(info)
        async with self._state.during(
            _State.TRIGGERING, end=_State.TRIGGERED, on_error=_State.STAGED
        ):
            with name_added_to_errors(self):
                async for update in self.logic.step(self._prepared_ctx("trigger")):
                    yield update

    async def describe_collect(self) -> dict[str, DataKey]:
        return self._prepared_ctx("describe_collect").datakeys

    async def _collect_asset_docs(
        self, index: int | None = None
    ) -> AsyncIterator[StreamAsset]:
        # Bound as collect_asset_docs when there is a streamable provider
        self._state.check_one_of(
            _State.TRIGGERED, _State.FLYING, _State.FLOWN, verb="collect_asset_docs"
        )
        with name_added_to_errors(self):
            async for doc in self.logic.asset_docs(
                self._prepared_ctx("collect_asset_docs"), index
            ):
                yield doc

    async def _collect_pages(
        self, index: int | None = None
    ) -> AsyncIterator[PartialEventPage]:
        # Bound as collect_pages when there is a pageable provider
        self._state.check_one_of(_State.FLYING, _State.FLOWN, verb="collect_pages")
        with name_added_to_errors(self):
            async for page in self.logic.event_pages(
                self._prepared_ctx("collect_pages"), index
            ):
                yield page

    async def get_index(self) -> int:
        return await self.logic.get_written_index(self._prepared_ctx("get_index"))

    # Back compat - delete before 1.0
    def add_config_signals(self, *signals: SignalR) -> None:
        """Add a signal to read_configuration().

        :param sig: The signal to add
        """
        warnings.warn(
            DeprecationWarning(
                "Use `set_readable_format(signal, StandardReadableFormat"
                ".CONFIG_SIGNAL)` instead of `add_config_signals(signal)`"
            ),
            stacklevel=2,
        )
        for signal in signals:
            self.set_readable_format(signal, StandardReadableFormat.CONFIG_SIGNAL)

    async def get_trigger_deadtime(
        self, settings: Settings | None = None
    ) -> tuple[set[DetectorTrigger], float | None]:
        """Get supported trigger types and deadtime for the detector.

        The trigger logic is given the value of every signal this detector
        reports as configuration, rather than nominating the ones it wants:
        the values that determine deadtime are exactly the ones that ought to
        be recorded for the scan anyway.

        :param settings: Optional settings to use when getting configuration values
        :return: Tuple of supported trigger types and deadtime in seconds
        """
        deadtime = None
        if self.logic.supports_deadtime:
            config_values = SignalDict()
            to_read: list[SignalR] = []
            for sig in _config_signals(self):
                if settings and sig in settings:
                    # Use value from settings if it is in there
                    # cast to a SignalRW because settings can only contain those
                    config_values[sig] = settings[cast(SignalRW, sig)]
                else:
                    to_read.append(sig)
            # Read live values concurrently: this is every configuration signal
            # rather than a handful the logic named, so doing it in series would
            # be one round trip per signal
            for sig, value in zip(
                to_read,
                await asyncio.gather(*(sig.get_value() for sig in to_read)),
                strict=True,
            ):
                config_values[sig] = value
            deadtime = self.logic.get_deadtime(config_values)
        return self.logic.supported_triggers, deadtime


class _LogicDetector(StandardDetector):
    """A concrete `StandardDetector` built around a logic object.

    `StandardDetector.logic` is abstract, so the class itself cannot be
    instantiated; `DetectorLogic.with_device` needs something concrete to wrap a
    logic object in.
    """

    def __init__(self, logic: DetectorLogic, name: str = "") -> None:
        self._logic = logic
        super().__init__(name=name)

    @cached_property
    def logic(self) -> DetectorLogic:
        return self._logic
