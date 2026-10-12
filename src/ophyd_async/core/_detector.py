"""Module which defines abstract classes to work with detectors."""

import asyncio
import os
import time
import warnings
from abc import abstractmethod
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Collection,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
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
    StreamAsset,
    Triggerable,
)
from event_model import DataKey
from event_model.documents import PartialEventPage
from pydantic import Field, NonNegativeInt, PositiveInt, computed_field

from ._data_providers import (
    DataProvider,
    PageableDataProvider,
    StreamableDataProvider,
)
from ._device import Device
from ._flyable import FlyableLogic, StandardFlyable, _FlyStage
from ._readable import StandardReadable, StandardReadableFormat
from ._settings import Settings
from ._signal import SignalDict, SignalR, SignalRW, observe_signals_value
from ._status import AsyncStatus, WatchableAsyncStatus
from ._utils import (
    DEFAULT_TIMEOUT,
    ConfinedModel,
    T,
    WatcherUpdate,
    abstract_cached_property,
    error_if_none,
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

    number_of_events: NonNegativeInt = Field(default=1)
    """Number of bluesky events that will be emitted, (0 means infinite)."""

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

    Subclasses implement four hooks that are called at defined points in the
    `[](#StandardDetector)` lifecycle:

    - `ensure_ready` — called from `stage()` to put the detector into a known
      idle state before a scan begins.
    - `start_acquiring` — called from `prepare()`, `kickoff()`, or `trigger()`
      to start the detector acquiring.
    - `wait_for_idle` — called after the final collection completes to confirm
      the detector has returned to idle.
    - `ensure_stopped` — called from `unstage()` to stop the detector and
      perform any end-of-scan cleanup.

    The default `ensure_ready` delegates to `ensure_stopped`, which is correct
    for detectors where stage-time reset and scan-end teardown are identical.
    Override `ensure_ready` when the two phases require different behaviour
    (e.g. arming the detector once at stage time while keeping it armed across
    multiple kickoff/complete cycles).
    """

    async def ensure_ready(self):
        """Ensure the detector is idle before a scan.

        Called from `stage()`. The default implementation delegates to
        `ensure_stopped`, which is sufficient for detectors that perform the
        same reset at stage time as at unstage time. Override this method when
        a different action is required at stage time (for example, arming the
        detector once so it is ready for multiple kickoff/complete cycles
        without re-arming).
        """
        await self.ensure_stopped()

    @abstractmethod
    async def start_acquiring(self):
        """Start the detector acquiring.

        Called from `prepare()` for external-trigger modes, and from
        `kickoff()` / `trigger()` for internal-trigger modes.
        """

    @abstractmethod
    async def wait_for_idle(self):
        """Wait for the detector to return to idle after the final collection."""

    @abstractmethod
    async def ensure_stopped(self):
        """Stop the detector and perform end-of-scan cleanup.

        Called from `unstage()`.
        """


#: What a `DetectorDataLogic` works out in `make_data_provider` and needs back
#: in `start`. Use `None` for a logic whose `start` needs nothing.
DataCtxT = TypeVar("DataCtxT")


class DetectorDataLogic(Generic[DataCtxT]):
    """Abstract base class for detector data logic and handling.

    An implementation describes the data it would produce in
    `make_data_provider`, and does the writes that make it happen in `start`.
    The detector asks every data logic it has what it would make, decides which
    providers it will use, and starts only those.

    What `make_data_provider` worked out and `start` needs is threaded between
    them through an explicit context object rather than stored on the logic, as
    `FlyableLogic` threads its own: the detector may make a provider and then
    never start it, so a logic holding that state on itself would be left
    carrying a plan that was abandoned. A logic whose `start` needs nothing from
    `make_data_provider` uses `None` for the context.

    Whether the data is published as stream assets or event pages is the type
    of the provider that `make_data_provider` returns, not a declaration on the
    logic, so one logic may produce either depending on the scan.

    A source that produces a single value per event, like a plugin scalar, needs
    no data logic at all: register its signal with
    [](#StandardReadable.set_readable_format) instead.
    """

    #: Add this suffix to the detector name to specify the datakey. These need to be
    #: different for each DetectorDataLogic added to a detector
    datakey_suffix: str = ""

    @abstractmethod
    async def make_data_provider(
        self,
        datakey_name: str,
        num_collections: int,
        period: float,
        flush_period: float,
    ) -> tuple[DataProvider, DataCtxT] | None:
        """Describe the data this logic would produce for this scan.

        This must not start anything acquiring: the detector may discard the
        provider without calling `start`, and only starts the ones it will use.
        Return a [](#StreamableDataProvider) to publish stream assets, or a
        [](#PageableDataProvider) to publish event pages, together with the
        context to hand back to `start`.

        Return `None` to sit this scan out, for instance a finite buffer asked
        for an unbounded number of collections, or a plugin that is switched
        off. This is not an error and is not warned about.

        :param datakey_name: the detector name plus this logic's datakey_suffix
        :param num_collections: total collections for the scan, 0 meaning
            unbounded
        :param period: how long each collection takes, livetime + deadtime, so
            the provider can size its chunks or its buffer. 0 means "use
            whatever is currently set on the hardware".
        :param flush_period: how many seconds between flushes of the data being
            written, so a logic that buffers can size that buffer. 0 means "use
            whatever is currently set on the hardware".
        """
        raise NotImplementedError(self)

    async def start(self, ctx: DataCtxT) -> None:
        """Make the provider just returned by `make_data_provider` take data.

        Called only for the providers the detector will actually use, so this is
        where writes that arm hardware or open files belong.

        :param ctx: the context returned by `make_data_provider`.
        """

    def get_hinted_fields(self, datakey_name: str) -> Sequence[str]:
        """Return the hinted streams."""
        return []

    async def stop(self) -> None:
        """Stop taking data."""
        return None


@dataclass
class _Served:
    """One data logic, the provider it made, and the context to start it with."""

    logic: DetectorDataLogic[Any]
    provider: DataProvider
    ctx: Any
    #: The datakeys the provider makes, as published in describe()
    datakeys: dict[str, DataKey] = field(default_factory=dict)

    async def start(self) -> None:
        await self.logic.start(self.ctx)


def _same_shape(a: DataKey, b: DataKey) -> bool:
    return (a["shape"], a.get("dtype_numpy")) == (b["shape"], b.get("dtype_numpy"))


def _merge_datakeys(tier: str, served: Sequence[_Served]) -> dict[str, DataKey]:
    """Merge the datakeys of one tier, raising on a key made twice."""
    merged: dict[str, DataKey] = {}
    for s in served:
        if duplicates := sorted(merged.keys() & s.datakeys.keys()):
            raise ValueError(f"Duplicate {tier} datakeys {duplicates}")
        merged |= s.datakeys
    return merged


def _check_shadow(
    lower: str, lower_keys: Mapping[str, DataKey], higher: Mapping[str, DataKey]
) -> set[str]:
    """Return the keys of `lower` that `higher` shadows.

    A key shadows another only with the same shape and dtype: the same name for
    a different quantity is a duplicate.
    """
    shadowed = lower_keys.keys() & higher.keys()
    if mismatched := sorted(
        k for k in shadowed if not _same_shape(lower_keys[k], higher[k])
    ):
        raise ValueError(
            f"Duplicate datakeys {mismatched}: {lower} with a different shape or "
            "dtype from the data that would replace them"
        )
    return shadowed


def _all_the_same(collections_written: Sequence[int]) -> int:
    if len(set(collections_written)) > 1:
        raise RuntimeError(
            "Detectors have written different numbers of collections: "
            f"{set(collections_written)}"
        )
    return collections_written[0] if collections_written else 0


def _merge_pages(pages: Sequence[PartialEventPage | None]) -> PartialEventPage | None:
    """Merge one page per pageable provider into a single page."""
    made = [page for page in pages if page is not None]
    if not made:
        return None
    if len(made) != len(pages) or len({len(page["time"]) for page in made}) > 1:
        raise ValueError(
            "Pageable data providers produced different numbers of events: "
            f"{[len(page['time']) if page else 0 for page in pages]}"
        )
    merged: PartialEventPage = {
        "data": {},
        "timestamps": {},
        # Each event happened when the last of its providers' data arrived
        "time": [
            max(times) for times in zip(*(page["time"] for page in made), strict=True)
        ],
    }
    for page in made:
        if overlap := sorted(merged["data"].keys() & page["data"].keys()):
            raise ValueError(f"Duplicate keys in event pages: {overlap}")
        merged["data"].update(page["data"])
        merged["timestamps"].update(page["timestamps"])
    return merged


class _DetectorDataSet:
    """The data a detector's data logics produce for the scan being prepared.

    Decides which data logics serve each prepare and publishes their data in
    three tiers, each shadowing the one before by datakey name: readables (the
    detector's registered signals), pageable providers (event pages) and
    streamable providers (stream assets). Everything it holds is cleared by
    `clear()`, called on stage and unstage.
    """

    def __init__(self, logics: Sequence[DetectorDataLogic[Any]]) -> None:
        self._logics = logics
        self.clear()

    def clear(self) -> None:
        #: The detector name the data was served under
        self.name = ""
        self.serving: list[_Served] = []
        #: The registered signals that a serving provider shadows
        self.shadowed_readables: set[str] = set()
        self._collections_per_event = 1
        #: What the serving providers were made for, None if not served
        self._made_for: tuple[int, float] | None = None
        #: Where every provider was when the current event or fly scan started
        self._baseline = 0
        #: How far collect_pages() has published
        self._paged_up_to = 0

    @property
    def streamable(self) -> list[StreamableDataProvider]:
        return [
            s.provider
            for s in self.serving
            if isinstance(s.provider, StreamableDataProvider)
        ]

    @property
    def pageable(self) -> list[PageableDataProvider]:
        return [
            s.provider
            for s in self.serving
            if isinstance(s.provider, PageableDataProvider)
        ]

    async def stop_all(self) -> None:
        await asyncio.gather(*(dl.stop() for dl in self._logics))

    async def serve(
        self,
        name: str,
        info: TriggerInfo,
        readables: Mapping[str, DataKey],
        readable_hints: Collection[str],
    ) -> None:
        """Choose and start the providers that serve this scan.

        :param name: the detector name, which prefixes every datakey
        :param info: the period-resolved TriggerInfo
        :param readables: the describe() of the detector's registered signals
        :param readable_hints: which of those are hinted
        """
        cpe = info.collections_per_event
        made_for = (cpe, info.livetime + info.deadtime)
        # A streamable provider is kept while it would be made the same, so a
        # step scan keeps one stream resource; event pages hold no such state,
        # so pageable providers are made afresh on every prepare
        kept = [
            s
            for s in self.serving
            if made_for == self._made_for and name == self.name
            if isinstance(s.provider, StreamableDataProvider)
        ]
        dropped = [s for s in self.serving if s not in kept]
        self.clear()
        await asyncio.gather(*(s.logic.stop() for s in dropped))
        try:
            await self._serve(name, info, made_for, kept, readables, readable_hints)
        except Exception:
            # Left unserved, so nothing may be left running untracked
            await asyncio.gather(*(s.logic.stop() for s in kept))
            raise

    async def _serve(
        self,
        name: str,
        info: TriggerInfo,
        made_for: tuple[int, float],
        kept: list[_Served],
        readables: Mapping[str, DataKey],
        readable_hints: Collection[str],
    ) -> None:
        cpe = info.collections_per_event
        kept_logics = [s.logic for s in kept]
        made = await asyncio.gather(
            *(
                self._make(dl, name, info)
                for dl in self._logics
                if not any(dl is k for k in kept_logics)
            )
        )
        candidates = kept + [s for s in made if s is not None]
        if self._logics and not candidates:
            raise ValueError(
                f"{name}: has data logics but none of them can serve this scan "
                f"({info.number_of_events=})"
            )
        await asyncio.gather(*(self._describe(s, cpe) for s in candidates))
        serving, shadowed = self._select(candidates, readables)
        self._warn_on_lost_hints(name, serving, shadowed, readable_hints)
        await asyncio.gather(*(s.start() for s in serving if s not in kept))
        self.name = name
        self.serving = serving
        self.shadowed_readables = shadowed
        self._collections_per_event = cpe
        self._made_for = made_for

    async def _make(
        self, dl: DetectorDataLogic[Any], name: str, info: TriggerInfo
    ) -> _Served | None:
        made = await dl.make_data_provider(
            name + dl.datakey_suffix,
            info.number_of_collections,
            info.livetime + info.deadtime,
            info.flush_period,
        )
        return None if made is None else _Served(dl, *made)

    @staticmethod
    async def _describe(served: _Served, collections_per_event: int) -> None:
        served.datakeys = await served.provider.make_datakeys(collections_per_event)

    def _select(
        self, candidates: Sequence[_Served], readables: Mapping[str, DataKey]
    ) -> tuple[list[_Served], set[str]]:
        """Apply the shadowing chain readables < pageable < streamable."""
        streamable = [
            s for s in candidates if isinstance(s.provider, StreamableDataProvider)
        ]
        pageable = [
            s for s in candidates if isinstance(s.provider, PageableDataProvider)
        ]
        stream_keys = _merge_datakeys("streamable", streamable)
        page_keys = _merge_datakeys("pageable", pageable)
        _check_shadow("pageable", page_keys, stream_keys)
        # A pageable provider whose every key the stream assets carry sits out
        pageable = [s for s in pageable if not s.datakeys.keys() <= stream_keys.keys()]
        # Only while bluesky's RunBundler collects stream assets XOR event pages
        if streamable and pageable:
            uncovered = sorted(
                _merge_datakeys("pageable", pageable).keys() - stream_keys.keys()
            )
            raise TypeError(
                f"would produce {uncovered} as event pages and the rest of its "
                "data as stream assets; these cannot be combined on one detector"
            )
        serving = streamable + pageable
        provided = _merge_datakeys("served", serving)
        return serving, _check_shadow("readables", readables, provided)

    def _warn_on_lost_hints(
        self,
        name: str,
        serving: Sequence[_Served],
        shadowed: set[str],
        readable_hints: Collection[str],
    ) -> None:
        hinted = {
            field
            for s in serving
            for field in s.logic.get_hinted_fields(name + s.logic.datakey_suffix)
        }
        if differ := sorted(
            k for k in shadowed if (k in readable_hints) != (k in hinted)
        ):
            warnings.warn(
                f"{name}: {differ} are hinted differently by the readables and the "
                "data logics that shadow them; the data logics' hints are used",
                stacklevel=2,
            )

    def hinted_fields(self, name: str) -> Iterator[Sequence[str]]:
        """Yield the hinted fields of each data logic producing data.

        Before anything is served every data logic is asked, since `hints` is
        read outside a scan too; once served, only the serving ones.
        """
        logics = [s.logic for s in self.serving] if self._made_for else self._logics
        for dl in logics:
            if fields := dl.get_hinted_fields(name + dl.datakey_suffix):
                yield fields

    async def _collections_written(self) -> list[int]:
        return await asyncio.gather(
            *(s.provider.collections_written_signal.get_value() for s in self.serving)
        )

    async def step_boundary(self) -> None:
        await asyncio.gather(*(s.provider.step_boundary() for s in self.serving))

    async def take_baseline(self) -> None:
        """Note where every provider is up to, as the start of an event or scan."""
        self._baseline = _all_the_same(await self._collections_written())
        self._paged_up_to = self._baseline

    async def wait_for(
        self, collections: int, timeout: float, divisor: int = 1
    ) -> AsyncIterator[WatcherUpdate[int]]:
        """Wait for `collections` past the baseline, yielding progress."""
        if not self.serving:
            return
        initial = self._baseline
        target = initial + collections
        signals = [s.provider.collections_written_signal for s in self.serving]
        current = dict.fromkeys(signals, initial)
        start_time = time.monotonic()
        async for signal, value in observe_signals_value(*signals, timeout=timeout):
            current[signal] = value
            written = min(current.values())
            yield WatcherUpdate(
                name=self.name,
                current=written // divisor,
                initial=initial // divisor,
                target=target // divisor,
                unit="",
                precision=0,
                time_elapsed=time.monotonic() - start_time,
            )
            if written >= target:
                break

    def describe(self) -> dict[str, DataKey]:
        return _merge_datakeys("served", self.serving)

    async def read(self) -> dict[str, Reading]:
        """Read the current step point's event from each pageable provider."""
        cpe = self._collections_per_event
        pages = await asyncio.gather(
            *(
                p.make_page(self._baseline, self._baseline + cpe, cpe)
                for p in self.pageable
            )
        )
        readings: dict[str, Reading] = {}
        for page in pages:
            if page is None or len(page["time"]) != 1:
                raise ValueError(
                    "read() expected a single event, got "
                    f"{0 if page is None else len(page['time'])}"
                )
            for key, values in page["data"].items():
                readings[key] = Reading(value=values[0], timestamp=page["time"][0])
        return readings

    async def get_index(self) -> int:
        written = await self._collections_written()
        return min(written, default=0) // self._collections_per_event

    async def collect_asset_docs(self, index: int) -> AsyncIterator[StreamAsset]:
        cpe = self._collections_per_event
        for provider in self.streamable:
            async for doc in provider.make_stream_docs(index * cpe, cpe):
                yield doc

    async def collect_pages(self, index: int) -> AsyncIterator[PartialEventPage]:
        cpe = self._collections_per_event
        up_to = index * cpe
        page = _merge_pages(
            await asyncio.gather(
                *(p.make_page(self._paged_up_to, up_to, cpe) for p in self.pageable)
            )
        )
        if page is not None:
            self._paged_up_to = up_to
            yield page


@dataclass
class _FlyCtx:
    """What prepare() set up, threaded through trigger() or kickoff()."""

    trigger_info: TriggerInfo
    #: How many trigger()s this prepare has served
    triggers: int = 0


def _preserve_detector_state() -> bool:
    # Opt-in: set OPHYD_ASYNC_PRESERVE_DETECTOR_STATE=YES to have the detector
    # read back current hardware state via default_trigger_info(). See ADR 0013.
    # TODO: flip default to YES and remove this guard in a future PR once
    # downstream code has had time to implement default_trigger_info().
    return os.environ.get("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "NO").upper() == "YES"


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectorLogic(FlyableLogic[TriggerInfo, _FlyCtx]):
    """Drive a detector through its trigger, acquire and data logics.

    The fields are configuration and never change; what a scan sets up is held
    separately, so `dataclasses.replace` makes an independent copy.

    :param trigger: sets up exposures; without one only internal triggering
        with the hardware's current timing is supported
    :param acquire: starts and stops acquisition
    :param data: the data logics, each publishing one part of the data
    """

    trigger: DetectorTriggerLogic | None = None
    acquire: DetectorAcquireLogic | None = None
    data: Sequence[DetectorDataLogic[Any]] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", tuple(self.data))

    @cached_property
    def _dataset(self) -> _DetectorDataSet:
        return _DetectorDataSet(self.data)

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

    def get_hinted_fields(self, name: str) -> Iterator[Sequence[str]]:
        """Yield the hinted fields of each data logic that is producing data.

        :param name: the detector name, which prefixes every datakey
        """
        return self._dataset.hinted_fields(name)

    async def stop(self) -> None:
        """Stop every data logic and the detector."""
        coros: list[Awaitable] = [self._dataset.stop_all()]
        if self.acquire:
            coros.append(self.acquire.ensure_stopped())
        await asyncio.gather(*coros)

    async def on_stage(self) -> None:
        """Drop prepared data, stop every data logic, and make the detector ready."""
        self._dataset.clear()
        coros: list[Awaitable] = [self._dataset.stop_all()]
        if self.acquire:
            coros.append(self.acquire.ensure_ready())
        await asyncio.gather(*coros)

    async def on_unstage(self) -> None:
        """Drop prepared data and stop everything."""
        self._dataset.clear()
        await self.stop()

    async def default_trigger_info(self) -> TriggerInfo:
        """The TriggerInfo to prepare with when a plan calls trigger() directly.

        The trigger logic's `default_trigger_info()` if
        [](#OPHYD_ASYNC_PRESERVE_DETECTOR_STATE) is `YES`, otherwise a bare
        [`TriggerInfo()`](#TriggerInfo).
        """
        if _preserve_detector_state() and self.trigger is not None:
            if not _trigger_logic_supported(self.trigger.default_trigger_info):
                raise RuntimeError(
                    "OPHYD_ASYNC_PRESERVE_DETECTOR_STATE=YES is set but the "
                    "trigger logic has no default_trigger_info() - implement "
                    "default_trigger_info() on your DetectorTriggerLogic subclass "
                    "or unset the environment variable."
                )
            return await self.trigger.default_trigger_info()
        return TriggerInfo()

    async def on_prepare(self, value: TriggerInfo) -> _FlyCtx:
        """Set the trigger logic up, filling in the period it reads back.

        The data is served afterwards by `serve_data`, which needs the detector
        name.
        """
        await self._prepare_trigger_logic(value)
        # After preparing the trigger logic, as the hardware then holds the
        # values read back
        return _FlyCtx(trigger_info=await self._resolve_period(value))

    async def serve_data(
        self,
        name: str,
        ctx: _FlyCtx,
        readables: Mapping[str, DataKey],
        readable_hints: Collection[str],
    ) -> None:
        """Choose and start the data providers, then arm an external trigger.

        :param name: the detector name, which prefixes every datakey
        :param ctx: what `on_prepare` returned
        :param readables: the describe() of the detector's registered signals,
            which providers may shadow
        :param readable_hints: which of those are hinted
        """
        info = ctx.trigger_info
        await self._dataset.serve(name, info, readables, readable_hints)
        # External triggering arms now and waits on its trigger source, after
        # the data logics are started so their buffers are sized
        if self.acquire and info.trigger is not DetectorTrigger.INTERNAL:
            await self.acquire.start_acquiring()

    async def on_kickoff(self, ctx: _FlyCtx) -> _FlyCtx:
        """Start the fly scan, noting where the providers had got to."""
        await self._dataset.take_baseline()
        if self.acquire and ctx.trigger_info.trigger is DetectorTrigger.INTERNAL:
            await self.acquire.start_acquiring()
        return ctx

    async def on_complete(self, ctx: _FlyCtx) -> AsyncIterator[WatcherUpdate]:
        """Wait for the scan to finish, reporting events written as progress."""
        info = ctx.trigger_info
        async for update in self._dataset.wait_for(
            info.number_of_collections,
            info.exposure_timeout,
            divisor=info.collections_per_event,
        ):
            yield update
        if self.acquire:
            await self.acquire.wait_for_idle()

    async def on_trigger(self, ctx: _FlyCtx) -> AsyncIterator[WatcherUpdate]:
        """Take one event's worth of exposures and wait for them to be written."""
        info = ctx.trigger_info
        if info.number_of_events != 1:
            raise ValueError(
                "trigger() is not supported for multiple events, the detector was "
                f"prepared with number_of_events={info.number_of_events}."
            )
        # prepare() readied the providers for the first point
        if ctx.triggers:
            await self._dataset.step_boundary()
        ctx.triggers += 1
        # Before acquisition starts, or frames written in between would count
        await self._dataset.take_baseline()
        # External triggering was armed at prepare
        if self.acquire and info.trigger is DetectorTrigger.INTERNAL:
            await self.acquire.start_acquiring()
        async for update in self._dataset.wait_for(
            info.collections_per_event, info.exposure_timeout
        ):
            yield update
        if self.acquire:
            await self.acquire.wait_for_idle()

    async def _prepare_trigger_logic(self, value: TriggerInfo) -> None:
        if self.trigger and _trigger_logic_supported(
            self.trigger.prepare_exposures_per_collection
        ):
            # Set up even for 1, to clear a previous setting
            await self.trigger.prepare_exposures_per_collection(
                value.exposures_per_collection
            )
        elif value.exposures_per_collection != 1:
            raise ValueError("Multiple exposures per collection not supported")
        if value.trigger not in self.supported_triggers:
            format_triggers = ", ".join(sorted(t.name for t in self.supported_triggers))
            raise ValueError(
                f"Trigger type {value.trigger} not supported, supported types are: "
                f"[{format_triggers}]"
            )
        if self.trigger:
            match value.trigger:
                case DetectorTrigger.INTERNAL:
                    await self.trigger.prepare_internal(
                        num=value.number_of_exposures,
                        livetime=value.livetime,
                        deadtime=value.deadtime,
                    )
                case DetectorTrigger.EXTERNAL_EDGE:
                    await self.trigger.prepare_edge(
                        num=value.number_of_exposures,
                        livetime=value.livetime,
                    )
                case DetectorTrigger.EXTERNAL_LEVEL:
                    await self.trigger.prepare_level(
                        num=value.number_of_exposures,
                    )
        elif value.livetime != 0.0 or value.deadtime != 0.0:
            raise ValueError("No trigger logic, so cannot set livetime or deadtime")

    async def _resolve_period(self, value: TriggerInfo) -> TriggerInfo:
        # A livetime of 0 means "use whatever the detector currently has set",
        # which a data logic sizing chunks or a buffer by the period needs to
        # know. Reading it back is reading hardware state, so is opt-in.
        if (
            value.livetime == 0.0
            and _preserve_detector_state()
            and self.trigger is not None
            and _trigger_logic_supported(self.trigger.default_trigger_info)
        ):
            current = await self.trigger.default_trigger_info()
            if current.livetime or current.deadtime:
                value = value.model_copy(
                    update={"livetime": current.livetime, "deadtime": current.deadtime}
                )
        return value


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
    StandardFlyable[TriggerInfo, _FlyCtx],
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

    Every verb but `stage()` raises before `stage()` is called. A step scan
    triggers and reads, and `trigger()` prepares implicitly if needed; a fly
    scan prepares, kicks off, completes and collects. `read()` is only for
    after a `trigger()`, and the collect verbs only for after a `kickoff()`;
    `describe()`, `describe_collect()` and `get_index()` need a `prepare()`.
    `read_configuration()` and `describe_configuration()` have no such
    requirement.

    `WritesStreamAssets` (and its `collect_asset_docs`) and
    `EventPageCollectable` (and its `collect_pages`) are *not* inherited: which
    of them applies depends on what the data logics produce for a given scan, so
    whichever it is gets bound as an instance attribute by `prepare()`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # What the data logics produce, gathered alongside the registered
        # children, so the verb methods themselves need no overriding
        self._read_funcs += (self._data_read,)
        self._describe_funcs += (self._data_describe,)
        self._hint_funcs += (self._data_hints,)

    @abstract_cached_property
    def logic(self) -> DetectorLogic:
        """The logic that drives this detector."""
        raise NotImplementedError

    @property
    def _dataset(self) -> _DetectorDataSet:
        # The logic's data set, which this Device publishes
        return self.logic._dataset  # noqa: SLF001

    @contextmanager
    def _noted(self) -> Iterator[None]:
        """Note the detector's name on any exception a logic raises."""
        try:
            yield
        except Exception as e:
            e.add_note(f"raised by {self.name}")
            raise

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

    async def _on_stage(self) -> None:
        try:
            await super()._on_stage()
        finally:
            self._bind_collect_verbs()

    async def _on_unstage(self) -> None:
        try:
            await super()._on_unstage()
        finally:
            self._bind_collect_verbs()

    def _bind_collect_verbs(self, *, prepared: bool | None = None) -> None:
        # Bind the collect verb matching what the data logics will produce, as a
        # real instance attribute so the bundler's isinstance checks
        # (WritesStreamAssets vs EventPageCollectable) see exactly one of them.
        # Those resolve with inspect.getattr_static on Python 3.12+, which does not
        # call __getattr__, so a dynamic hook would be invisible; both names are
        # reserved by Device, hence object.__setattr__.
        # TODO: all of this goes once the run bundler picks the collect verb from
        # what a device actually produced rather than from its static type - a
        # change in progress upstream. _collect_asset_docs and _collect_pages then
        # become plain public methods and a detector declares both. See ADR 0023.
        if prepared is None:
            prepared = self._fly_stage not in (_FlyStage.UNSTAGED, _FlyStage.STAGED)
        dataset = self._dataset
        for verb, method, wanted in (
            ("collect_asset_docs", self._collect_asset_docs, dataset.streamable),
            ("collect_pages", self._collect_pages, dataset.pageable),
        ):
            if prepared and wanted:
                object.__setattr__(self, verb, method)
            elif verb in self.__dict__:
                object.__delattr__(self, verb)

    async def _prepare_logic(self, value: TriggerInfo) -> _FlyCtx:
        # Called by StandardFlyable.prepare, which checks and sets the fly state
        self._bind_collect_verbs()
        readables, readable_hints = await self._readables_for_shadowing()
        with self._noted():
            ctx = await self.logic.on_prepare(value)
            await self.logic.serve_data(self.name, ctx, readables, readable_hints)
        self._bind_collect_verbs(prepared=True)
        return ctx

    async def _readables_for_shadowing(self) -> tuple[dict[str, DataKey], set[str]]:
        if not self.logic.data:
            # Nothing can shadow them, so save describing them
            return {}, set()
        readables = await super()._formatted_describe()
        hints = {f for h in super()._formatted_hints() for f in h.get("fields", [])}
        return readables, hints

    @WatchableAsyncStatus.wrap
    async def trigger(self) -> AsyncIterator[WatcherUpdate[int]]:
        """Trigger a single exposure.

        If [`prepare()`](#StandardFlyable.prepare) has not been called since the
        last `stage()`, an implicit prepare is performed. When
        [](#OPHYD_ASYNC_PRESERVE_DETECTOR_STATE) is `YES`
        [](#DetectorTriggerLogic.default_trigger_info) is called to read current
        hardware state; otherwise a bare [`TriggerInfo()`](#TriggerInfo) is used.
        """
        self._check_staged("trigger")
        match self._fly_stage:
            case _FlyStage.STAGED:
                with self._noted():
                    info = await self.logic.default_trigger_info()
                await self.prepare(info)
            case _FlyStage.KICKED_OFF:
                raise RuntimeError(
                    f"{self.name}: trigger() cannot be called while kicked off"
                )
            case _FlyStage.COMPLETED:
                raise RuntimeError(
                    f"{self.name}: complete() was called, so call prepare() again "
                    "before trigger()"
                )
            case _FlyStage.PREPARED:
                ctx = self._fly_ctx
                if (
                    ctx.triggers
                    and ctx.trigger_info.trigger != DetectorTrigger.INTERNAL
                ):
                    raise RuntimeError(
                        f"{self.name}: an externally triggered detector needs "
                        "prepare() again before another trigger()"
                    )
            case _FlyStage.UNSTAGED:
                pass  # already raised
        with self._noted():
            async for update in self.logic.on_trigger(self._fly_ctx):
                yield update

    @AsyncStatus.wrap
    async def kickoff(self) -> None:
        """Start the fly scan."""
        self._check_staged("kickoff")
        if self._fly_stage is _FlyStage.PREPARED:
            if self._fly_ctx.triggers:
                raise RuntimeError(
                    f"{self.name}: kickoff() cannot follow trigger(), "
                    "call prepare() again first"
                )
            if not self.logic.data:
                raise ValueError(
                    f"Detector {self.name} has no collectable data, so cannot kickoff"
                )
        await super().kickoff()

    def _check_read(self) -> None:
        self._check_staged("read")
        match self._fly_stage:
            case _FlyStage.STAGED:
                raise RuntimeError(f"{self.name}: prepare() must be called first")
            case _FlyStage.PREPARED if self._fly_ctx.triggers:
                pass
            case _FlyStage.PREPARED:
                raise RuntimeError(
                    f"{self.name}: trigger() must be called before read()"
                )
            case _FlyStage.KICKED_OFF | _FlyStage.COMPLETED:
                raise RuntimeError(
                    f"{self.name}: read() is not available in a fly scan, "
                    "use the collect verbs"
                )
            case _FlyStage.UNSTAGED:
                pass  # already raised

    def _check_collect(self, verb: str, *, after_trigger: bool = False) -> None:
        self._prepared_fly_ctx  # noqa: B018 - raises if unstaged or unprepared
        if self._fly_stage is _FlyStage.PREPARED and not (
            after_trigger and self._fly_ctx.triggers
        ):
            raise RuntimeError(f"{self.name}: kickoff() must be called before {verb}()")

    # The registered signals contribute to read(), describe() and hints, less
    # those that a serving data provider shadows

    async def _formatted_read(self) -> dict[str, Reading]:
        return self._unshadowed(await super()._formatted_read())

    async def _formatted_describe(self) -> dict[str, DataKey]:
        return self._unshadowed(await super()._formatted_describe())

    def _formatted_hints(self) -> Iterator[Hints]:
        shadowed = self._dataset.shadowed_readables
        for hint in super()._formatted_hints():
            if fields := hint.get("fields"):
                unshadowed = [f for f in fields if f not in shadowed]
                if not unshadowed:
                    continue
                hint = hint.copy()
                hint["fields"] = unshadowed
            yield hint

    def _unshadowed(self, values: dict[str, T]) -> dict[str, T]:
        shadowed = self._dataset.shadowed_readables
        return {k: v for k, v in values.items() if k not in shadowed}

    async def _data_read(self) -> dict[str, Reading]:
        """What the data logics contribute to `read()`: pageable data only.

        Streamable providers produce their data through `collect_asset_docs`.
        """
        self._check_read()
        return await self._dataset.read()

    async def _data_describe(self) -> dict[str, DataKey]:
        """What the data logics contribute to `describe()`.

        Every serving provider describes its shape for a step scan, whichever
        way it goes on to produce the data. Raises if nothing has been
        prepared, so that a descriptor is never emitted without the detector's
        data keys.
        """
        self._prepared_fly_ctx  # noqa: B018 - raises if unstaged or unprepared
        return self._dataset.describe()

    async def describe_collect(self) -> dict[str, DataKey]:
        self._prepared_fly_ctx  # noqa: B018 - raises if unstaged or unprepared
        return self._dataset.describe()

    def _data_hints(self) -> Iterator[Hints]:
        """The data logics' hinted fields, alongside the children's."""
        for fields in self.logic.get_hinted_fields(self.name):
            yield {"fields": list(fields)}

    async def _collect_asset_docs(
        self, index: int | None = None
    ) -> AsyncIterator[StreamAsset]:
        # Bound as collect_asset_docs when there is a streamable provider
        # bluesky's read() collects the stream assets of each step scan point
        self._check_collect("collect_asset_docs", after_trigger=True)
        if index is None:
            # The index is optional, and provided for fly scans, if there is
            # more than one detector to make sure they collect in step
            index = await self.get_index()
        async for doc in self._dataset.collect_asset_docs(index):
            yield doc

    async def _collect_pages(
        self, index: int | None = None
    ) -> AsyncIterator[PartialEventPage]:
        # Bound as collect_pages when there is a pageable provider
        self._check_collect("collect_pages")
        if index is None:
            index = await self.get_index()
        with self._noted():
            async for page in self._dataset.collect_pages(index):
                yield page

    async def get_index(self) -> int:
        self._prepared_fly_ctx  # noqa: B018 - raises if unstaged or unprepared
        return await self._dataset.get_index()


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
