"""Use-case tests for the detector core (spec: detector-logic-spec.md).

Written from the spec for the detector-core rewrite. The strict xfails are
behaviour the spec wants that something outside ophyd-async still prevents.

The fakes in this file stand in for the three kinds of device a detector drives:
a device-side `Source` (counter plus data), an acquire logic that fills it, and
data logics that publish it as stream assets.
"""

import asyncio
import re
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Sequence
from functools import cached_property
from typing import Any, cast

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np
import pytest
from bluesky import RunEngine

from ophyd_async.core import (
    KICKED_OFF,
    Array1D,
    AsyncStatus,
    DetectorAcquireLogic,
    DetectorLogic,
    DetectorTrigger,
    DetectorTriggerLogic,
    FlyableLogic,
    SignalR,
    StandardDetector,
    StandardFlyable,
    StreamResourceDataProvider,
    StreamResourceInfo,
    TransitionError,
    TriggerInfo,
    UnboundedDataLogic,
    soft_signal_rw,
)
from ophyd_async.core import StandardReadableFormat as Format

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class Source:
    """The device side: a counter of collections written, and their data.

    Each collection written holds the next number from a counter that never
    resets, so every point of a scan has distinguishable data.
    """

    def __init__(self) -> None:
        self.written = soft_signal_rw(int)
        self.data: list[int] = []
        self._next = 0

    async def produce(self, collections: int) -> None:
        for _ in range(collections):
            self.data.append(self._next)
            self._next += 1
        await self.written.set(len(self.data))

    async def rearm(self) -> None:
        self.data.clear()
        await self.written.set(0)


class InternalTrigger(DetectorTriggerLogic):
    async def prepare_internal(self, num: int, livetime: float, deadtime: float): ...

    async def default_trigger_info(self) -> TriggerInfo:
        return TriggerInfo()


class ExternalTrigger(InternalTrigger):
    async def prepare_edge(self, num: int, livetime: float): ...


class FillingAcquire(DetectorAcquireLogic):
    """Every acquire() produces `per_start` collections in each source."""

    def __init__(self, *sources: Source, per_start: int = 1) -> None:
        self.sources = sources
        self.per_start = per_start
        self.idle_waits = 0

    async def acquire(self):
        for source in self.sources:
            await source.produce(self.per_start)
        yield
        self.idle_waits += 1

    async def ensure_stopped(self): ...


class BoomError(Exception):
    """What a fake raises when a test tells it to."""


class FakeStream(StreamResourceDataProvider):
    def __init__(self, *args: Any, owner: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.owner = owner

    async def make_stream_docs(self, index):
        if self.owner.fail:
            raise BoomError("make_stream_docs")
        async for doc in super().make_stream_docs(index):
            yield doc


class FakeData(UnboundedDataLogic[Any]):
    """Publishes a `Source` as stream assets.

    :param suffix: the suffix of the hinted datakey
    """

    def __init__(
        self,
        source: Source,
        *,
        suffix: str = "",
        shape_tail: tuple[int, ...] = (),
        dtype_numpy: str = "<i8",
        tmp_path: Any = None,
        hints: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self.source = source
        self.datakey_suffix = suffix
        self.shape_tail = shape_tail
        self.dtype_numpy = dtype_numpy
        self.tmp_path = tmp_path
        self.hints = hints
        self.periods: list[float] = []
        self.starts = 0
        self.stops = 0
        self.fail = False
        self.fail_start = False
        self.sit_out = False
        #: What was done to this logic, in order
        self.log: list[str] = []

    async def start_provider(
        self,
        name: str,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> Any:
        self.periods.append(collection_period)
        if self.sit_out:
            return None
        self.starts += 1
        self.log.append("start")
        if self.fail_start:
            raise BoomError("start")
        datakey_name = name + self.datakey_suffix
        return FakeStream(
            uri=f"file://localhost/{self.tmp_path.as_posix().lstrip('/')}/f.h5",
            resources=[
                StreamResourceInfo(
                    data_key=datakey_name,
                    shape=self.shape_tail,
                    chunk_shape=(1, *self.shape_tail),
                    dtype_numpy=self.dtype_numpy,
                    parameters={"dataset": "/data"},
                )
            ],
            mimetype="application/x-hdf5",
            collections_written=self.source.written,
            collections_per_event=collections_per_event,
            owner=self,
        )

    async def ensure_stopped(self) -> None:
        self.stops += 1
        self.log.append("stop")

    def get_hinted_fields(self, name: str) -> Sequence[str]:
        datakey_name = name + self.datakey_suffix
        return self.hints(datakey_name) if self.hints else [datakey_name]


def make_detector(
    name: str = "det",
    *data: FakeData,
    acquire: DetectorAcquireLogic | None = None,
    trigger: DetectorTriggerLogic | None = None,
    readable: SignalR | None = None,
) -> StandardDetector:
    det = DetectorLogic(
        trigger=trigger or InternalTrigger(), acquire=acquire, data=data
    ).with_device(name)
    if readable is not None:
        det.counts = readable
        det.set_readable_format(readable, Format.HINTED_SIGNAL)
    return det


def run_plan(RE: RunEngine, plan) -> dict[str, list[dict]]:
    docs: dict[str, list[dict]] = defaultdict(list)
    RE(plan, lambda name, doc: docs[name].append(doc))
    return docs


def step_scan(det: StandardDetector, points: int, info: TriggerInfo | None = None):
    @bpp.stage_decorator([det])
    @bpp.run_decorator()
    def plan():
        if info is not None:
            # prepare() comes after stage(), which resets the detector
            yield from bps.prepare(det, info, wait=True)
        for _ in range(points):
            yield from bps.trigger_and_read([det])

    return plan()


def fly_scan(det: StandardDetector, info: TriggerInfo):
    @bpp.stage_decorator([det])
    @bpp.run_decorator()
    def plan():
        yield from bps.prepare(det, info, wait=True)
        yield from bps.declare_stream(det, name="primary", collect=True)
        yield from bps.kickoff(det, wait=True)
        yield from bps.collect_while_completing(
            flyers=[det], dets=[det], flush_period=0.05
        )

    return plan()


# ---------------------------------------------------------------------------
# Use cases 1, 2, 3 and 11, through the RunEngine
# ---------------------------------------------------------------------------


def test_uc1_readables_only_step_scan_needs_no_prepare(RE: RunEngine):
    """Use case 1: no data logic, trigger() prepares implicitly."""
    counts = soft_signal_rw(int, 42, name="det-counts")
    acquire = FillingAcquire()
    det = make_detector("det", acquire=acquire, readable=counts)

    docs = run_plan(RE, step_scan(det, points=2))

    assert [e["data"] for e in docs["event"]] == [{"det-counts": 42}] * 2
    assert set(docs["descriptor"][0]["data_keys"]) == {"det-counts"}
    assert docs["stream_resource"] == []
    assert acquire.idle_waits == 2


def test_uc2_streamable_step_scan_with_readables(RE: RunEngine, tmp_path):
    """Use case 2: one StreamResource, one StreamDatum per point, Events too."""
    source = Source()
    counts = soft_signal_rw(int, 7, name="det-counts")
    det = make_detector(
        "det",
        FakeData(source, tmp_path=tmp_path),
        acquire=FillingAcquire(source),
        readable=counts,
    )

    docs = run_plan(RE, step_scan(det, points=3))

    (resource,) = docs["stream_resource"]
    assert resource["data_key"] == "det"
    assert [
        (d["indices"]["start"], d["indices"]["stop"]) for d in docs["stream_datum"]
    ] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert {d["stream_resource"] for d in docs["stream_datum"]} == {resource["uid"]}
    assert [e["data"] for e in docs["event"]] == [{"det-counts": 7}] * 3
    keys = docs["descriptor"][0]["data_keys"]
    assert keys["det"]["external"] == "STREAM:"
    assert "external" not in keys["det-counts"]


def test_uc3_streamable_fly_scan_makes_assets_and_no_events(RE: RunEngine, tmp_path):
    """Use case 3: stream assets only, covering all N events."""
    source = Source()
    det = make_detector(
        "det",
        FakeData(source, tmp_path=tmp_path),
        acquire=FillingAcquire(source, per_start=6),
    )

    docs = run_plan(
        RE, fly_scan(det, TriggerInfo(number_of_events=3, collections_per_event=2))
    )

    assert docs["event"] == [] and docs["event_page"] == []
    (resource,) = docs["stream_resource"]
    assert resource["data_key"] == "det"
    stops = [d["indices"]["stop"] for d in docs["stream_datum"]]
    assert stops[-1] == 3
    assert docs["stop"][0]["num_events"] == {"primary": 3}


def test_two_runs_in_one_stage_each_have_their_own_stream_resource(
    RE: RunEngine, tmp_path
):
    source = Source()
    data = FakeData(source, tmp_path=tmp_path)
    det = make_detector("det", data, acquire=FillingAcquire(source))

    @bpp.stage_decorator([det])
    def plan():
        for _ in range(2):
            yield from bpp.run_wrapper(
                bpp.pchain(
                    bps.prepare(det, TriggerInfo(), wait=True),
                    bps.trigger_and_read([det]),
                )
            )

    runs: list[dict[str, list[dict]]] = []

    def collect(name: str, doc: dict) -> None:
        if name == "start":
            runs.append(defaultdict(list))
        runs[-1][name].append(doc)

    RE(plan(), collect)

    # The provider carries over, so the same file with the indices carrying on
    assert data.log.count("start") == 1
    for run, index in zip(runs, (0, 1), strict=True):
        (resource,) = run["stream_resource"]
        (datum,) = run["stream_datum"]
        assert datum["stream_resource"] == resource["uid"]
        assert datum["indices"] == {"start": index, "stop": index + 1}
    assert runs[0]["stream_resource"][0]["uid"] != runs[1]["stream_resource"][0]["uid"]


async def test_d2_progress_baseline_is_read_live_for_unbounded_sources(tmp_path):
    """A source that already holds collections (e.g. left running) is not stale."""
    source = Source()
    await source.produce(5)
    det = make_detector(
        "det",
        FakeData(source, tmp_path=tmp_path),
        acquire=FillingAcquire(source, per_start=1),
    )
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()
    await det.trigger()
    # Events since prepare(), from the 5 already there
    assert await det.get_index() == 2


# ---------------------------------------------------------------------------
# A data logic is started once a stage while its provider can carry over (D3)
# ---------------------------------------------------------------------------


async def test_d3_unbounded_logic_is_never_restarted_and_fly_never_rearms(tmp_path):
    source = Source()
    data = FakeData(source, tmp_path=tmp_path)
    det = make_detector("det", data, acquire=FillingAcquire(source, per_start=2))
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=2))
    await det.kickoff()
    await det.complete()
    assert data.starts == 1
    await det.prepare(TriggerInfo())
    await det.trigger()
    await det.trigger()
    # The provider carried over to the second prepare
    assert data.starts == 1


async def test_d3_fly_scan_with_an_unbounded_source_non_zero_at_kickoff(tmp_path):
    source = Source()
    await source.produce(4)
    det = make_detector(
        "det",
        FakeData(source, tmp_path=tmp_path),
        acquire=FillingAcquire(source, per_start=3),
    )
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=3))
    await det.kickoff()
    await det.complete()
    # Events since prepare()
    assert await det.get_index() == 3
    docs = [doc async for doc in det.collect_asset_docs()]
    # The StreamDatum indexes the events in the file, after the 4 already there
    assert [doc["indices"] for name, doc in docs if name == "stream_datum"] == [
        {"start": 4, "stop": 7}
    ]


# ---------------------------------------------------------------------------
# Lifecycle state machine (spec §3), every verb from every state
# ---------------------------------------------------------------------------

STATES = ["U", "S", "P", "T", "K", "C"]


class GatedAcquire(FillingAcquire):
    """Runs acquire() to its end only once `done` is set, so a fly stays FLYING."""

    def __init__(self, *sources: Source) -> None:
        super().__init__(*sources)
        self.done = asyncio.Event()

    async def acquire(self):
        async for _ in super().acquire():
            yield
        await self.done.wait()


async def reach(state: str, tmp_path) -> StandardDetector:
    source = Source()
    acquire = GatedAcquire(source)
    det = make_detector(
        "det",
        FakeData(source, tmp_path=tmp_path),
        acquire=acquire,
    )
    # "K" is kicked off with the fly still running
    if state != "K":
        acquire.done.set()
    if state == "U":
        return det
    await det.stage()
    if state == "S":
        return det
    await det.prepare(TriggerInfo())
    if state == "T":
        await det.trigger()
    elif state in "KC":
        await det.kickoff()
        if state == "C":
            await det.complete()
    return det


async def call(det: StandardDetector, verb: str) -> None:
    if verb == "prepare":
        await det.prepare(TriggerInfo())
    elif verb == "complete":
        # Let a running fly finish
        cast(GatedAcquire, det.logic.acquire).done.set()
        await asyncio.wait_for(det.complete(), 2)
    elif verb == "collect_asset_docs":
        _ = [doc async for doc in det.collect_asset_docs()]
    else:
        await asyncio.wait_for(getattr(det, verb)(), 2)


# (state, verb) -> does the verb succeed. From the §3 table.
TABLE: dict[tuple[str, str], bool] = {}
for _verb, _row in {
    "prepare": "FTFTFT",
    "trigger": "FTTTFF",
    "kickoff": "FFTFFF",
    "complete": "FFFFTT",
    "read": "FFFTFF",
    "describe": "FFTTTT",
    "describe_collect": "FFTTTT",
    "get_index": "FFTTTT",
}.items():
    for _state, _ok in zip(STATES, _row, strict=True):
        TABLE[(_state, _verb)] = _ok == "T"
for _state in "PTKC":
    # Allowed after a trigger() too: bluesky's read() collects stream assets
    TABLE[(_state, "collect_asset_docs")] = True


# What each verb that fails from a state says, as the start of the message
MESSAGES: dict[tuple[str, str], str] = {
    ("U", "prepare"): "cannot go to PREPARING from UNSTAGED",
    ("P", "prepare"): "cannot go to PREPARING from PREPARED",
    ("K", "prepare"): "cannot go to PREPARING from FLYING",
    ("U", "trigger"): "cannot go to TRIGGERING from UNSTAGED",
    ("K", "trigger"): "cannot go to TRIGGERING from FLYING",
    ("C", "trigger"): "cannot go to TRIGGERING from FLOWN",
    ("U", "kickoff"): "cannot go to FLYING from UNSTAGED",
    ("S", "kickoff"): "cannot go to FLYING from STAGED",
    ("T", "kickoff"): "cannot go to FLYING from TRIGGERED",
    ("K", "kickoff"): "cannot go to FLYING from FLYING",
    ("C", "kickoff"): "cannot go to FLYING from FLOWN",
    ("U", "complete"): "cannot complete from UNSTAGED, allowed only after kickoff()",
    ("S", "complete"): "cannot complete from STAGED, allowed only after kickoff()",
    ("P", "complete"): "cannot complete from PREPARED, allowed only after kickoff()",
    ("T", "complete"): "cannot complete from TRIGGERED, allowed only after kickoff()",
    **{
        (state, "read"): f"det: read not allowed from {name}, allowed from: TRIGGERED"
        for state, name in {
            "U": "UNSTAGED",
            "S": "STAGED",
            "P": "PREPARED",
            "K": "FLYING",
            "C": "FLOWN",
        }.items()
    },
    **{
        (state, verb): f"det: {verb} needs a successful prepare() first"
        for state in "US"
        for verb in ("describe", "describe_collect", "get_index")
    },
}


def _rows():
    for (state, verb), ok in TABLE.items():
        id = f"{state}-{verb}"
        yield pytest.param(state, verb, ok, id=id)


@pytest.mark.parametrize("state,verb,ok", list(_rows()))
async def test_lifecycle_verb_from_state(state, verb, ok, tmp_path):
    det = await reach(state, tmp_path)
    if ok:
        await call(det, verb)
    else:
        with pytest.raises(RuntimeError, match=re.escape(MESSAGES[(state, verb)])):
            await call(det, verb)
    # Ends a fly still running in "K"
    await det.unstage()


@pytest.mark.parametrize("state", STATES)
async def test_restage_returns_to_staged_from_any_state(state, tmp_path):
    det = await reach(state, tmp_path)
    await det.unstage()
    await det.stage()
    with pytest.raises(TransitionError, match="describe needs a successful prepare"):
        await det.describe()
    await det.prepare(TriggerInfo())
    await det.describe()


@pytest.mark.parametrize("state", STATES)
async def test_unstage_from_any_state_is_never_an_error(state, tmp_path):
    det = await reach(state, tmp_path)
    await det.unstage()
    await det.unstage()


@pytest.mark.parametrize("state", ["S", "P", "T", "K", "C"])
async def test_unstage_leads_to_unstaged(state, tmp_path):
    det = await reach(state, tmp_path)
    await det.unstage()
    with pytest.raises(RuntimeError, match="from UNSTAGED"):
        await det.prepare(TriggerInfo())


class _Flyer(StandardFlyable[int, None]):
    @cached_property
    def logic(self) -> FlyableLogic[int, None]:
        return _NoopFlyableLogic()


class _NoopFlyableLogic(FlyableLogic[int, None]):
    async def prepare_to_fly(
        self, name: str, value: int, previous_ctx: None
    ) -> AsyncIterator[None]:
        yield None

    async def fly(self, ctx: None) -> AsyncIterator[KICKED_OFF]:
        yield KICKED_OFF

    async def ensure_stopped(self) -> None: ...


@pytest.mark.parametrize("verb", ["prepare", "kickoff", "complete"])
async def test_r10_standard_flyable_verbs_raise_before_stage(verb: str):
    flyer = _Flyer(name="flyer")
    with pytest.raises(RuntimeError, match="from UNSTAGED"):
        await (flyer.prepare(1) if verb == "prepare" else getattr(flyer, verb)())
    await flyer.stage()
    await flyer.prepare(1)
    await flyer.unstage()
    await flyer.unstage()  # a no-op
    with pytest.raises(RuntimeError, match="from UNSTAGED"):
        await flyer.prepare(1)


async def test_trigger_after_complete_says_to_prepare_again(tmp_path):
    det = await reach("C", tmp_path)
    with pytest.raises(
        TransitionError,
        match="cannot go to TRIGGERING from FLOWN, allowed from: PREPARED, TRIGGERED",
    ):
        await det.trigger()


async def test_external_trigger_second_trigger_without_prepare_raises():
    source = Source()
    det = make_detector(
        "det", trigger=ExternalTrigger(), acquire=FillingAcquire(source)
    )
    await det.stage()
    await det.prepare(
        TriggerInfo(trigger=DetectorTrigger.EXTERNAL_EDGE, exposure_timeout=0.1)
    )
    await det.trigger()
    with pytest.raises(RuntimeError, match="prepare"):
        await det.trigger()


async def test_internal_trigger_repeats_on_one_prepare():
    acquire = FillingAcquire()
    det = make_detector("det", acquire=acquire)
    await det.stage()
    await det.prepare(TriggerInfo())
    for _ in range(5):
        await det.trigger()
    assert acquire.idle_waits == 5


class HeldAcquire(DetectorAcquireLogic):
    """Arms with a status that only ensure_stopped() finishes, like a driver."""

    def __init__(self) -> None:
        self.statuses: list[AsyncStatus] = []
        self.closed = 0
        self.armed = asyncio.Event()
        self._stopped = asyncio.Event()

    async def ensure_ready(self):
        self._stopped.clear()

    async def acquire(self):
        try:
            async with AsyncStatus(self._stopped.wait()) as status:
                self.statuses.append(status)
                self.armed.set()
                yield
                await status
        finally:
            self.closed += 1

    async def ensure_stopped(self):
        self._stopped.set()


async def test_unstage_closes_the_acquisition_an_external_prepare_armed():
    acquire = HeldAcquire()
    det = make_detector("det", trigger=ExternalTrigger(), acquire=acquire)
    await det.stage()
    await det.prepare(TriggerInfo(trigger=DetectorTrigger.EXTERNAL_EDGE))
    assert acquire.armed.is_set()
    assert acquire.closed == 0

    await det.unstage()
    assert acquire.closed == 1
    assert acquire.statuses[0].done


class ArmLoggingAcquire(FillingAcquire):
    """Writes "arm" to `log` once armed."""

    def __init__(self, log: list[str], *sources: Source) -> None:
        super().__init__(*sources)
        self.log = log

    async def acquire(self):
        async for _ in super().acquire():
            self.log.append("arm")
            yield


async def test_an_external_prepare_starts_the_data_before_arming(tmp_path):
    source = Source()
    data = FakeData(source, tmp_path=tmp_path)
    det = make_detector(
        "det",
        data,
        trigger=ExternalTrigger(),
        acquire=ArmLoggingAcquire(data.log, source),
    )
    await det.stage()
    data.log.clear()

    await det.prepare(TriggerInfo(trigger=DetectorTrigger.EXTERNAL_EDGE))
    assert data.log == ["start", "arm"]
    await det.unstage()


async def test_the_last_prepare_is_forgotten_at_stage_not_unstage(tmp_path):
    source = Source()
    serving = FakeData(source, tmp_path=tmp_path)
    sitting_out = FakeData(source, suffix="-b", tmp_path=tmp_path)
    sitting_out.sit_out = True
    det = make_detector("det", serving, sitting_out)
    # Before a prepare, every data logic's fields are hinted
    await det.stage()
    assert det.hints == {"fields": ["det", "det-b"]}

    await det.prepare(TriggerInfo())
    assert det.hints == {"fields": ["det"]}
    await det.unstage()
    assert det.hints == {"fields": ["det"]}

    await det.stage()
    assert det.hints == {"fields": ["det", "det-b"]}


class StopData(FakeData):
    """Its `ensure_stopped` raises if `stop_mode` is "fail", runs until
    cancelled if "hang"."""

    stop_mode = ""
    stopping = False

    async def ensure_stopped(self) -> None:
        await super().ensure_stopped()
        if self.stop_mode == "fail":
            raise BoomError("stop")
        self.stopping = self.stop_mode == "hang"
        try:
            if self.stopping:
                await asyncio.Event().wait()
        finally:
            self.stopping = False


# A prepare() for a different scan restarts the data logics
@pytest.mark.parametrize("verb", ["stage", "prepare", "unstage"])
async def test_a_failed_stop_leaves_no_other_stop_running(verb: str, tmp_path):
    source, other_source = Source(), Source()
    failing = StopData(source, tmp_path=tmp_path)
    hanging = StopData(other_source, suffix="-h", tmp_path=tmp_path)
    det = make_detector(
        "det", failing, hanging, acquire=FillingAcquire(source, other_source)
    )
    if verb != "stage":
        await det.stage()
        await det.prepare(TriggerInfo())
        await det.trigger()
    failing.stop_mode, hanging.stop_mode = "fail", "hang"

    with pytest.raises(BoomError, match="stop"):
        if verb == "prepare":
            await det.prepare(TriggerInfo(livetime=0.1))
        else:
            await getattr(det, verb)()

    assert not hanging.stopping


async def test_unstage_cancels_a_running_trigger_that_kickoff_cannot_overlap(
    tmp_path,
):
    acquire = HeldAcquire()
    det = make_detector("det", FakeData(Source(), tmp_path=tmp_path), acquire=acquire)
    await det.stage()
    # Prepares implicitly, arms, then waits for data that never comes
    status = det.trigger()
    await acquire.armed.wait()
    with pytest.raises(RuntimeError, match="cannot go to FLYING from TRIGGERING"):
        await det.kickoff()
    await det.unstage()
    with pytest.raises(asyncio.CancelledError):
        await status
    assert acquire.closed == 1
    assert acquire.statuses[0].done


# ---------------------------------------------------------------------------
# Use case 11
# ---------------------------------------------------------------------------


async def test_a_data_logic_that_served_the_stage_must_serve_all_of_it(tmp_path):
    source = Source()
    data = FakeData(source, tmp_path=tmp_path)
    det = make_detector("det", data, acquire=FillingAcquire(source))
    await det.stage()
    await det.trigger()

    data.sit_out = True
    # A prepare() for a different scan starts the logic again
    with pytest.raises(RuntimeError, match=r"made \[\] at .*, but \['det'\] before it"):
        await det.prepare(TriggerInfo(collections_per_event=2))


async def test_uc11_no_data_no_readables_steps_and_flies():
    acquire = FillingAcquire()
    det = make_detector("det", acquire=acquire)
    await det.stage()

    await det.trigger()
    await det.trigger()
    assert acquire.idle_waits == 2
    assert await det.read() == {}

    # A fly has nothing to collect, so completes once acquisition has run
    await det.prepare(TriggerInfo(number_of_events=3))
    await det.kickoff()
    await det.complete()
    assert acquire.idle_waits == 3
    assert [doc async for doc in det.collect_asset_docs()] == []


# ---------------------------------------------------------------------------
# Data-set rules (§5)
# ---------------------------------------------------------------------------


def _reading_detector(suppressor: FakeData, source: Source, value: Any):
    reading = soft_signal_rw(Array1D[np.int64], value, name="det-reading")
    return make_detector(
        "det", suppressor, acquire=FillingAcquire(source, per_start=3), readable=None
    ), reading


async def test_readable_is_suppressed_by_streamable_keys(tmp_path):
    source = Source()
    data = FakeData(
        source,
        suffix="-reading",
        shape_tail=(),
        dtype_numpy="<i8",
        tmp_path=tmp_path,
    )
    det, reading = _reading_detector(data, source, np.array([9, 9, 9], dtype=np.int64))
    det.reading = reading
    det.set_readable_format(reading, Format.HINTED_SIGNAL)
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=3))
    await det.trigger()

    description = await det.describe()

    assert list(description) == ["det-reading"]
    assert await det.read() == {}


async def test_same_name_different_shape_is_a_duplicate_not_a_suppression(tmp_path):
    source = Source()
    data = FakeData(source, suffix="-reading", tmp_path=tmp_path)
    det, reading = _reading_detector(data, source, np.array([9, 9], dtype=np.int64))
    det.reading = reading
    det.set_readable_format(reading, Format.HINTED_SIGNAL)
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=3))

    with pytest.raises(ValueError, match=r"duplicate datakeys \['det-reading'\]"):
        await det.describe()


async def test_u9_a_suppressed_readable_keeps_its_hint(tmp_path):
    source = Source()
    data = FakeData(source, suffix="-reading", hints=lambda _: [], tmp_path=tmp_path)
    det, reading = _reading_detector(data, source, np.array([9, 9, 9], dtype=np.int64))
    det.reading = reading
    det.set_readable_format(reading, Format.HINTED_SIGNAL)
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=3))

    await det.describe()
    # The data that replaces the signal has its name, so the signal's hint stands
    assert det.hints == {"fields": ["det-reading"]}


async def test_u9_a_suppressed_readable_and_its_suppressor_both_hinting_is_an_error(
    tmp_path,
):
    source = Source()
    data = FakeData(
        source, suffix="-reading", hints=lambda name: [name], tmp_path=tmp_path
    )
    det, reading = _reading_detector(data, source, np.array([9, 9, 9], dtype=np.int64))
    det.reading = reading
    det.set_readable_format(reading, Format.HINTED_SIGNAL)
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=3))

    with pytest.raises(
        RuntimeError, match=r"Hint 'fields' \['det-reading'\] override existing hint"
    ):
        det.hints  # noqa: B018


async def test_u7_duplicate_datakeys_raise_at_prepare(tmp_path):
    det = make_detector(
        "det",
        FakeData(Source(), tmp_path=tmp_path),
        FakeData(Source(), tmp_path=tmp_path),
        acquire=FillingAcquire(),
    )
    await det.stage()
    with pytest.raises(
        RuntimeError,
        match=r"^More than one data logic makes the same datakey: "
        r"\{'det': \['FakeData', 'FakeData'\]\}\. Each datakey must come from "
        r"one data logic, so enable only one of them, or give them different "
        r"datakeys$",
    ):
        await det.prepare(TriggerInfo())


# ---------------------------------------------------------------------------
# U12: hardware read-back is gated by OPHYD_ASYNC_PRESERVE_DETECTOR_STATE
# ---------------------------------------------------------------------------


class ReadbackTrigger(InternalTrigger):
    async def default_trigger_info(self) -> TriggerInfo:
        return TriggerInfo(livetime=0.1, deadtime=0.02)


@pytest.mark.parametrize(
    "env,expected_period",
    [
        ("YES", pytest.approx(0.12)),
        (None, 0.0),
    ],
)
async def test_u12_zero_livetime_readback_is_gated_by_the_env_var(
    monkeypatch: pytest.MonkeyPatch, env, expected_period, tmp_path
):
    if env is None:
        monkeypatch.delenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", raising=False)
    else:
        monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", env)
    data = FakeData(Source(), tmp_path=tmp_path)
    det = make_detector(
        "det", data, trigger=ReadbackTrigger(), acquire=FillingAcquire()
    )
    await det.stage()

    await det.prepare(TriggerInfo(number_of_events=1))

    assert data.periods == [expected_period]


# ---------------------------------------------------------------------------
# Verifier findings 2 and 4 to 6: re-asking, notes, failed prepare and start
# ---------------------------------------------------------------------------


async def test_an_unbounded_provider_carries_over_without_its_logic_being_asked(
    tmp_path,
):
    source = Source()
    unbounded = FakeData(source, tmp_path=tmp_path)
    det = make_detector("det", unbounded, acquire=FillingAcquire(source))
    await det.stage()
    unbounded.log.clear()
    info = TriggerInfo(livetime=0.1, deadtime=0.01)
    await det.prepare(info)
    await det.trigger()
    await det.prepare(info)
    await det.trigger()
    await det.prepare(info)

    assert unbounded.log == ["start"]
    assert set(await det.describe_collect()) == {"det"}


class NewFileData(FakeData):
    """A file writer: each start opens a new file, counting from 0."""

    async def start_provider(self, *args: Any, **kwargs: Any) -> Any:
        await self.source.rearm()
        return await super().start_provider(*args, **kwargs)


@pytest.mark.parametrize(
    "extra, info, indices",
    [
        # Run 1 wrote one whole event in each file, so run 2 carries on after it
        ((0, 0), TriggerInfo(collections_per_event=2), {"start": 1, "stop": 2}),
        # Half an event more in each file
        ((1, 1), TriggerInfo(collections_per_event=2), {"start": 0, "stop": 1}),
        # A whole event more in one file only
        ((2, 0), TriggerInfo(collections_per_event=2), {"start": 0, "stop": 1}),
        # Run 2 takes a different number of collections per event
        (
            (0, 0),
            TriggerInfo(collections_per_event=1, number_of_events=2),
            {"start": 0, "stop": 2},
        ),
    ],
    ids=["carried-over", "ragged", "different-origins", "changed-cpe"],
)
async def test_unbounded_providers_carry_over_only_from_a_whole_shared_event(
    extra: tuple[int, int], info: TriggerInfo, indices: dict[str, int], tmp_path
):
    sources = Source(), Source()
    det = make_detector(
        "det",
        NewFileData(sources[0], tmp_path=tmp_path),
        NewFileData(sources[1], tmp_path=tmp_path, suffix="-s"),
        acquire=FillingAcquire(*sources, per_start=2),
    )
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=2))
    await det.kickoff()
    await det.complete()
    # Frames written between the runs, say by an external trigger
    for source, frames in zip(sources, extra, strict=True):
        await source.produce(frames)
    await det.prepare(info)
    await det.kickoff()
    await det.complete()

    docs = [doc async for doc in det.collect_asset_docs()]
    assert [doc["indices"] for name, doc in docs if name == "stream_datum"] == [
        indices,
        indices,
    ]


async def test_a_changed_scan_stops_the_old_provider_before_making_the_new(tmp_path):
    sources = Source(), Source()
    unbounded = FakeData(sources[0], tmp_path=tmp_path)
    steady = FakeData(sources[1], tmp_path=tmp_path, suffix="-s")
    det = make_detector(
        "det", unbounded, steady, acquire=FillingAcquire(*sources, per_start=2)
    )
    await det.stage()
    for data in (
        unbounded,
        steady,
    ):
        data.log.clear()
    await det.prepare(TriggerInfo(livetime=0.1))
    assert unbounded.log == steady.log == ["start"]

    # Each of what a provider is made for, in turn
    for changed in (
        TriggerInfo(livetime=0.2),
        TriggerInfo(livetime=0.2, flush_period=2.0),
        TriggerInfo(livetime=0.2, flush_period=2.0, collections_per_event=2),
    ):
        await det.trigger()
        unbounded.log.clear()
        steady.log.clear()
        await det.prepare(changed)
        assert unbounded.log == ["stop", "start"]
        assert steady.log == ["stop", "start"]


async def test_a_settings_change_takes_effect_at_the_next_stage_not_prepare(tmp_path):
    # Stands for a plugin's EnableCallbacks being switched off in the Settings
    source = Source()
    plugin = FakeData(source, tmp_path=tmp_path)
    det = make_detector("det", plugin, acquire=FillingAcquire(source))
    await det.stage()
    info = TriggerInfo()
    await det.prepare(info)
    await det.trigger()

    plugin.sit_out = True
    await det.prepare(info)
    assert set(await det.describe_collect()) == {"det"}

    await det.unstage()
    assert plugin.log[-1] == "stop"
    await det.stage()
    await det.prepare(info)
    assert await det.describe_collect() == {}


@pytest.mark.parametrize(
    "verb",
    [
        "stage",
        "unstage",
        "trigger",
        "kickoff",
        "complete",
        "collect_asset_docs",
    ],
)
async def test_f4_exceptions_from_logic_propagate(verb: str, tmp_path):
    class Failing(FillingAcquire):
        failing = ""

        async def ensure_ready(self):
            if self.failing == "stage":
                raise BoomError(self.failing)

        async def ensure_stopped(self):
            if self.failing == "unstage":
                raise BoomError(self.failing)

        async def acquire(self):
            if self.failing in ("trigger", "kickoff"):
                raise BoomError(self.failing)
            async for _ in super().acquire():
                yield
            if self.failing == "complete":
                raise BoomError(self.failing)

    source = Source()
    acquire = Failing(source)
    data = FakeData(source, tmp_path=tmp_path)
    det = make_detector("det", data, acquire=acquire)
    # Before kickoff, as fly() runs on to the end of acquire() as soon as it can
    acquire.failing = verb
    if verb != "stage":
        await det.stage()
    if verb not in ("stage", "unstage"):
        await det.prepare(TriggerInfo(number_of_events=1))
    if verb in ("complete", "collect_asset_docs"):
        await det.kickoff()
    data.fail = verb == "collect_asset_docs"

    with pytest.raises(BoomError):
        if verb == "collect_asset_docs":
            _ = [doc async for doc in det.collect_asset_docs()]
        else:
            await getattr(det, verb)()


class YieldsTwice(FillingAcquire):
    async def acquire(self):
        yield
        yield


async def test_acquire_that_yields_twice_raises():
    det = make_detector("det", acquire=YieldsTwice())
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=1))

    with pytest.raises(RuntimeError, match=r"YieldsTwice\.acquire\(\) must yield only"):
        await det.trigger()


@pytest.mark.parametrize("failing_index", [0, 1])
async def test_prepare_after_a_failed_start_is_a_first_prepare(
    failing_index: int, tmp_path
):
    sources = Source(), Source()
    data = [
        FakeData(sources[0], tmp_path=tmp_path),
        FakeData(sources[1], tmp_path=tmp_path, suffix="-b"),
    ]
    failing = data[failing_index]
    det = make_detector("det", *data, acquire=FillingAcquire(*sources))
    await det.stage()
    failing.fail_start = True
    with pytest.raises(BoomError, match="start"):
        await det.prepare(TriggerInfo())

    failing.fail_start = False
    await det.unstage()
    await det.stage()
    await det.prepare(TriggerInfo())

    assert list(await det.describe_collect()) == ["det", "det-b"]


@pytest.mark.parametrize("cpe", [1, 3])
async def test_scalar_readable_is_suppressed_by_a_key_of_any_collections_per_event(
    cpe: int, tmp_path
):
    """A readable gives one value per event, so [cpe] suppresses it, not just []."""
    source = Source()
    data = FakeData(source, suffix="-reading", tmp_path=tmp_path)
    det = make_detector("det", data, acquire=FillingAcquire(source, per_start=cpe))
    det.reading = soft_signal_rw(int, 9, name="det-reading")
    det.set_readable_format(det.reading, Format.HINTED_SIGNAL)
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=cpe))
    await det.trigger()

    description = await det.describe()

    assert description["det-reading"]["shape"] == [cpe]
    assert await det.read() == {}


def test_data_logic_with_the_removed_stop_raises():
    with pytest.raises(TypeError, match=r"StaleData defines stop, which .* -> ensure"):
        type("StaleData", (UnboundedDataLogic,), {"stop": lambda self: None})
