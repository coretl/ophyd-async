"""Use-case tests for the detector core (spec: detector-logic-spec.md).

Written from the spec for the detector-core rewrite. The strict xfails are
behaviour the spec wants that something outside ophyd-async still prevents.

The fakes in this file stand in for the three kinds of device a detector drives:
a device-side `Source` (counter plus data), an acquire logic that fills it, and
data logics that publish it as readables, event pages or stream assets.
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
from event_model import DataKey
from event_model.documents import PartialEventPage

from ophyd_async.core import (
    KICKED_OFF,
    Array1D,
    AsyncStatus,
    BoundedDataLogic,
    DetectorAcquireLogic,
    DetectorDataLogic,
    DetectorLogic,
    DetectorTrigger,
    DetectorTriggerLogic,
    FlyableLogic,
    PageableDataProvider,
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


class FakePageable(PageableDataProvider):
    """Event pages straight from a source, one value per collection.

    Every datakey holds the same values. `make_page` carries on from the last
    page.
    """

    def __init__(
        self,
        datakey_names: Sequence[str],
        source: Source,
        collections_per_event: int,
        shape_tail: tuple[int, ...] = (),
        dtype_numpy: str = "<i8",
        owner: Any = None,
    ) -> None:
        super().__init__(collections_per_event)
        self.datakey_names = datakey_names
        self.source = source
        self.owner = owner
        self.shape_tail = shape_tail
        self.dtype_numpy = dtype_numpy
        self.start_new_resources(0)

    def start_new_resources(self, origin: int) -> None:
        self.origin = origin
        self.published = 0

    async def make_datakeys(self) -> dict[str, DataKey]:
        return {
            datakey_name: DataKey(
                source="fake",
                shape=[self.collections_per_event, *self.shape_tail],
                dtype="array",
                dtype_numpy=self.dtype_numpy,
            )
            for datakey_name in self.datakey_names
        }

    def _page(self, first: int, last: int) -> PartialEventPage:
        cpe = self.collections_per_event
        events = range(first, last)
        starts = [self.origin + e * cpe for e in events]
        values = [self.source.data[start : start + cpe] for start in starts]
        times = [float(e) for e in events]
        return {
            "data": dict.fromkeys(self.datakey_names, values),
            "time": times,
            "timestamps": dict.fromkeys(self.datakey_names, times),
        }

    async def make_page(self, index: int) -> PartialEventPage | None:
        if self.owner and self.owner.fail:
            raise BoomError("make_page")
        if index <= self.published:
            return None
        page = self._page(self.published, index)
        self.published = index
        return page


class _FakeData:
    """Publishes a `Source` as event pages or stream assets.

    :param bounded: a finite buffer: it needs a finite N (raises for N=0 if it
        has datakeys left to make) and is cleared at each start
    :param suffix: the suffix of the hinted datakey
    :param also: the suffixes of further datakeys holding the same data
    """

    bounded: bool

    def __init__(
        self,
        source: Source,
        *,
        publication: str,
        suffix: str = "",
        also: Sequence[str] = (),
        shape_tail: tuple[int, ...] = (),
        dtype_numpy: str = "<i8",
        tmp_path: Any = None,
        hints: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self.source = source
        self.publication = publication
        self.datakey_suffix = suffix
        self.also = also
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

    @property
    def collections_written(self) -> SignalR[int]:
        return self.source.written

    async def _start(
        self,
        name: str,
        existing_datakeys: Sequence[str],
        num_collections: int,
        collections_per_event: int,
        collection_period: float,
    ) -> Any:
        self.periods.append(collection_period)
        datakey_names = [
            datakey_name
            for datakey_name in (
                name + suffix for suffix in (self.datakey_suffix, *self.also)
            )
            if datakey_name not in existing_datakeys
        ]
        if self.sit_out or not datakey_names:
            return None
        self.starts += 1
        self.log.append("start")
        if self.fail_start:
            raise BoomError("start")
        if self.bounded:
            # A finite buffer is armed empty
            await self.source.rearm()
        if self.publication == "stream":
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
                    for datakey_name in datakey_names
                ],
                mimetype="application/x-hdf5",
                collections_per_event=collections_per_event,
                owner=self,
            )
        return FakePageable(
            datakey_names,
            self.source,
            collections_per_event,
            shape_tail=self.shape_tail,
            dtype_numpy=self.dtype_numpy,
            owner=self,
        )

    async def ensure_stopped(self) -> None:
        self.stops += 1
        self.log.append("stop")

    def get_hinted_fields(self, name: str) -> Sequence[str]:
        datakey_name = name + self.datakey_suffix
        return self.hints(datakey_name) if self.hints else [datakey_name]


class FakeBoundedData(_FakeData, BoundedDataLogic[Any]):
    bounded = True

    async def start_provider(
        self,
        name: str,
        existing_datakeys: Sequence[str],
        num_collections: int,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> Any:
        return await self._start(
            name,
            existing_datakeys,
            num_collections,
            collections_per_event,
            collection_period,
        )


class FakeUnboundedData(_FakeData, UnboundedDataLogic[Any]):
    bounded = False

    async def start_provider(
        self,
        name: str,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> Any:
        # An unbounded source takes data for as long as it is told to
        return await self._start(name, (), 1, collections_per_event, collection_period)


def FakeData(  # noqa: N802
    source: Source, *, bounded: bool, **kwargs: Any
) -> FakeBoundedData | FakeUnboundedData:
    return (FakeBoundedData if bounded else FakeUnboundedData)(source, **kwargs)


def make_detector(
    name: str = "det",
    *data: _FakeData,
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
# Use cases 1, 2, 3, 5, 11 and the four quadrants of D2, through the RunEngine
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
        FakeData(source, publication="stream", bounded=False, tmp_path=tmp_path),
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
        FakeData(source, publication="stream", bounded=False, tmp_path=tmp_path),
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
    data = FakeData(source, publication="stream", bounded=False, tmp_path=tmp_path)
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


def test_uc5_pageable_step_scan_with_several_collections_per_event(RE: RunEngine):
    """Use case 5 (and 4's buffer): one array reading of shape [N] per point.

    The finite buffer is cleared between points, so each point holds its own N.
    """
    source = Source()
    det = make_detector(
        "det",
        FakeData(source, publication="page", bounded=True),
        acquire=FillingAcquire(source, per_start=3),
    )

    docs = run_plan(RE, step_scan(det, 2, TriggerInfo(collections_per_event=3)))

    assert docs["descriptor"][0]["data_keys"]["det"]["shape"] == [3]
    assert [e["data"]["det"] for e in docs["event"]] == [[0, 1, 2], [3, 4, 5]]


def test_uc4_pageable_fly_scan_collects_event_pages(RE: RunEngine):
    """Use case 4: N events come out of collect_pages as event pages."""
    source = Source()
    det = make_detector(
        "det",
        FakeData(source, publication="page", bounded=True),
        acquire=FillingAcquire(source, per_start=4),
    )

    docs = run_plan(RE, fly_scan(det, TriggerInfo(number_of_events=4)))

    (page,) = docs["event_page"]
    assert page["data"]["det"] == [[0], [1], [2], [3]]
    assert docs["stop"][0]["num_events"] == {"primary": 4}


@pytest.mark.parametrize(
    "publication,bounded",
    [
        ("stream", False),
        ("stream", True),
        ("page", True),
        ("page", False),
    ],
)
def test_d2_all_four_bound_by_publication_quadrants_step_scan(
    RE: RunEngine, tmp_path, publication: str, bounded: bool
):
    """Bound and publication are separate axes (D2): every quadrant step-scans.

    An unbounded pageable source (a monitor-fed accumulator) is never re-armed
    and its count keeps rising, so the progress baseline of each point must be
    read live; a baseline of 0 reads stale data.
    """
    source = Source()
    det = make_detector(
        "det",
        FakeData(source, publication=publication, bounded=bounded, tmp_path=tmp_path),
        acquire=FillingAcquire(source, per_start=1),
    )

    docs = run_plan(RE, step_scan(det, 3))

    if publication == "page":
        assert [e["data"]["det"] for e in docs["event"]] == [[0], [1], [2]]
    else:
        # A bounded provider is re-made at each point, so it starts a new stream
        # resource each time; an unbounded one carries on in the same one
        stops = [1, 1, 1] if bounded else [1, 2, 3]
        assert [d["indices"]["stop"] for d in docs["stream_datum"]] == stops
        assert len(docs["stream_resource"]) == (3 if bounded else 1)
        assert docs["event"][0]["data"] == {}


@pytest.mark.parametrize("publication", ["stream", "page"])
async def test_d2_progress_baseline_is_read_live_for_unbounded_sources(
    publication: str, tmp_path
):
    """A source that already holds collections (e.g. left running) is not stale."""
    source = Source()
    await source.produce(5)
    det = make_detector(
        "det",
        FakeData(source, publication=publication, bounded=False, tmp_path=tmp_path),
        acquire=FillingAcquire(source, per_start=1),
    )
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()
    await det.trigger()
    # Events since prepare(), from the 5 already there
    assert await det.get_index() == 2


# ---------------------------------------------------------------------------
# Bounded logics are re-armed at each step point after the first (D2, D3)
# ---------------------------------------------------------------------------


async def test_d3_bounded_logic_is_restarted_at_each_point_after_the_first():
    source = Source()
    data = FakeData(source, publication="page", bounded=True)
    det = make_detector("det", data, acquire=FillingAcquire(source))
    await det.stage()
    await det.prepare(TriggerInfo())

    await det.trigger()
    assert data.starts == 1
    await det.trigger()
    await det.trigger()
    assert data.starts == 3


async def test_d3_step_scan_index_counts_since_prepare_across_points(tmp_path):
    """get_index counts every point; a bounded provider only holds its own."""
    bounded_source, open_source = Source(), Source()
    det = make_detector(
        "det",
        FakeData(
            bounded_source,
            publication="stream",
            bounded=True,
            suffix="-b",
            tmp_path=tmp_path,
        ),
        FakeData(open_source, publication="stream", bounded=False, tmp_path=tmp_path),
        acquire=FillingAcquire(bounded_source, open_source),
    )
    await det.stage()
    await det.prepare(TriggerInfo())
    points = []
    for _ in range(3):
        await det.trigger()
        index = await det.get_index()
        docs = [d async for d in det.collect_asset_docs(index)]  # type: ignore[attr-defined]
        points.append(
            (
                index,
                [d["data_key"] for name, d in docs if name == "stream_resource"],
                [
                    (d["indices"]["start"], d["indices"]["stop"])
                    for name, d in docs
                    if name == "stream_datum"
                ],
            )
        )
    # The bounded provider is restarted at each point, so has a new resource
    # holding one event, while the unbounded one carries on in its first
    assert points == [
        (1, ["det-b", "det"], [(0, 1), (0, 1)]),
        (2, ["det-b"], [(0, 1), (1, 2)]),
        (3, ["det-b"], [(0, 1), (2, 3)]),
    ]


async def test_d3_unbounded_logic_is_never_restarted_and_fly_never_rearms():
    source = Source()
    bounded = FakeData(source, publication="page", bounded=True)
    other = Source()
    unbounded = FakeData(other, publication="page", bounded=False, suffix="-u")
    det = make_detector(
        "det", bounded, unbounded, acquire=FillingAcquire(source, other, per_start=2)
    )
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=2))
    await det.kickoff()
    await det.complete()
    assert (bounded.starts, unbounded.starts) == (1, 1)
    await det.prepare(TriggerInfo())
    await det.trigger()
    await det.trigger()
    # The unbounded provider carried over to the second prepare
    assert (bounded.starts, unbounded.starts) == (3, 1)


async def test_d3_mixed_bounded_and_unbounded_pageable_steps_repeatedly():
    """One baseline per bound group: the bounded one resets, the other counts on."""
    bounded_source, open_source = Source(), Source()
    await open_source.produce(3)
    bounded = FakeData(bounded_source, publication="page", bounded=True, suffix="-b")
    unbounded = FakeData(open_source, publication="page", bounded=False, suffix="-u")
    det = make_detector(
        "det",
        bounded,
        unbounded,
        acquire=FillingAcquire(bounded_source, open_source),
    )
    await det.stage()
    await det.prepare(TriggerInfo())
    values = []
    for _ in range(3):
        await det.trigger()
        reading = await det.read()
        values.append((reading["det-b"]["value"], reading["det-u"]["value"]))
    assert values == [([0], [3]), ([1], [4]), ([2], [5])]


async def test_d3_fly_scan_with_an_unbounded_source_non_zero_at_kickoff():
    bounded_source, open_source = Source(), Source()
    await open_source.produce(4)
    det = make_detector(
        "det",
        FakeData(bounded_source, publication="page", bounded=True, suffix="-b"),
        FakeData(open_source, publication="page", bounded=False, suffix="-u"),
        acquire=FillingAcquire(bounded_source, open_source, per_start=3),
    )
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=3))
    await det.kickoff()
    await det.complete()
    # Events since prepare()
    assert await det.get_index() == 3
    (page,) = [p async for p in det.collect_pages()]  # type: ignore[attr-defined]
    assert page["data"]["det-b"] == [[0], [1], [2]]
    assert page["data"]["det-u"] == [[4], [5], [6]]


class _BoundedAndUnbounded(BoundedDataLogic[Any], UnboundedDataLogic[Any]): ...


@pytest.mark.parametrize("dl", [DetectorDataLogic(), _BoundedAndUnbounded()])
def test_d3_data_must_be_a_bounded_or_an_unbounded_logic(dl):
    with pytest.raises(TypeError, match="must inherit from exactly one of"):
        DetectorLogic(data=[dl])


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
        FakeData(source, publication="stream", bounded=False, tmp_path=tmp_path),
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
        _ = [doc async for doc in det.collect_asset_docs()]  # type: ignore[attr-defined]
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
    TABLE[(_state, "collect_asset_docs")] = _state in "TKC"


#: The states in which a verb may read the prepared context
_PREPARED = "FLOWN, FLYING, PREPARED, TRIGGERED, TRIGGERING"

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
    ("U", "read"): "det: read not allowed from UNSTAGED, allowed from: TRIGGERED",
    ("S", "read"): "det: read not allowed from STAGED, allowed from: TRIGGERED",
    ("P", "read"): "det: read not allowed from PREPARED, allowed from: TRIGGERED",
    ("K", "read"): "det: read not allowed from FLYING, allowed from: TRIGGERED",
    ("C", "read"): "det: read not allowed from FLOWN, allowed from: TRIGGERED",
    (
        "U",
        "describe",
    ): f"det: describe not allowed from UNSTAGED, allowed from: {_PREPARED}",
    (
        "S",
        "describe",
    ): f"det: describe not allowed from STAGED, allowed from: {_PREPARED}",
    (
        "U",
        "describe_collect",
    ): f"det: describe_collect not allowed from UNSTAGED, allowed from: {_PREPARED}",
    (
        "S",
        "describe_collect",
    ): f"det: describe_collect not allowed from STAGED, allowed from: {_PREPARED}",
    (
        "U",
        "get_index",
    ): f"det: get_index not allowed from UNSTAGED, allowed from: {_PREPARED}",
    (
        "S",
        "get_index",
    ): f"det: get_index not allowed from STAGED, allowed from: {_PREPARED}",
    ("P", "collect_asset_docs"): (
        "det: collect_asset_docs not allowed from PREPARED, allowed from: "
        "FLOWN, FLYING, TRIGGERED"
    ),
}


def _rows():
    for (state, verb), ok in TABLE.items():
        id = f"{state}-{verb}"
        yield pytest.param(state, verb, ok, id=id)


@pytest.mark.parametrize("state,verb,ok", _rows())
async def test_lifecycle_verb_from_state(state, verb, ok, tmp_path):
    det = await reach(state, tmp_path)
    if ok:
        await call(det, verb)
    else:
        with pytest.raises(RuntimeError, match=re.escape(MESSAGES[(state, verb)])):
            await call(det, verb)
    # Ends a fly still running in "K"
    await det.unstage()


async def test_a_prepare_refused_mid_fly_leaves_the_collect_verb_bound(tmp_path):
    det = await reach("K", tmp_path)
    with pytest.raises(RuntimeError, match="cannot go to PREPARING from FLYING"):
        await det.prepare(TriggerInfo())
    cast(GatedAcquire, det.logic.acquire).done.set()
    await det.complete()
    assert [doc async for doc in det.collect_asset_docs()]  # type: ignore[attr-defined]
    await det.unstage()


@pytest.mark.parametrize("state", STATES)
async def test_restage_returns_to_staged_from_any_state(state, tmp_path):
    det = await reach(state, tmp_path)
    await det.unstage()
    await det.stage()
    with pytest.raises(TransitionError, match="describe not allowed from STAGED"):
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
        match="cannot go to TRIGGERING from FLOWN, allowed: PREPARING",
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


@pytest.mark.xfail(
    reason="an armed acquisition never taken is dropped, not closed (Tom 2026-10-22)",
    strict=True,
)
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


async def test_unstage_cancels_a_running_trigger_that_kickoff_cannot_overlap():
    acquire = HeldAcquire()
    det = make_detector(
        "det", FakeData(Source(), publication="page", bounded=False), acquire=acquire
    )
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
# Use cases 10 and 11
# ---------------------------------------------------------------------------


async def test_a_bounded_logic_publishes_only_the_keys_unbounded_do_not_make():
    unbounded_source, bounded_source = Source(), Source()
    det = make_detector(
        "det",
        FakeData(unbounded_source, publication="page", bounded=False, suffix="-u"),
        FakeData(
            bounded_source, publication="page", bounded=True, suffix="-u", also=["-b"]
        ),
        acquire=FillingAcquire(unbounded_source, bounded_source, per_start=2),
    )
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=2))
    await det.kickoff()
    await det.complete()

    (page,) = [p async for p in det.collect_pages()]  # type: ignore[attr-defined]
    assert page["data"] == {"det-u": [[0], [1]], "det-b": [[0], [1]]}
    # Hinted fields are those the providers make
    assert det.hints == {"fields": ["det-u"]}


@pytest.mark.parametrize(
    "bounded, again",
    [
        # A bounded logic is restarted at the next step point
        (True, lambda det: det.trigger()),
        # An unbounded one at a prepare() for a different scan
        (False, lambda det: det.prepare(TriggerInfo(collections_per_event=2))),
    ],
)
async def test_a_data_logic_that_served_the_stage_must_serve_all_of_it(
    bounded: bool, again
):
    source = Source()
    data = FakeData(source, publication="page", bounded=bounded)
    det = make_detector("det", data, acquire=FillingAcquire(source))
    await det.stage()
    await det.trigger()

    data.sit_out = True
    with pytest.raises(RuntimeError, match=r"made \[\] at .*, but \['det'\] before it"):
        await again(det)


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
    assert not hasattr(det, "collect_asset_docs")
    assert not hasattr(det, "collect_pages")


# ---------------------------------------------------------------------------
# Data-set rules (§5)
# ---------------------------------------------------------------------------


async def test_u5_read_is_idempotent_for_pageable_data():
    source = Source()
    det = make_detector(
        "det",
        FakeData(source, publication="page", bounded=True),
        acquire=FillingAcquire(source, per_start=2),
    )
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=2))
    await det.trigger()

    first = await det.read()
    second = await det.read()

    assert first["det"]["value"] == [0, 1]
    assert second == first


async def test_u5_collect_pages_takes_an_index():
    source = Source()
    det = make_detector(
        "det",
        FakeData(source, publication="page", bounded=True),
        acquire=FillingAcquire(source, per_start=5),
    )
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=5))
    await det.kickoff()

    first = [p async for p in det.collect_pages(index=2)]  # type: ignore[attr-defined]
    rest = [p async for p in det.collect_pages(index=5)]  # type: ignore[attr-defined]

    assert [p["data"]["det"] for p in first] == [[[0], [1]]]
    assert [p["data"]["det"] for p in rest] == [[[2], [3], [4]]]


async def test_uc12b_multiple_pageable_on_one_detector_merge_into_one_page():
    """Use case 12(b), written as if the bundler limits were fixed."""
    a, b = Source(), Source()
    det = make_detector(
        "det",
        FakeData(a, publication="page", bounded=True, suffix="-a"),
        FakeData(b, publication="page", bounded=True, suffix="-b"),
        acquire=FillingAcquire(a, b, per_start=3),
    )
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=3))
    await det.kickoff()
    await det.complete()

    pages = [p async for p in det.collect_pages()]  # type: ignore[attr-defined]

    (page,) = pages
    assert page["data"] == {"det-a": [[0], [1], [2]], "det-b": [[0], [1], [2]]}
    assert set(page["timestamps"]) == {"det-a", "det-b"}


@pytest.mark.xfail(
    strict=True,
    reason="bluesky's RunBundler collects stream assets XOR event pages per "
    "device, so prepare() raises on the mix; passes once that goes",
)
async def test_uc12a_mixed_pageable_and_streamable_on_one_detector(tmp_path):
    """Use case 12(a), written as if the bundler limits were fixed."""
    stream_source, page_source = Source(), Source()
    det = make_detector(
        "det",
        FakeData(stream_source, publication="stream", bounded=False, tmp_path=tmp_path),
        FakeData(page_source, publication="page", bounded=True, suffix="-stats"),
        acquire=FillingAcquire(stream_source, page_source, per_start=2),
    )
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=2))
    assert set(await det.describe_collect()) == {"det", "det-stats"}
    await det.kickoff()
    await det.complete()

    docs = [d async for d in det.collect_asset_docs()]  # type: ignore[attr-defined]
    pages = [p async for p in det.collect_pages()]  # type: ignore[attr-defined]

    assert [name for name, _ in docs] == ["stream_resource", "stream_datum"]
    (page,) = pages
    assert page["data"] == {"det-stats": [[0], [1]]}


class _BreachingData(FakeUnboundedData):
    """Its provider's pages drop their last event, are `None`, or page `det` too."""

    breach = ""

    async def start_provider(self, *args: Any) -> Any:
        provider = await super().start_provider(*args)
        make_page = provider.make_page

        async def breaching_make_page(index: int) -> PartialEventPage | None:
            page = await make_page(index)
            if page and self.breach == "short":
                page = {
                    "data": {k: v[:-1] for k, v in page["data"].items()},
                    "timestamps": {k: v[:-1] for k, v in page["timestamps"].items()},
                    "time": page["time"][:-1],
                }
            elif self.breach == "none":
                page = None
            elif page and self.breach == "extra_key":
                page["data"]["det"] = page["data"]["det-b"]
            return page

        provider.make_page = breaching_make_page
        return provider


@pytest.mark.parametrize(
    "breach, error, match",
    [
        (
            "short",
            TypeError,
            "Differing numbers of events: FakePageable has 1, FakePageable has 0",
        ),
        (
            "none",
            TypeError,
            "Differing numbers of events: FakePageable has 1, FakePageable has None",
        ),
        (
            "extra_key",
            ValueError,
            "Duplicate keys found while merging dictionaries: det",
        ),
    ],
)
async def test_pageable_providers_must_page_each_event_of_their_own_keys(
    breach: str, error: type[Exception], match: str
):
    source = Source()
    breaching = _BreachingData(source, publication="page", suffix="-b")
    breaching.breach = breach
    det = make_detector(
        "det",
        FakeData(source, publication="page", bounded=False),
        breaching,
        acquire=FillingAcquire(source),
    )
    await det.stage()
    await det.trigger()
    with pytest.raises(error, match=match):
        await det.read()


def _reading_detector(suppressor: _FakeData, source: Source, value: Any):
    reading = soft_signal_rw(Array1D[np.int64], value, name="det-reading")
    return make_detector(
        "det", suppressor, acquire=FillingAcquire(source, per_start=3), readable=None
    ), reading


@pytest.mark.parametrize("publication", ["page", "stream"])
async def test_readable_is_suppressed_by_pageable_and_streamable_keys(
    publication, tmp_path
):
    source = Source()
    data = FakeData(
        source,
        publication=publication,
        bounded=False,
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
    if publication == "page":
        assert (await det.read())["det-reading"]["value"] == [0, 1, 2]
    else:
        assert await det.read() == {}


async def test_same_name_different_shape_is_a_duplicate_not_a_suppression():
    source = Source()
    data = FakeData(source, publication="page", bounded=False, suffix="-reading")
    det, reading = _reading_detector(data, source, np.array([9, 9], dtype=np.int64))
    det.reading = reading
    det.set_readable_format(reading, Format.HINTED_SIGNAL)
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=3))

    with pytest.raises(ValueError, match=r"duplicate datakeys \['det-reading'\]"):
        await det.describe()


async def test_u9_a_suppressed_readable_keeps_its_hint():
    source = Source()
    data = FakeData(
        source, publication="page", bounded=False, suffix="-reading", hints=lambda _: []
    )
    det, reading = _reading_detector(data, source, np.array([9, 9, 9], dtype=np.int64))
    det.reading = reading
    det.set_readable_format(reading, Format.HINTED_SIGNAL)
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=3))

    await det.describe()
    # The data that replaces the signal has its name, so the signal's hint stands
    assert det.hints == {"fields": ["det-reading"]}


async def test_u9_a_suppressed_readable_and_its_suppressor_both_hinting_is_an_error():
    source = Source()
    data = FakeData(
        source,
        publication="page",
        bounded=False,
        suffix="-reading",
        hints=lambda name: [name],
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


@pytest.mark.parametrize("publication", ["page", "stream"])
async def test_u7_duplicate_datakeys_within_a_tier_raise_at_prepare(
    publication, tmp_path
):
    det = make_detector(
        "det",
        FakeData(Source(), publication=publication, bounded=False, tmp_path=tmp_path),
        FakeData(Source(), publication=publication, bounded=False, tmp_path=tmp_path),
        acquire=FillingAcquire(),
    )
    await det.stage()
    with pytest.raises(ValueError, match="[Dd]uplicate"):
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
    monkeypatch: pytest.MonkeyPatch, env, expected_period
):
    if env is None:
        monkeypatch.delenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", raising=False)
    else:
        monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", env)
    data = FakeData(Source(), publication="page", bounded=False)
    det = make_detector(
        "det", data, trigger=ReadbackTrigger(), acquire=FillingAcquire()
    )
    await det.stage()

    await det.prepare(TriggerInfo(number_of_events=1))

    assert data.periods == [expected_period]


# ---------------------------------------------------------------------------
# collect_pages from each state (U1): only after kickoff()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,error",
    [
        (
            "P",
            "det: collect_pages not allowed from PREPARED, allowed from: FLOWN, FLYING",
        ),
        (
            "T",
            "det: collect_pages not allowed from TRIGGERED, "
            "allowed from: FLOWN, FLYING",
        ),
        ("K", None),
        ("C", None),
    ],
)
async def test_collect_pages_from_each_state(state: str, error: str | None):
    source = Source()
    det = make_detector(
        "det",
        FakeData(source, publication="page", bounded=False),
        acquire=FillingAcquire(source),
    )
    await det.stage()
    await det.prepare(TriggerInfo())
    if state == "T":
        await det.trigger()
    elif state in "KC":
        await det.kickoff()
        if state == "C":
            await det.complete()

    collect = det.collect_pages()  # type: ignore[attr-defined]
    if error is None:
        assert [page async for page in collect] != []
    else:
        with pytest.raises(RuntimeError, match=re.escape(error)):
            _ = [page async for page in collect]


# ---------------------------------------------------------------------------
# Verifier findings 2 and 4 to 6: re-asking, notes, failed prepare and start
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("publication", ["stream", "page"])
async def test_an_unbounded_provider_carries_over_without_its_logic_being_asked(
    publication: str, tmp_path
):
    source = Source()
    unbounded = FakeData(
        source, publication=publication, bounded=False, tmp_path=tmp_path
    )
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


async def test_a_carried_pageable_provider_first_page_starts_at_the_new_baseline():
    source = Source()
    det = make_detector(
        "det",
        FakeData(source, publication="page", bounded=False),
        acquire=FillingAcquire(source),
    )
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()
    await det.prepare(TriggerInfo())
    await det.kickoff()
    await det.complete()

    # The first run's collection is not repeated in the second run's page
    pages = [page async for page in det.collect_pages()]  # type: ignore[attr-defined]
    assert [page["data"] for page in pages] == [{"det": [[1]]}]


async def test_a_changed_scan_stops_the_old_provider_before_making_the_new(tmp_path):
    sources = Source(), Source()
    unbounded = FakeData(
        sources[0], publication="stream", bounded=False, tmp_path=tmp_path
    )
    steady = FakeData(
        sources[1], publication="stream", bounded=False, tmp_path=tmp_path, suffix="-s"
    )
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
    plugin = FakeData(source, publication="stream", bounded=False, tmp_path=tmp_path)
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
        "read",
        "collect_pages",
        "collect_asset_docs",
    ],
)
async def test_f4_exceptions_from_logic_carry_the_detectors_name(verb: str, tmp_path):
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
    publication = "stream" if verb == "collect_asset_docs" else "page"
    data = FakeData(source, publication=publication, bounded=False, tmp_path=tmp_path)
    det = make_detector("det", data, acquire=acquire)
    # Before kickoff, as fly() runs on to the end of acquire() as soon as it can
    acquire.failing = verb
    if verb != "stage":
        await det.stage()
    if verb not in ("stage", "unstage"):
        await det.prepare(TriggerInfo(number_of_events=1))
    if verb == "read":
        await det.trigger()
    if verb in ("complete", "collect_pages", "collect_asset_docs"):
        await det.kickoff()
    data.fail = verb in ("read", "collect_pages", "collect_asset_docs")

    with pytest.raises(BoomError) as error:
        if verb == "collect_pages":
            _ = [page async for page in det.collect_pages()]  # type: ignore[attr-defined]
        elif verb == "collect_asset_docs":
            _ = [doc async for doc in det.collect_asset_docs()]  # type: ignore[attr-defined]
        else:
            await getattr(det, verb)()
    assert error.value.__notes__ == ["raised by det"]


@pytest.mark.parametrize("publication", ["page", "stream"])
@pytest.mark.parametrize("cpe", [1, 3])
async def test_scalar_readable_is_suppressed_by_a_key_of_any_collections_per_event(
    publication: str, cpe: int, tmp_path
):
    """A readable gives one value per event, so [cpe] suppresses it, not just []."""
    source = Source()
    data = FakeData(
        source,
        publication=publication,
        bounded=False,
        suffix="-reading",
        tmp_path=tmp_path,
    )
    det = make_detector("det", data, acquire=FillingAcquire(source, per_start=cpe))
    det.reading = soft_signal_rw(int, 9, name="det-reading")
    det.set_readable_format(det.reading, Format.HINTED_SIGNAL)
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=cpe))
    await det.trigger()

    description = await det.describe()

    assert description["det-reading"]["shape"] == [cpe]
    reading = await det.read()
    if publication == "page":
        assert reading["det-reading"]["value"] == list(range(cpe))
    else:
        assert reading == {}


def test_data_logic_with_the_removed_stop_raises():
    with pytest.raises(TypeError, match=r"StaleData defines stop, which .* -> ensure"):
        type("StaleData", (BoundedDataLogic,), {"stop": lambda self: None})
