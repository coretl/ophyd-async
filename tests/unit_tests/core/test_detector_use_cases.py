"""Use-case tests for the detector core (spec: detector-logic-spec.md).

Written from the spec for the detector-core rewrite. The strict xfails are
behaviour the spec wants that something outside ophyd-async still prevents.

The fakes in this file stand in for the three kinds of device a detector drives:
a device-side `Source` (counter plus data), an acquire logic that fills it, and
data logics that publish it as readables, event pages or stream assets.
"""

import asyncio
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np
import pytest
from bluesky import RunEngine
from event_model import DataKey
from event_model.documents import PartialEventPage

from ophyd_async.core import (
    Array1D,
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
    TriggerInfo,
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
    """Every start_acquiring() produces `per_start` collections in each source."""

    def __init__(self, *sources: Source, per_start: int = 1) -> None:
        self.sources = sources
        self.per_start = per_start
        self.idle_waits = 0

    async def start_acquiring(self):
        for source in self.sources:
            await source.produce(self.per_start)

    async def wait_for_idle(self):
        self.idle_waits += 1

    async def ensure_stopped(self): ...


class FakePageable(PageableDataProvider):
    """Event pages straight from a source, one value per collection.

    Implements both the pageable verb of today's code and the stateless
    `make_page` of ruling U5, and the step-boundary hook of D2 (assumed name
    `step_boundary`), so the same fake serves before and after the rewrite.
    """

    def __init__(
        self,
        datakey_name: str,
        source: Source,
        rearms: bool,
        shape_tail: tuple[int, ...] = (),
        dtype_numpy: str = "<i8",
    ) -> None:
        self.datakey_name = datakey_name
        self.source = source
        self.rearms = rearms
        self.shape_tail = shape_tail
        self.dtype_numpy = dtype_numpy
        self.collections_written_signal = source.written
        self.last_emitted = 0
        self.hook_calls = 0

    async def make_datakeys(self, collections_per_event: int) -> dict[str, DataKey]:
        return {
            self.datakey_name: DataKey(
                source="fake",
                shape=[collections_per_event, *self.shape_tail],
                dtype="array",
                dtype_numpy=self.dtype_numpy,
            )
        }

    def _page(self, first: int, last: int, cpe: int) -> PartialEventPage:
        events = range(first, last)
        return {
            "data": {
                self.datakey_name: [
                    self.source.data[e * cpe : (e + 1) * cpe] for e in events
                ]
            },
            "time": [float(e) for e in events],
            "timestamps": {self.datakey_name: [float(e) for e in events]},
        }

    async def make_pages(self, collections_written: int, collections_per_event: int):
        # Today's stateful verb
        events = collections_written // collections_per_event
        if events > self.last_emitted:
            page = self._page(self.last_emitted, events, collections_per_event)
            self.last_emitted = events
            yield page

    async def make_page(
        self, collections_from: int, collections_to: int, collections_per_event: int
    ) -> PartialEventPage | None:
        # Ruling U5's stateless verb
        first = collections_from // collections_per_event
        last = collections_to // collections_per_event
        return self._page(first, last, collections_per_event) if last > first else None

    async def step_boundary(self) -> None:
        # D2's hook: a finite buffer re-arms itself
        self.hook_calls += 1
        if self.rearms:
            await self.source.rearm()


class FakeData(DetectorDataLogic[None]):
    """Publishes a `Source` as event pages or stream assets.

    :param bounded: needs a finite N (sits out N=0) and, if pageable, is a finite
        buffer that has to be cleared between step points
    """

    def __init__(
        self,
        source: Source,
        *,
        publication: str,
        bounded: bool,
        suffix: str = "",
        shape_tail: tuple[int, ...] = (),
        dtype_numpy: str = "<i8",
        tmp_path: Any = None,
        hints: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self.source = source
        self.publication = publication
        self.bounded = bounded
        self.datakey_suffix = suffix
        self.shape_tail = shape_tail
        self.dtype_numpy = dtype_numpy
        self.tmp_path = tmp_path
        self.hints = hints
        self.periods: list[float] = []
        self.starts = 0
        self.providers: list[FakePageable] = []

    @property
    def hook_calls(self) -> int:
        return sum(p.hook_calls for p in self.providers)

    async def make_data_provider(
        self,
        datakey_name: str,
        num_collections: int,
        period: float,
        flush_period: float,
    ) -> tuple[Any, None] | None:
        self.periods.append(period)
        if self.bounded and num_collections == 0:
            return None
        if self.publication == "stream":
            resource = StreamResourceInfo(
                data_key=datakey_name,
                shape=self.shape_tail,
                chunk_shape=(1, *self.shape_tail),
                dtype_numpy=self.dtype_numpy,
                parameters={"dataset": "/data"},
            )
            return StreamResourceDataProvider(
                uri=f"file://localhost/{self.tmp_path.as_posix().lstrip('/')}/f.h5",
                resources=[resource],
                mimetype="application/x-hdf5",
                collections_written_signal=self.source.written,
            ), None
        provider = FakePageable(
            datakey_name,
            self.source,
            rearms=self.bounded,
            shape_tail=self.shape_tail,
            dtype_numpy=self.dtype_numpy,
        )
        self.providers.append(provider)
        return provider, None

    async def start(self, ctx: None) -> None:
        self.starts += 1
        if self.bounded and self.publication == "page":
            # A finite buffer is armed empty
            await self.source.rearm()

    async def stop(self) -> None: ...

    def get_hinted_fields(self, datakey_name: str) -> Sequence[str]:
        return self.hints(datakey_name) if self.hints else [datakey_name]


def make_detector(
    name: str = "det",
    *data: FakeData,
    acquire: FillingAcquire | None = None,
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
        assert [d["indices"]["stop"] for d in docs["stream_datum"]] == [1, 2, 3]
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
    assert await det.get_index() == 7


# ---------------------------------------------------------------------------
# The step-boundary hook (D2): DataProvider.step_boundary()
# ---------------------------------------------------------------------------


async def test_d2_step_boundary_hook_runs_at_each_point_after_the_first():
    source = Source()
    data = FakeData(source, publication="page", bounded=True)
    det = make_detector("det", data, acquire=FillingAcquire(source))
    await det.stage()
    await det.prepare(TriggerInfo())

    await det.trigger()
    assert data.hook_calls == 0
    await det.trigger()
    await det.trigger()
    assert data.hook_calls == 2


async def test_d2_kickoff_never_calls_the_step_boundary_hook_but_trigger_does():
    """Needs the hook to exist at all to tell fly (never) from step (after 1st)."""
    source = Source()
    data = FakeData(source, publication="page", bounded=True)
    det = make_detector("det", data, acquire=FillingAcquire(source, per_start=2))
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=2))
    await det.kickoff()
    assert data.hook_calls == 0
    await det.complete()
    assert data.hook_calls == 0
    await det.prepare(TriggerInfo())
    await det.trigger()
    await det.trigger()
    assert data.hook_calls == 1


# ---------------------------------------------------------------------------
# Lifecycle state machine (spec §3), every verb from every state
# ---------------------------------------------------------------------------

STATES = ["U", "S", "P", "T", "K", "C"]


async def reach(state: str, tmp_path) -> StandardDetector:
    source = Source()
    det = make_detector(
        "det",
        FakeData(source, publication="stream", bounded=False, tmp_path=tmp_path),
        acquire=FillingAcquire(source),
    )
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
    elif verb == "collect_asset_docs":
        _ = [doc async for doc in det.collect_asset_docs()]  # type: ignore[attr-defined]
    else:
        await asyncio.wait_for(getattr(det, verb)(), 2)


# (state, verb) -> does the verb succeed. From the §3 table.
TABLE: dict[tuple[str, str], bool] = {}
for _verb, _row in {
    "prepare": "FTTTFT",
    "trigger": "FTTTFF",
    "kickoff": "FFTFFF",
    "complete": "FFFFTF",
    "read": "FFFTFF",
    "describe": "FFTTTT",
    "describe_collect": "FFTTTT",
    "get_index": "FFTTTT",
}.items():
    for _state, _ok in zip(STATES, _row, strict=True):
        TABLE[(_state, _verb)] = _ok == "T"
for _state in "PTKC":
    TABLE[(_state, "collect_asset_docs")] = _state in "KC"

# Rows the code does not honour, with why
_NOT_TODAY: dict[tuple[str, str], str] = {
    ("T", "collect_asset_docs"): "§3 U1 says collect_* in a step scan raises, but "
    "bluesky's read() collects stream assets after each trigger() (use case 2)",
}


def _rows():
    for (state, verb), ok in TABLE.items():
        marks = []
        if (state, verb) in _NOT_TODAY:
            marks.append(
                pytest.mark.xfail(
                    strict=True,
                    reason=f"detector-core rewrite: {_NOT_TODAY[(state, verb)]}",
                )
            )
        yield pytest.param(state, verb, ok, marks=marks, id=f"{state}-{verb}")


@pytest.mark.parametrize("state,verb,ok", _rows())
async def test_lifecycle_verb_from_state(state, verb, ok, tmp_path):
    det = await reach(state, tmp_path)
    if ok:
        await call(det, verb)
    else:
        # The wording is TBD (§7), but in U it has to point at stage()
        with pytest.raises(
            (RuntimeError, ValueError), match="stage" if state == "U" else None
        ):
            await call(det, verb)


@pytest.mark.parametrize("state", STATES)
async def test_stage_returns_to_staged_from_any_state(state, tmp_path):
    det = await reach(state, tmp_path)
    await det.stage()
    with pytest.raises(RuntimeError, match="prepare.. must be called first"):
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
    with pytest.raises(RuntimeError, match="stage"):
        await det.prepare(TriggerInfo())


class _Flyer(StandardFlyable[int, None]):
    @cached_property
    def logic(self) -> FlyableLogic[int, None]:
        return _NoopFlyableLogic()


class _NoopFlyableLogic(FlyableLogic[int, None]):
    async def on_prepare(self, value: int) -> None: ...

    async def on_kickoff(self, ctx: None) -> None: ...

    async def on_complete(self, ctx: None) -> None: ...


@pytest.mark.parametrize("verb", ["prepare", "kickoff", "complete"])
async def test_r10_standard_flyable_verbs_raise_before_stage(verb: str):
    flyer = _Flyer(name="flyer")
    with pytest.raises(RuntimeError, match="stage"):
        await (flyer.prepare(1) if verb == "prepare" else getattr(flyer, verb)())
    await flyer.stage()
    await flyer.prepare(1)
    await flyer.unstage()
    await flyer.unstage()  # a no-op
    with pytest.raises(RuntimeError, match="stage"):
        await flyer.prepare(1)


async def test_trigger_after_complete_says_to_prepare_again(tmp_path):
    det = await reach("C", tmp_path)
    with pytest.raises(RuntimeError, match=r"complete.*prepare\(\) again"):
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


# ---------------------------------------------------------------------------
# Use cases 10 and 11
# ---------------------------------------------------------------------------


async def test_uc10_logics_needing_n_sit_out_a_zero_event_fly(tmp_path):
    """Use case 10: the bounded logic sits out, the unbounded one serves."""
    bounded_source, open_source = Source(), Source()
    bounded = FakeData(bounded_source, publication="page", bounded=True, suffix="-b")
    unbounded = FakeData(open_source, publication="page", bounded=False, suffix="-u")
    det = make_detector("det", bounded, unbounded, acquire=FillingAcquire())
    await det.stage()

    await det.prepare(TriggerInfo(number_of_events=0))

    assert list(await det.describe_collect()) == ["det-u"]
    assert det.hints == {"fields": ["det-u"]}


async def test_uc10_a_logic_that_sat_out_serves_again_on_the_next_prepare():
    bounded = FakeData(Source(), publication="page", bounded=True, suffix="-b")
    unbounded = FakeData(Source(), publication="page", bounded=False, suffix="-u")
    det = make_detector("det", bounded, unbounded, acquire=FillingAcquire())
    await det.stage()
    await det.prepare(TriggerInfo(number_of_events=0))
    await det.prepare(TriggerInfo(number_of_events=1))

    assert set(await det.describe_collect()) == {"det-b", "det-u"}


async def test_uc10_all_logics_sitting_out_raises_at_prepare():
    det = make_detector(
        "det",
        FakeData(Source(), publication="page", bounded=True),
        acquire=FillingAcquire(),
    )
    await det.stage()
    with pytest.raises(ValueError, match="serve|data"):
        await det.prepare(TriggerInfo(number_of_events=0))


async def test_uc11_no_data_no_readables_steps_but_does_not_fly():
    acquire = FillingAcquire()
    det = make_detector("det", acquire=acquire)
    await det.stage()

    await det.trigger()
    await det.trigger()
    assert acquire.idle_waits == 2
    assert await det.read() == {}

    await det.prepare(TriggerInfo(number_of_events=3))
    with pytest.raises(ValueError, match="no collectable data"):
        await det.kickoff()


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
    "device, so prepare() raises on the mix (§5.3/D2); passes once that goes",
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


def _reading_detector(shadower: FakeData, source: Source, value: Any):
    reading = soft_signal_rw(Array1D[np.int64], value, name="det-reading")
    return make_detector(
        "det", shadower, acquire=FillingAcquire(source, per_start=3), readable=None
    ), reading


@pytest.mark.parametrize("publication", ["page", "stream"])
async def test_readable_is_shadowed_by_pageable_and_streamable_keys(
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


async def test_same_name_different_shape_is_a_duplicate_not_a_shadow():
    source = Source()
    data = FakeData(source, publication="page", bounded=False, suffix="-reading")
    det, reading = _reading_detector(data, source, np.array([9, 9], dtype=np.int64))
    det.reading = reading
    det.set_readable_format(reading, Format.HINTED_SIGNAL)
    await det.stage()

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        await det.prepare(TriggerInfo(collections_per_event=3))


@pytest.mark.parametrize(
    "shadower_hints,expected_hints,warns",
    [
        (lambda name: [name], {"fields": ["det-reading"]}, False),
        (lambda name: [], {}, True),
    ],
)
async def test_u9_hints_of_a_shadowed_readable(shadower_hints, expected_hints, warns):
    source = Source()
    data = FakeData(
        source,
        publication="page",
        bounded=False,
        suffix="-reading",
        hints=shadower_hints,
    )
    det, reading = _reading_detector(data, source, np.array([9, 9, 9], dtype=np.int64))
    det.reading = reading
    det.set_readable_format(reading, Format.HINTED_SIGNAL)
    await det.stage()

    with warnings.catch_warnings():
        warnings.simplefilter("error" if not warns else "default")
        if warns:
            with pytest.warns(Warning):
                await det.prepare(TriggerInfo(collections_per_event=3))
        else:
            await det.prepare(TriggerInfo(collections_per_event=3))

    assert det.hints == expected_hints


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
