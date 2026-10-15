import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast
from unittest.mock import Mock

import pytest

from ophyd_async.core import (
    AsyncStatus,
    FlyableLogic,
    MovableLogic,
    StandardFlyable,
    StandardMovable,
    StandardReadable,
    StandardReadableFormat,
    TransitionError,
    WatchableAsyncStatus,
    WatcherUpdate,
    init_devices,
    set_mock_value,
    soft_signal_rw,
    watch_readback,
)


@dataclass
class RecordingFlyableLogic(FlyableLogic[int]):
    """A `FlyableLogic` that records its calls. Not movable, so prepare() and
    complete() report no progress to watchers."""

    calls: list[str] = field(default_factory=list)
    prepared_with: list[int] = field(default_factory=list)

    async def on_prepare(self, value: int) -> None:
        self.calls.append("on_prepare")
        self.prepared_with.append(value)

    async def on_kickoff(self, ctx: None) -> None:
        self.calls.append("on_kickoff")

    async def on_complete(self, ctx: None, status: None) -> None:
        self.calls.append("on_complete")

    async def stop(self) -> None:
        self.calls.append("stop")


@dataclass(frozen=True)
class Ctx:
    value: int
    # Anything that changes during the scan is an object the ctx holds
    log: list[str] = field(default_factory=list)


@dataclass
class CtxFlyableLogic(FlyableLogic[int, Ctx, AsyncStatus]):
    """A `FlyableLogic` that threads a ctx, and the kickoff status, between hooks."""

    seen: list[tuple[str, Ctx, AsyncStatus | None]] = field(default_factory=list)
    made: list[Ctx] = field(default_factory=list)
    status: AsyncStatus | None = None

    async def on_prepare(self, value: int) -> Ctx:
        self.made.append(Ctx(value))
        return self.made[0]

    async def on_kickoff(self, ctx: Ctx) -> AsyncStatus:
        self.seen.append(("on_kickoff", ctx, None))
        self.status = AsyncStatus(asyncio.sleep(0))
        return self.status

    async def on_complete(self, ctx: Ctx, status: AsyncStatus) -> None:
        self.seen.append(("on_complete", ctx, status))


@dataclass
class MovableFlyableLogic(MovableLogic[float], FlyableLogic[float, float, AsyncStatus]):
    """Both movable and flyable, mirroring how `Motor` combines the two.

    `prepare()` and `complete()` should report progress by watching the readback
    with `watch_readback`, reusing the same watcher updates as
    `StandardMovable.set`.
    """

    async def _step_to(self, target: float) -> None:
        # Step the readback towards the target so watchers see progress.
        for position in (target / 2, target):
            set_mock_value(self.readback, position)
            await asyncio.sleep(0)

    async def on_prepare(
        self, value: float
    ) -> tuple[float, AsyncIterator[WatcherUpdate]]:
        return value, watch_readback(self, self._step_to(value), target=value)

    async def on_kickoff(self, ctx: float) -> AsyncStatus:
        return AsyncStatus(self._step_to(2 * ctx))

    async def on_complete(
        self, ctx: float, status: AsyncStatus
    ) -> AsyncIterator[WatcherUpdate]:
        return watch_readback(self, status, 2 * ctx, initial=ctx)


class MovableFlyer(StandardMovable[float], StandardFlyable[float, float, AsyncStatus]):
    def __init__(self, name: str = ""):
        self.readback = soft_signal_rw(float)
        self.setpoint = soft_signal_rw(float)
        super().__init__(name=name)

    @cached_property
    def logic(self) -> MovableFlyableLogic:
        return MovableFlyableLogic(setpoint=self.setpoint, readback=self.readback)


@dataclass
class BlockingFlyableLogic(RecordingFlyableLogic):
    """Blocks forever inside the named hook, so a verb can be left in flight."""

    block: str = ""

    async def _maybe_block(self, hook: str) -> None:
        self.calls.append(hook)
        if self.block == hook:
            await asyncio.Event().wait()

    async def on_prepare(self, value: int) -> None:
        await self._maybe_block("on_prepare")

    async def on_kickoff(self, ctx: None) -> None:
        await self._maybe_block("on_kickoff")

    async def on_complete(self, ctx: None, status: None) -> None:
        await self._maybe_block("on_complete")


@dataclass
class FlyingLogic(FlyableLogic[int, None, AsyncStatus]):
    """Kicks off a status that runs forever; `on_complete` waits for something else
    entirely, so only `StandardFlyable` can end the status."""

    fail: bool = False
    fly_status: AsyncStatus | None = None
    running_when_stopped: list[bool] = field(default_factory=list)

    async def on_prepare(self, value: int) -> None: ...

    async def on_kickoff(self, ctx: None) -> AsyncStatus:
        self.fly_status = AsyncStatus(asyncio.Event().wait())
        return self.fly_status

    async def on_complete(self, ctx: None, status: AsyncStatus) -> None:
        if self.fail:
            raise ValueError("bad complete")
        await asyncio.Event().wait()

    async def stop(self) -> None:
        status = self.fly_status
        self.running_when_stopped.append(status is not None and not status.done)


async def make_staged_flyer(logic: FlyableLogic[Any, Any, Any]):
    async with init_devices(mock=True):
        flyer = logic.with_device(name="flyer")
    await flyer.stage()
    return flyer


@pytest.fixture
async def recording_flyer():
    """A staged ephemeral `StandardFlyable` wrapping a `RecordingFlyableLogic`."""
    logic = RecordingFlyableLogic()
    flyer = await make_staged_flyer(logic)
    logic.calls.clear()  # forget the stop() from stage
    return logic, flyer


async def test_ephemeral_flyable_drives_logic_hooks(recording_flyer):
    logic, flyer = recording_flyer

    assert flyer.name == "flyer"
    assert isinstance(flyer, StandardFlyable)
    # The ephemeral device exposes the same logic instance
    assert flyer.logic is logic

    # Every point of a scan reuses the staged flyer: COMPLETED can prepare again
    for value in (5, 6):
        await flyer.prepare(value)
        await flyer.kickoff()
        await flyer.complete()

    assert logic.calls == ["on_prepare", "on_kickoff", "on_complete"] * 2
    assert logic.prepared_with == [5, 6]
    await flyer.unstage()
    assert logic.calls[-1] == "stop"


async def test_flyable_threads_ctx_and_status_between_hooks():
    logic = CtxFlyableLogic()
    flyer = await make_staged_flyer(logic)

    await flyer.prepare(5)
    await flyer.kickoff()
    await flyer.complete()

    # Every hook sees the one ctx on_prepare returned, and on_complete sees
    # what on_kickoff returned
    assert [(name, ctx.value) for name, ctx, _ in logic.seen] == [
        ("on_kickoff", 5),
        ("on_complete", 5),
    ]
    assert all(ctx is logic.made[0] for _, ctx, _ in logic.seen)
    assert logic.seen[-1][2] is logic.status


@pytest.mark.parametrize(
    "setup, verb, current",
    [
        # Not staged
        ([], "prepare", "UNSTAGED"),
        # Out of order
        (["stage"], "kickoff", "STAGED"),
        (["stage"], "complete", "STAGED"),
        (["stage", "prepare"], "complete", "PREPARED"),
        (["stage", "prepare", "kickoff"], "kickoff", "KICKED_OFF"),
        # A running fly cannot be prepared over
        (["stage", "prepare", "kickoff"], "prepare", "KICKED_OFF"),
        # Completed: only a new prepare (or stage/unstage) is allowed
        (["stage", "prepare", "kickoff", "complete"], "complete", "COMPLETED"),
        (["stage", "prepare", "kickoff", "complete"], "kickoff", "COMPLETED"),
    ],
)
async def test_flyable_verb_in_the_wrong_state_raises(setup, verb, current):
    logic = RecordingFlyableLogic()
    async with init_devices(mock=True):
        flyer = logic.with_device(name="flyer")
    for step in setup:
        await (getattr(flyer, step)(1) if step == "prepare" else getattr(flyer, step)())
    logic.calls.clear()

    with pytest.raises(TransitionError, match=rf"flyer.*from {current}.*allowed"):
        await (getattr(flyer, verb)(1) if verb == "prepare" else getattr(flyer, verb)())
    assert logic.calls == []


async def test_second_concurrent_complete_raises():
    logic = BlockingFlyableLogic(block="on_complete")
    flyer = await make_staged_flyer(logic)
    await flyer.prepare(1)
    await flyer.kickoff()
    completing = flyer.complete()
    await asyncio.sleep(0)

    with pytest.raises(TransitionError, match="from COMPLETING"):
        await flyer.complete()
    await flyer.unstage()
    with pytest.raises(asyncio.CancelledError):
        await completing


@pytest.mark.parametrize("hook", ["on_prepare", "on_kickoff", "on_complete"])
async def test_stage_during_a_verb_raises_and_unstage_cancels_it(hook):
    logic = BlockingFlyableLogic(block=hook)
    flyer = await make_staged_flyer(logic)
    verbs = ["on_prepare", "on_kickoff", "on_complete"]
    running = None
    for name, verb in zip(
        verbs, (flyer.prepare, flyer.kickoff, flyer.complete), strict=True
    ):
        running = verb(1) if name == "on_prepare" else verb()
        if name == hook:
            break
        await running
    await asyncio.sleep(0)
    assert logic.calls[-1] == hook

    # Only unstage can end a verb that is in flight
    with pytest.raises(TransitionError, match="cannot go to STAGED"):
        await flyer.stage()
    await flyer.unstage()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(cast(AsyncStatus, running), timeout=1)

    # The cancelled verb did not touch the state unstage left behind
    logic.block = ""
    await flyer.stage()
    await flyer.prepare(2)


async def test_flyable_unstage_resets_lifecycle(recording_flyer):
    # After prepare+kickoff, unstaging should reset so a later complete errors.
    logic, flyer = recording_flyer

    await flyer.prepare(5)
    await flyer.kickoff()
    await flyer.unstage()
    await flyer.stage()
    with pytest.raises(TransitionError, match="from STAGED"):
        await flyer.complete()


async def test_flyable_stage_twice_raises(recording_flyer):
    logic, flyer = recording_flyer

    with pytest.raises(TransitionError, match="cannot go to STAGED from STAGED"):
        await flyer.stage()
    # Nothing ran, and it is still staged
    assert logic.calls == []
    await flyer.prepare(1)


async def test_flyable_without_progress_yields_no_updates(recording_flyer):
    # A logic whose hooks return coroutines reports no progress, but prepare()
    # and complete() are still watchable
    logic, flyer = recording_flyer

    prepare_watcher, complete_watcher = Mock(), Mock()
    status = flyer.prepare(5)
    assert isinstance(status, WatchableAsyncStatus)
    status.watch(prepare_watcher)
    await status
    await flyer.kickoff()
    status = flyer.complete()
    assert isinstance(status, WatchableAsyncStatus)
    status.watch(complete_watcher)
    await status

    prepare_watcher.assert_not_called()
    complete_watcher.assert_not_called()
    assert logic.calls == ["on_prepare", "on_kickoff", "on_complete"]


async def test_flyable_reports_progress_from_logic():
    async with init_devices(mock=True):
        flyer = MovableFlyer(name="flyer")
    await flyer.stage()

    prepare_watcher, complete_watcher = Mock(), Mock()
    status = flyer.prepare(10.0)
    assert isinstance(status, WatchableAsyncStatus)
    status.watch(prepare_watcher)
    await status
    await flyer.kickoff()
    status = flyer.complete()
    assert isinstance(status, WatchableAsyncStatus)
    status.watch(complete_watcher)
    await status

    # Progress is stamped with the flyer's name, and prepare() moves to its
    # start before kickoff is allowed
    updates = [c.kwargs for c in prepare_watcher.call_args_list]
    assert [u["current"] for u in updates] == [0.0, 5.0, 10.0]
    assert {u["initial"] for u in updates} == {0.0}
    assert {u["target"] for u in updates} == {10.0}
    assert {u["name"] for u in updates} == {"flyer"}
    updates = [c.kwargs for c in complete_watcher.call_args_list]
    assert updates[-1]["current"] == 20.0
    assert {u["initial"] for u in updates} == {10.0}
    assert {u["target"] for u in updates} == {20.0}
    assert {u["name"] for u in updates} == {"flyer"}


async def test_watch_readback_reraises_the_actions_error():
    async with init_devices(mock=True):
        flyer = MovableFlyer(name="flyer")

    async def fail():
        await asyncio.sleep(0)
        raise ValueError("boom")

    updates = []
    with pytest.raises(ValueError, match="boom"):
        async for update in watch_readback(flyer.logic, fail(), target=1.0):
            updates.append(update)
    # Initial defaults to the readback now
    assert {u.initial for u in updates} == {0.0}


async def test_flyable_stage_unstage_default_to_stop(recording_flyer):
    logic, flyer = recording_flyer

    await flyer.unstage()
    await flyer.stage()
    # on_stage/on_unstage both default to stop()
    assert logic.calls == ["stop", "stop"]


async def test_flyable_stage_unstage_can_differ():
    # A flyer (e.g. PmacTrajectoryFlyableLogic) may need distinct stage/unstage.
    @dataclass
    class StageUnstageLogic(RecordingFlyableLogic):
        async def on_stage(self) -> None:
            self.calls.append("on_stage")

        async def on_unstage(self) -> None:
            self.calls.append("on_unstage")

    logic = StageUnstageLogic()
    async with init_devices(mock=True):
        flyer = logic.with_device(name="flyer")

    await flyer.stage()
    await flyer.unstage()
    assert logic.calls == ["on_stage", "on_unstage"]


async def test_flyable_composes_with_readable_staging():
    # Motor is StandardMovable + StandardFlyable + StandardReadable; the flyer's
    # stage/unstage must NOT shadow StandardReadable's caching of hinted signals.
    logic = RecordingFlyableLogic()

    class ReadableFlyer(StandardReadable, StandardFlyable[int, None]):
        def __init__(self, name: str = ""):
            with self.add_children_as_readables(StandardReadableFormat.HINTED_SIGNAL):
                self.sig = soft_signal_rw(float)
            super().__init__(name=name)

        @cached_property
        def logic(self) -> FlyableLogic[int, None]:
            return logic

    async with init_devices(mock=True):
        device = ReadableFlyer(name="rf")

    await device.stage()
    # StandardReadable contribution: the hinted signal is now cached (this raises
    # "not being monitored" if StandardFlyable.stage had shadowed it).
    await device.sig.read(cached=True)
    # StandardFlyable contribution ran too.
    assert logic.calls == ["stop"]

    await device.unstage()
    assert logic.calls == ["stop", "stop"]
    # The cache is torn down on unstage.
    with pytest.raises(RuntimeError, match="not being monitored"):
        await device.sig.read(cached=True)


# Bug: fly statuses were never cancelled
async def test_unstage_cancels_the_running_status_after_stopping_the_hardware():
    logic = FlyingLogic()
    flyer = await make_staged_flyer(logic)
    await flyer.prepare(1)
    await flyer.kickoff()
    await asyncio.sleep(0)
    assert logic.fly_status is not None
    assert not logic.fly_status.done

    await flyer.unstage()
    await asyncio.sleep(0)

    assert logic.fly_status.task.cancelled()
    # stage's stop saw nothing running, unstage's stop saw the status still running
    assert logic.running_when_stopped == [False, True]


# Bug: a failed re-prepare left the flyer prepared with the failed prepare's ctx
async def test_failed_reprepare_leaves_nothing_to_kick_off(recording_flyer):
    logic, flyer = recording_flyer
    await flyer.prepare(1)

    async def fail(value: int) -> None:
        raise ValueError("bad prepare")

    logic.on_prepare = fail  # type: ignore
    with pytest.raises(ValueError, match="bad prepare"):
        await flyer.prepare(2)

    with pytest.raises(TransitionError, match="cannot go to KICKING_OFF"):
        await flyer.kickoff()
    assert "on_kickoff" not in logic.calls


# Bug: a verb cancelled by unstage hit TransitionError if a new run of the same
# verb had started by the time it finished cancelling
async def test_cancelled_verb_does_not_end_the_new_runs_state():
    cleanup = asyncio.Event()

    class SlowCancelLogic(RecordingFlyableLogic):
        async def on_prepare(self, value: int) -> None:
            if value == 1:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await cleanup.wait()  # slow cleanup
                    raise
            await cleanup.wait()

    flyer = await make_staged_flyer(SlowCancelLogic())
    first = flyer.prepare(1)
    await asyncio.sleep(0)
    await flyer.unstage()
    await flyer.stage()
    second = flyer.prepare(2)
    await asyncio.sleep(0)

    cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    await flyer.kickoff()  # the new prepare stands


# Bug: a complete() cancelled by unstage cancelled the next run's fly status if
# its cleanup finished after that run had kicked off
async def test_cancelled_complete_does_not_cancel_the_new_runs_fly():
    cleanup = asyncio.Event()

    class SlowCleanupLogic(FlyingLogic):
        async def on_complete(self, ctx: None, status: AsyncStatus) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                await cleanup.wait()  # slow cleanup

    logic = SlowCleanupLogic()
    flyer = await make_staged_flyer(logic)
    await flyer.prepare(1)
    await flyer.kickoff()
    first = flyer.complete()
    await asyncio.sleep(0)
    await flyer.unstage()
    await flyer.stage()
    await flyer.prepare(2)
    await flyer.kickoff()
    new_fly = logic.fly_status
    assert new_fly is not None

    cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not new_fly.done
    # The new run can still complete
    second = flyer.complete()
    await asyncio.sleep(0)
    assert not second.done
    await flyer.unstage()
    with pytest.raises(asyncio.CancelledError):
        await second


# A failed kickoff leaves the flyer staged, so it has to be prepared again
async def test_failed_kickoff_goes_back_to_staged(recording_flyer):
    logic, flyer = recording_flyer
    await flyer.prepare(1)

    async def fail(ctx: None) -> None:
        raise ValueError("bad kickoff")

    logic.on_kickoff = fail  # type: ignore
    with pytest.raises(ValueError, match="bad kickoff"):
        await flyer.kickoff()

    with pytest.raises(TransitionError, match="from STAGED"):
        await flyer.kickoff()
    await flyer.prepare(2)


# A complete() that does not succeed cancels the fly and leaves the flyer staged,
# whether it is cancelled directly (not via unstage) or its on_complete raises
@pytest.mark.parametrize("fail", [False, True])
async def test_unsuccessful_complete_cancels_the_fly_and_goes_to_staged(fail):
    logic = FlyingLogic(fail=fail)
    flyer = await make_staged_flyer(logic)
    await flyer.prepare(1)
    await flyer.kickoff()
    completing = flyer.complete()
    await asyncio.sleep(0)
    assert logic.fly_status is not None
    assert not logic.fly_status.done

    if fail:
        with pytest.raises(ValueError, match="bad complete"):
            await completing
    else:
        completing.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await completing
    await asyncio.sleep(0)

    assert logic.fly_status.task.cancelled()
    with pytest.raises(TransitionError, match="cannot go to COMPLETING from STAGED"):
        await flyer.complete()
    # Prepared again, it can fly again
    await flyer.prepare(2)
    await flyer.kickoff()
    await flyer.unstage()
    await asyncio.sleep(0)


async def test_tuple_context_without_an_iterator_is_the_context():
    class TupleCtxLogic(FlyableLogic[int, tuple[int, int]]):
        async def on_prepare(self, value: int) -> tuple[int, int]:
            return (value, value + 1)

        async def on_kickoff(self, ctx: tuple[int, int]) -> None: ...

        async def on_complete(self, ctx: tuple[int, int], status: None) -> None: ...

    # A tuple context whose second item is not an async iterator is the context
    flyer = await make_staged_flyer(TupleCtxLogic())
    await flyer.prepare(1)
    await flyer.kickoff()
    await flyer.complete()
