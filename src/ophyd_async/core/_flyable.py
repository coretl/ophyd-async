from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import replace
from functools import cached_property
from typing import Any, Generic, TypeVar, cast

from bluesky.protocols import Flyable, Preparable
from pydantic import Field

from ._standard_base import _StandardBase, _State, _StateMachine
from ._status import AsyncStatus, WatchableAsyncStatus
from ._utils import (
    CALCULATE_TIMEOUT,
    CalculatableTimeout,
    ConfinedModel,
    CtxT,
    FlyStatusT,
    WatcherUpdate,
    abstract_cached_property,
)

#: The per-scan info passed to `FlyableLogic.on_prepare` (often a
#: [`scanspec.core.Path`](inv:scanspec#scanspec.core.Path), but any type).
PrepareT = TypeVar("PrepareT")


class FlyableLogic(Generic[PrepareT, CtxT, FlyStatusT]):
    """Minimum logic needed for controlling a `StandardFlyable`.

    Inherit and fill in the hooks for a particular flyable (e.g. a motion or
    trigger system). This base holds no state; concrete subclasses are typically
    `@dataclass`es that hold whatever signals or sub-devices they need as fields.

    What a scan needs between stages is carried by a context object rather than
    stored on the logic: `on_prepare` works it out and returns it, and it is
    passed to every later hook. It is fixed from then on (use a frozen dataclass
    where practical); anything that changes during a scan is an object the
    context holds. `StandardFlyable` owns the context and enforces the call
    ordering, so subclasses do not need to guard against being called out of
    order. A flyer with no cross-stage state leaves the context type as `None`
    (see the PandA trigger logics).

    The type parameters are the per-scan info `PrepareT`, the context `CtxT` and
    the status `FlyStatusT` that `on_kickoff` returns; the last two default to
    `None`, so `FlyableLogic[PrepareT]` is a logic with neither.

    The hook names are deliberately not bluesky verbs, so a logic passed to a
    plan stub does not look like a device.
    """

    @abstractmethod
    async def on_prepare(
        self, value: PrepareT
    ) -> CtxT | tuple[CtxT, AsyncIterator[WatcherUpdate]]:
        """Work out the scan, set up the hardware, and return the scan's context.

        To report progress to watchers (see `watch_readback`) while the slow part
        of prepare runs, e.g. moving to the start of the scan, return the context
        together with an async iterator of `WatcherUpdate` as a 2-tuple; the
        iterator is run after the context is made. A context that is itself a
        2-tuple ending in an async iterator would be misread as this.

        :param value: the per-scan info for this fly scan.
        """

    @abstractmethod
    async def on_kickoff(self, ctx: CtxT) -> FlyStatusT:
        """Start the fly scan, returning once it is up to speed.

        Return the status of what is now running, to be passed to `on_complete`,
        or `None` if there is nothing to wait for. `StandardFlyable` cancels it
        on `unstage()`, and when `complete()` is cancelled or fails.

        If this awaits after starting the status, it is its job to cancel that
        status if it is itself cancelled.

        :param ctx: the context returned by `on_prepare`.
        """

    @abstractmethod
    async def on_complete(
        self, ctx: CtxT, status: FlyStatusT
    ) -> AsyncIterator[WatcherUpdate] | None:
        """Wait for the end of the fly scan.

        Either wait here and return `None`, or return an async iterator of
        `WatcherUpdate` to report progress to watchers (see `watch_readback`),
        which finishes when the fly scan does.

        :param ctx: the context returned by `on_prepare`.
        :param status: what `on_kickoff` returned.
        """

    async def stop(self) -> None:
        """Stop/disarm the flyer and wait for everything to be stopped.

        Called by `on_stage` and `on_unstage` by default; override those instead
        if stage and unstage need to differ.

        Deliberately the same name as `MovableLogic.stop`, so that a logic
        implementing both -- a motor record's, which is moved *and* flown --
        writes one stop that serves as the move-stop verb and as fly-scan
        teardown. The hardware action is the same in both cases.
        """
        pass

    async def on_stage(self) -> None:
        """Set the flyer up on `stage()`. Defaults to `stop`."""
        await self.stop()

    async def on_unstage(self) -> None:
        """Clean the flyer up on `unstage()`. Defaults to `stop`."""
        await self.stop()

    def with_device(self, name: str = "") -> "StandardFlyable[Any, Any, Any]":
        """Wrap this logic in an ephemeral `StandardFlyable` for use in a plan."""
        flyer = _EphemeralFlyable[PrepareT, CtxT, FlyStatusT](name=name)
        # cached_property writes straight to the instance __dict__, so this
        # satisfies the abstract declaration without a subclass per logic type.
        flyer.logic = self
        return flyer


class FlyMotorInfo(ConfinedModel):
    """Minimal set of information required to fly a motor."""

    start_position: float = Field(frozen=True)
    """Absolute position of the motor once it finishes accelerating to desired
    velocity, in motor EGUs"""

    end_position: float = Field(frozen=True)
    """Absolute position of the motor once it begins decelerating from desired
    velocity, in EGUs"""

    time_for_move: float = Field(frozen=True, gt=0)
    """Time taken for the motor to get from start_position to end_position, excluding
    run-up and run-down, in seconds."""

    timeout: CalculatableTimeout = Field(frozen=True, default=CALCULATE_TIMEOUT)
    """Maximum time for the complete motor move, including run up and run down.
    Defaults to `time_for_move` + run up and run down times + 10s."""

    @property
    def speed(self) -> float:
        """Calculate the speed of the constant velocity phase, always positive."""
        return abs(self.end_position - self.start_position) / self.time_for_move

    def ramp_up_start_pos(self, acceleration_time: float) -> float:
        """Calculate the start position with run-up distance added on."""
        return self.start_position - self._ramp_distance(acceleration_time)

    def ramp_down_end_pos(self, acceleration_time: float) -> float:
        """Calculate the end position with run-down distance added on."""
        return self.end_position + self._ramp_distance(acceleration_time)

    def _ramp_distance(self, acceleration_time: float) -> float:
        # Signed so run-up/run-down land on the correct side for either direction.
        return (
            acceleration_time
            * (self.end_position - self.start_position)
            / (2 * self.time_for_move)
        )


class StandardFlyable(
    _StandardBase,
    Preparable,
    Flyable,
    Generic[PrepareT, CtxT, FlyStatusT],
):
    """Device that provides standard logic for flying.

    This class must be inherited and have a `logic` @cached_property.
    For an ephemeral flyer in a plan, call `FlyableLogic.with_device` instead of
    inheriting. It must be staged before it is prepared. `prepare()`, `kickoff()`
    and `complete()` must then be called in that order, otherwise they raise
    `TransitionError`; `complete()` ends in the completed state, from which it can
    be prepared again. It holds the context returned by the logic's
    `on_prepare` and the status returned by its `on_kickoff`.

    `stage()` runs the logic's `on_stage` and resets the fly; it raises unless the
    device is unstaged. `unstage()` runs the logic's `on_unstage` (both default to
    `stop`), cancels the running `prepare()`, `kickoff()` or `complete()` and the
    status held from `on_kickoff`, and resets the fly.
    A `complete()` that is cancelled or fails cancels the held status and leaves
    the device staged. `prepare()` and `complete()` report to watchers whatever
    progress the logic's hooks return, named after this device; `kickoff()`
    reports none.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Contribute the flyer's stage/unstage hooks (composes with any other
        # _StandardBase mix-in, e.g. StandardReadable on a Motor).
        self._stage_funcs += (self._on_stage,)
        self._unstage_funcs += (self._on_unstage,)
        # Context and status threaded prepare -> kickoff -> complete. The context
        # is only set by prepare(), and the state machine stops anything reading
        # it before then.
        self._fly_ctx: CtxT
        self._fly_status: FlyStatusT | None = None

    @cached_property
    def _state(self) -> _StateMachine:
        # super()._state caches the base machine in the instance first, then this
        # property overwrites it with the combined one
        return super()._state | {
            _State.STAGED: {_State.PREPARING},
            _State.PREPARING: {_State.PREPARED, _State.STAGED},
            _State.PREPARED: {_State.PREPARING, _State.KICKING_OFF},
            _State.KICKING_OFF: {_State.KICKED_OFF, _State.STAGED},
            _State.KICKED_OFF: {_State.COMPLETING},
            _State.COMPLETING: {_State.COMPLETED, _State.STAGED},
            _State.COMPLETED: {_State.PREPARING},
        }

    @abstract_cached_property
    def logic(self) -> FlyableLogic[PrepareT, CtxT, FlyStatusT]:
        """The logic object that describes how this device flies.

        A static flyer (e.g. `Motor`) provides this as a `@cached_property` that
        builds a `FlyableLogic` from its signals. An ephemeral flyer created by
        `FlyableLogic.with_device` has it set directly on the instance.

        A Device that is flyable *and* movable implements this once with a logic
        object inheriting both `FlyableLogic` and `MovableLogic`; each mix-in
        declares `logic` with its own required type, so the type
        checker verifies the one implementation against both.
        """
        raise NotImplementedError

    def _reset_fly_state(self) -> None:
        # The state machine has already cancelled any verb in flight; this ends
        # our wait for what the kickoff started.
        if self._fly_status is not None and not self._fly_status.done:
            self._fly_status.task.cancel()
        self._fly_status = None

    async def _on_stage(self) -> None:
        await self.logic.on_stage()
        self._reset_fly_state()

    async def _on_unstage(self) -> None:
        await self.logic.on_unstage()
        self._reset_fly_state()

    async def _on_prepare_context(self, ctx: CtxT) -> None:
        """Called by `prepare()` with the context just made, before its progress runs.

        Does nothing; a subclass overrides it to start what the rest of the
        prepare, and later verbs, depend on.
        """

    @WatchableAsyncStatus.wrap
    async def prepare(self, value: PrepareT) -> AsyncIterator[WatcherUpdate]:
        """Set up the fly scan and move to its start.

        Reports whatever progress the logic's `on_prepare` returns.
        """
        # A failed prepare must leave nothing to kick off
        async with self._state.during(
            _State.PREPARING, end=_State.PREPARED, on_error=_State.STAGED
        ):
            match await self.logic.on_prepare(value):
                case (ctx, AsyncIterator() as progress):
                    pass
                case result:
                    ctx, progress = cast(CtxT, result), None
            await self._on_prepare_context(ctx)
            if progress is not None:
                async for update in progress:
                    yield replace(update, name=self.name)
            self._fly_ctx = ctx

    @AsyncStatus.wrap
    async def kickoff(self) -> None:
        """Start the fly scan, returning once it is up to speed."""
        async with self._state.during(
            _State.KICKING_OFF, end=_State.KICKED_OFF, on_error=_State.STAGED
        ):
            self._fly_status = await self.logic.on_kickoff(self._fly_ctx)

    @WatchableAsyncStatus.wrap
    async def complete(self) -> AsyncIterator[WatcherUpdate]:
        """Block until the fly scan is done.

        Reports whatever progress the logic's `on_complete` returns. If it is
        cancelled or fails, the status held from `on_kickoff` is cancelled and the
        device is left staged.
        """
        async with self._state.during(
            _State.COMPLETING, end=_State.COMPLETED, on_error=_State.STAGED
        ):
            try:
                progress = await self.logic.on_complete(
                    self._fly_ctx,
                    # None only if kickoff returned nothing, which FlyStatusT says
                    cast(FlyStatusT, self._fly_status),
                )
                if progress is not None:
                    async for update in progress:
                        yield replace(update, name=self.name)
            except BaseException:
                self._reset_fly_state()
                raise
            # Completed: nothing left to wait for
            self._fly_status = None


class _EphemeralFlyable(StandardFlyable[PrepareT, CtxT, FlyStatusT]):
    """A concrete `StandardFlyable` whose logic is set on the instance.

    `StandardFlyable.logic` is abstract, so the class itself cannot be
    instantiated; `FlyableLogic.with_device` needs something concrete to attach
    a logic object to.
    """

    @cached_property
    def logic(self) -> FlyableLogic[PrepareT, CtxT, FlyStatusT]:
        raise NotImplementedError(
            f"{self.name}: no logic set; create this via FlyableLogic.with_device()"
        )
