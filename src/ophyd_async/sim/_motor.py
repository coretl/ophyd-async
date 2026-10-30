import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from functools import cached_property

from ophyd_async.core import (
    KICKED_OFF,
    AsyncStatus,
    DeviceMock,
    FlyableLogic,
    FlyMotorInfo,
    MovableLogic,
    SignalRW,
    StandardFlyable,
    StandardMovable,
    StandardReadable,
    TimeoutCalculator,
    WatcherUpdate,
    default_mock_class,
    simulate_move,
    soft_signal_r_and_setter,
    soft_signal_rw,
    watch_readback,
)
from ophyd_async.core import StandardReadableFormat as Format


@dataclass(frozen=True)
class SimMotorPrepareCtx:
    """What a simulated motor fly scan needs after `prepare_to_fly`."""

    #: Where to start from: the start position with the run-up distance added.
    start: float
    #: Where to end: the end position with the run-down distance added.
    end: float
    #: How long the motor takes to get up to speed.
    acceleration_time: float


@dataclass
class SimMotorLogic(
    MovableLogic[float], FlyableLogic[FlyMotorInfo, SimMotorPrepareCtx]
):
    """Combined move + fly logic for a simulated motor."""

    readback_set: Callable[[float], None]
    velocity: SignalRW[float]
    acceleration_time: SignalRW[float]
    _motion: asyncio.Task | None = field(default=None, init=False, repr=False)

    async def ensure_stopped(self) -> None:
        """Stop the simulated motion, a fly included, and wait for it to finish.

        Then set the setpoint to where it stopped.
        """
        if (motion := self._motion) is not None:
            motion.cancel()
            await asyncio.wait([motion])
        await self.setpoint.set(await self.readback.get_value())

    async def move(self, new_position: float, timeout: TimeoutCalculator) -> None:
        """Simulate the move, which ends early if `ensure_stopped` stops it."""
        old_position, velocity, acceleration_time = await asyncio.gather(
            self.readback.get_value(),
            self.velocity.get_value(),
            self.acceleration_time.get_value(),
        )

        await self.setpoint.set(new_position)

        async def simulate() -> None:
            moves = simulate_move(
                old_position, new_position, velocity, acceleration_time
            )
            async for position in moves:
                self.readback_set(position)

        # In a task of its own, so that ensure_stopped can cancel it and wait
        async with AsyncStatus(simulate()) as motion:
            self._motion = motion.task
            try:
                await asyncio.wait([motion.task])
            finally:
                if self._motion is motion.task:
                    self._motion = None

    async def prepare_to_fly(
        self, name: str, value: FlyMotorInfo, previous_ctx: SimMotorPrepareCtx | None
    ) -> AsyncGenerator[SimMotorPrepareCtx | WatcherUpdate, None]:
        """Work out the run-up and run-down positions, then move to the run-up start.

        Progress of the move is reported.
        """
        acceleration_time = await self.acceleration_time.get_value()
        ctx = SimMotorPrepareCtx(
            start=value.ramp_up_start_pos(acceleration_time),
            end=value.ramp_down_end_pos(acceleration_time),
            acceleration_time=acceleration_time,
        )
        yield ctx
        # Move to start as fast as we can
        await self.velocity.set(0)
        async with AsyncStatus(self.move(ctx.start, lambda: None)) as status:
            async for update in watch_readback(self, status, target=ctx.start):
                yield update
        # Set the velocity for the actual move
        await self.velocity.set(value.speed)

    async def fly(
        self, ctx: SimMotorPrepareCtx
    ) -> AsyncGenerator[KICKED_OFF | WatcherUpdate, None]:
        """Move from the run-up start to the run-down end, reporting progress.

        Kicked off once the run-up is done and the motor is at constant velocity.
        """
        async with AsyncStatus(self.move(ctx.end, lambda: None)) as status:
            await asyncio.sleep(ctx.acceleration_time)
            yield KICKED_OFF
            async for update in watch_readback(
                self, status, target=ctx.end, initial=ctx.start
            ):
                yield update


# Remove InstantMovableMock as SimMotor owns this logic, depends if instant=True/False
@default_mock_class(DeviceMock)
class SimMotor(
    StandardReadable,
    StandardMovable[float],
    StandardFlyable[FlyMotorInfo, SimMotorPrepareCtx],
):
    """For usage when simulating a motor.

    `stage()` calls the logic's `ensure_ready`, which calls `ensure_stopped` and so
    stops the simulated motion and sets the setpoint to the current readback.
    `unstage()` cancels a fly in progress, then calls `ensure_stopped` too.
    """

    def __init__(
        self,
        name: str = "",
        instant: bool = True,
        initial_value: float = 0.0,
        units: str = "mm",
    ) -> None:
        """Simulate a motor, with optional velocity.

        :param name: name of device
        :param instant: whether to move instantly or calculate move time using velocity
        :param initial_value: initial position of the motor
        :param units: units of the motor position
        """
        # Define some signals
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.user_readback, self._user_readback_set = soft_signal_r_and_setter(
                float, initial_value, units=units
            )
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.velocity = soft_signal_rw(float, 0 if instant else 1.0)
            self.acceleration_time = soft_signal_rw(float, 0.5)
        self.user_setpoint = soft_signal_rw(float, initial_value, units=units)

        super().__init__(name=name)

    @cached_property
    def logic(self) -> SimMotorLogic:
        return SimMotorLogic(
            readback=self.user_readback,
            readback_set=self._user_readback_set,
            setpoint=self.user_setpoint,
            velocity=self.velocity,
            acceleration_time=self.acceleration_time,
        )
