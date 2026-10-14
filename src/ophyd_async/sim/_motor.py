import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from functools import cached_property

from ophyd_async.core import (
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
    """What a simulated motor fly scan needs after `on_prepare`."""

    #: Where to start from: the start position with the run-up distance added.
    start: float
    #: Where to end: the end position with the run-down distance added.
    end: float
    #: The speed of the constant velocity phase.
    speed: float
    #: How long the motor takes to get up to speed.
    acceleration_time: float


@dataclass
class SimMotorLogic(
    MovableLogic[float], FlyableLogic[FlyMotorInfo, SimMotorPrepareCtx, AsyncStatus]
):
    """Combined move + fly logic for a simulated motor."""

    readback_set: Callable[[float], None]
    velocity: SignalRW[float]
    acceleration_time: SignalRW[float]

    async def stop(self) -> None:
        """Stop the motion."""
        await self.setpoint.set(await self.readback.get_value())

    async def move(self, new_position: float, timeout: TimeoutCalculator) -> None:
        old_position, velocity, acceleration_time = await asyncio.gather(
            self.readback.get_value(),
            self.velocity.get_value(),
            self.acceleration_time.get_value(),
        )

        await self.setpoint.set(new_position)
        async for position in simulate_move(
            old_position, new_position, velocity, acceleration_time
        ):
            self.readback_set(position)

    async def on_prepare(
        self, value: FlyMotorInfo
    ) -> tuple[SimMotorPrepareCtx, AsyncIterator[WatcherUpdate]]:
        """Work out the run-up and run-down positions, then move to the run-up start.

        Progress of the move is reported.
        """
        acceleration_time = await self.acceleration_time.get_value()
        ctx = SimMotorPrepareCtx(
            start=value.ramp_up_start_pos(acceleration_time),
            end=value.ramp_down_end_pos(acceleration_time),
            speed=value.speed,
            acceleration_time=acceleration_time,
        )
        return ctx, watch_readback(self, self._move_to_start(ctx), target=ctx.start)

    async def _move_to_start(self, ctx: SimMotorPrepareCtx) -> None:
        # Move to start as fast as we can
        await self.velocity.set(0)
        await self.move(ctx.start, lambda: None)
        # Set the velocity for the actual move
        await self.velocity.set(ctx.speed)

    async def on_kickoff(self, ctx: SimMotorPrepareCtx) -> AsyncStatus:
        """Begin moving motor from prepared position to final position."""
        status = AsyncStatus(self.move(ctx.end, lambda: None))
        # Wait for the acceleration time to ensure we are at velocity
        await asyncio.sleep(ctx.acceleration_time)
        return status

    async def on_complete(
        self, ctx: SimMotorPrepareCtx, status: AsyncStatus
    ) -> AsyncIterator[WatcherUpdate]:
        """Report progress until the motor reaches the completed position."""
        return watch_readback(self, status, target=ctx.end, initial=ctx.start)


# Remove InstantMovableMock as SimMotor owns this logic, depends if instant=True/False
@default_mock_class(DeviceMock)
class SimMotor(
    StandardReadable,
    StandardMovable[float],
    StandardFlyable[FlyMotorInfo, SimMotorPrepareCtx, AsyncStatus],
):
    """For usage when simulating a motor.

    Like the EPICS `Motor`, `stage()` and `unstage()` stop any motion.
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
