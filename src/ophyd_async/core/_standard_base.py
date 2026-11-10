import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from enum import Enum
from functools import cached_property

from ._device import Device
from ._protocol import AsyncStageable
from ._status import AsyncStatus
from ._utils import _gather_or_cancel_all, cancel_and_wait


class _State(Enum):
    """Every state of `_StandardBase`, `StandardFlyable` and `StandardDetector`.

    Each class in the stack adds the states it needs and the edges between them.
    """

    # _StandardBase
    UNSTAGED = "UNSTAGED"
    STAGED = "STAGED"
    # StandardFlyable
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    FLYING = "FLYING"
    FLOWN = "FLOWN"
    # StandardDetector
    TRIGGERING = "TRIGGERING"
    TRIGGERED = "TRIGGERED"


#: States that mean a verb is running; leaving one is only allowed to the task
#: that entered it (or by `unstage()`, which cancels that task).
_IN_PROGRESS = frozenset({_State.PREPARING, _State.FLYING, _State.TRIGGERING})


class TransitionError(RuntimeError):
    """A verb was called in a state that does not allow it."""


class _StateMachine:
    """Current state of a device, the edges it may follow and the running verb."""

    def __init__(self, device: Device, edges: Mapping[_State, set[_State]]) -> None:
        self._device = device
        self._edges = edges
        self.state = _State.UNSTAGED
        self._task: asyncio.Task | None = None

    def __or__(self, edges: Mapping[_State, set[_State]]) -> "_StateMachine":
        """A new machine for the same device, with these edges added to ours."""
        return _StateMachine(
            self._device,
            {s: self._edges.get(s, set()) | edges.get(s, set()) for s in _State},
        )

    def transition(self, to: _State) -> None:
        """Move to `to`, or raise `TransitionError` if there is no such edge."""
        # Unstage is the cleanup path: works from any state
        allowed = self._edges.get(self.state, set()) | {_State.UNSTAGED}
        if to not in allowed:
            raise TransitionError(
                f"{self._device.name}: cannot go to {to.name} from "
                f"{self.state.name}, allowed: "
                f"{', '.join(sorted(s.name for s in allowed))}"
            )
        if (
            self.state in _IN_PROGRESS
            and to is not _State.UNSTAGED
            and self._task is not asyncio.current_task()
        ):
            raise TransitionError(
                f"{self._device.name}: cannot go to {to.name} from "
                f"{self.state.name}, a verb is running in another task and "
                "only unstage can end it"
            )
        self._task = asyncio.current_task() if to in _IN_PROGRESS else None
        self.state = to

    def running_verb(self) -> asyncio.Task | None:
        """The task of the verb in progress, unless it is the current task."""
        running = None
        if self.state in _IN_PROGRESS and self._task is not asyncio.current_task():
            running = self._task
        return running

    def check_one_of(self, *states: _State, verb: str) -> None:
        """Raise `TransitionError` naming `verb` unless the state is in `states`."""
        if self.state not in states:
            raise TransitionError(
                f"{self._device.name}: {verb} not allowed from {self.state.name}, "
                f"allowed from: {', '.join(sorted(s.name for s in states))}"
            )

    @asynccontextmanager
    async def during(
        self,
        state: _State,
        *,
        end: _State,
        on_error: _State,
    ) -> AsyncGenerator[None, None]:
        """Run a verb in the in-progress `state`, then go to `end` or `on_error`.

        Safe to wrap `yield`s of an async generator. If `unstage()` has moved the
        machine on in the meantime, even into a new run of the same verb, this
        verb no longer owns the state and is left alone to finish cancelling.
        """
        self.transition(state)
        try:
            yield
        except BaseException:
            # Can be false: unstage cancelled this verb and moved the state on,
            # or an abandoned async generator was closed later from another task
            if self.state is state and self._task is asyncio.current_task():
                self.transition(on_error)
            raise
        else:
            if self.state is state and self._task is asyncio.current_task():
                self.transition(end)


class StoppedError(RuntimeError):
    """A set, prepare or fly was ended early by `stop()` or a new `set()`."""


class Logic(ABC):
    """Base of `MovableLogic` and `FlyableLogic`: stopping."""

    @abstractmethod
    async def ensure_stopped(self) -> None:
        """Stop the hardware and wait until it has stopped.

        Called by `StandardMovable.stop()`, which bluesky calls on pause and at
        the end of a run, and by `StandardFlyable` on `unstage()`.
        """


class _StandardBase(Device, AsyncStageable):
    """Base for `Device` mix-ins that contribute to `stage()`/`unstage()`.

    `StandardReadable`, `StandardFlyable` and similar mix-ins each need to run
    something on `stage()`/`unstage()`. They register callables into
    `_stage_funcs`/`_unstage_funcs`, which this base gathers. Each mix-in names
    its hooks `__on_stage`/`__on_unstage`, name-mangled so they never collide.
    """

    # Immutable defaults to avoid accidental sharing between instances
    _stage_funcs: tuple[Callable[[], Awaitable[None]], ...] = ()
    _unstage_funcs: tuple[Callable[[], Awaitable[None]], ...] = ()

    @cached_property
    def _state(self) -> _StateMachine:
        # cached_property as Device registration runs before super().__init__
        return _StateMachine(self, {_State.UNSTAGED: {_State.STAGED}})

    @AsyncStatus.wrap
    async def stage(self) -> None:
        self._state.transition(_State.STAGED)
        await _gather_or_cancel_all([func() for func in self._stage_funcs])

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        verb = self._state.running_verb()
        self._state.transition(_State.UNSTAGED)
        # Made now, so a hook that is not an async def takes its state before a
        # restage; an async def hook reads its state when it runs, below
        unstaging = [func() for func in self._unstage_funcs]
        try:
            if verb is not None:
                await cancel_and_wait(verb)
        finally:
            await _gather_or_cancel_all(unstaging)
