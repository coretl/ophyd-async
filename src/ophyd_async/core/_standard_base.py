import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from enum import Enum
from functools import cached_property

from ._device import Device
from ._protocol import AsyncStageable
from ._status import AsyncStatus


class _State(Enum):
    """Every state of the linear stack `_StandardBase` <- `StandardFlyable`.

    Each class in the stack adds the states it needs and the edges between them.
    """

    # _StandardBase
    UNSTAGED = "UNSTAGED"
    STAGED = "STAGED"
    # StandardFlyable
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    KICKING_OFF = "KICKING_OFF"
    KICKED_OFF = "KICKED_OFF"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"


#: States that mean a verb is running; leaving one is only allowed to the task
#: that entered it (or by `unstage()`, which cancels that task).
_IN_PROGRESS = frozenset({_State.PREPARING, _State.KICKING_OFF, _State.COMPLETING})


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
        # Unstage is the cleanup path: works from any state, cancels the verb
        allowed = self._edges.get(self.state, set()) | {_State.UNSTAGED}
        if to not in allowed:
            raise TransitionError(
                f"{self._device.name}: cannot go to {to.name} from "
                f"{self.state.name}, allowed: "
                f"{', '.join(sorted(s.name for s in allowed))}"
            )
        if self.state in _IN_PROGRESS:
            if to is _State.UNSTAGED:
                # Unstage is the only thing that ends a verb from outside
                if self._task is not None and self._task is not asyncio.current_task():
                    self._task.cancel()
            elif self._task is not asyncio.current_task():
                raise TransitionError(
                    f"{self._device.name}: cannot go to {to.name} from "
                    f"{self.state.name}, a verb is running in another task and "
                    "only unstage can end it"
                )
            self._task = None
        if to in _IN_PROGRESS:
            self._task = asyncio.current_task()
        self.state = to

    @asynccontextmanager
    async def during(
        self, state: _State, *, end: _State, on_error: _State
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


class _StandardBase(Device, AsyncStageable):
    """Base for `Device` mix-ins that contribute to `stage()`/`unstage()`.

    `StandardReadable`, `StandardFlyable` and similar mix-ins each need to run
    something on `stage()`/`unstage()`. Rather than each defining those methods
    and shadowing the others via the MRO when combined on one `Device` (e.g. a
    motor that is Readable + Movable + Flyable), they register callables into
    `_stage_funcs`/`_unstage_funcs`, which this single base gathers.
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
        await asyncio.gather(*(func() for func in self._stage_funcs))

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        self._state.transition(_State.UNSTAGED)
        await asyncio.gather(*(func() for func in self._unstage_funcs))
