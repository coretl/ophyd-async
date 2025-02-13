import asyncio
import time
from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import Mock, call

import pytest
from bluesky.protocols import Reading
from event_model import DataKey

from ophyd_async.core import (
    AsyncConfigurable,
    AsyncReadable,
    SignalDatatypeT,
    SignalR,
    Table,
    WatchableAsyncStatus,
    Watcher,
)

from ._utils import T


def approx_value(value: Any):
    """Allow any value to be compared to another in tests.

    This is needed because numpy arrays give a numpy array back when compared,
    not a bool. This means that you can't ``assert array1==array2``. Numpy
    arrays can be wrapped with `pytest.approx`, but this doesn't work for
    `Table` instances: in this case we use `ApproxTable`.
    """
    return ApproxTable(value) if isinstance(value, Table) else pytest.approx(value)


async def assert_value(signal: SignalR[SignalDatatypeT], value: Any) -> None:
    """Assert that a Signal has the given value.

    :param signal: Signal with get_value.
    :param value: The expected value from the signal.
    """
    actual_value = await signal.get_value()
    assert approx_value(value) == actual_value


async def assert_reading(
    readable: AsyncReadable, expected_reading: dict[str, Reading]
) -> None:
    """Assert that a readable Device has the given reading.

    :param readable: Device with an async ``read()`` method to get the reading from.
    :param expected_reading: The expected reading from the readable.
    """
    actual_reading = await readable.read()
    approx_expected_reading = {
        k: dict(v, value=approx_value(expected_reading[k]["value"]))
        for k, v in expected_reading.items()
    }
    assert actual_reading == approx_expected_reading


async def assert_configuration(
    configurable: AsyncConfigurable,
    configuration: dict[str, Reading],
) -> None:
    """Assert that a configurable Device has the given configuration.

    :param configurable:
        Device with an async ``read_configuration()`` method to get the
        configuration from.
    :param configuration: The expected configuration from the configurable.
    """
    actual_configuration = await configurable.read_configuration()
    approx_expected_configuration = {
        k: dict(v, value=approx_value(configuration[k]["value"]))
        for k, v in configuration.items()
    }
    assert actual_configuration == approx_expected_configuration


async def assert_describe_signal(signal: SignalR, /, **metadata):
    """Assert the describe of a signal matches the expected metadata.

    :param signal: The signal to describe.
    :param metadata: The expected metadata.
    """
    actual_describe = await signal.describe()
    assert list(actual_describe) == [signal.name]
    (actual_datakey,) = actual_describe.values()
    expected_datakey = DataKey(source=signal.source, **metadata)
    assert actual_datakey == expected_datakey


def assert_emitted(docs: dict[str, list[dict]], **numbers: int):
    """Assert emitted document generated by running a Bluesky plan.

    :param docs: A mapping of document type -> list of documents that have been emitted.
    :param numbers: The number of each document type expected.

    :example:
    ```python
    docs = defaultdict(list)
    RE.subscribe(lambda name, doc: docs[name].append(doc))
    RE(my_plan())
    assert_emitted(docs, start=1, descriptor=1, event=1, stop=1)
    ```
    """
    assert list(docs) == list(numbers)
    actual_numbers = {name: len(d) for name, d in docs.items()}
    assert actual_numbers == numbers


class ApproxTable:
    """For approximating two tables are equivalent.

    :param expected: The expected table.
    :param rel: The relative tolerance.
    :param abs: The absolute tolerance.
    :param nan_ok: Whether NaNs are allowed.
    """

    def __init__(self, expected: Table, rel=None, abs=None, nan_ok: bool = False):
        self.expected = expected
        self.rel = rel
        self.abs = abs
        self.nan_ok = nan_ok

    def __eq__(self, value):
        approx_fields = {
            k: pytest.approx(v, self.rel, self.abs, self.nan_ok)
            for k, v in self.expected
        }
        expected = type(self.expected).model_construct(**approx_fields)  # type: ignore
        return expected == value


class MonitorQueue(AbstractContextManager):
    """Monitors a `Signal` and stores its updates."""

    def __init__(self, signal: SignalR):
        self.signal = signal
        self.updates: asyncio.Queue[dict[str, Reading]] = asyncio.Queue()
        self.signal.subscribe(self.updates.put_nowait)

    async def assert_updates(self, expected_value):
        # Get an update, value and reading
        expected_type = type(expected_value)
        expected_value = approx_value(expected_value)
        update = await asyncio.wait_for(self.updates.get(), timeout=1)
        value = await self.signal.get_value()
        reading = await self.signal.read()
        # Check they match what we expected
        assert value == expected_value
        assert type(value) is expected_type
        expected_reading = {
            self.signal.name: {
                "value": expected_value,
                "timestamp": pytest.approx(time.time(), rel=0.1),
                "alarm_severity": 0,
            }
        }
        assert reading == update == expected_reading

    def __enter__(self):
        self.signal.subscribe(self.updates.put_nowait)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.signal.clear_sub(self.updates.put_nowait)


class StatusWatcher(Watcher[T]):
    """Watches an `AsyncStatus`, storing the calls within."""

    def __init__(self, status: WatchableAsyncStatus) -> None:
        self._event = asyncio.Event()
        self._mock = Mock()
        status.watch(self)

    def __call__(
        self,
        current: T | None = None,
        initial: T | None = None,
        target: T | None = None,
        name: str | None = None,
        unit: str | None = None,
        precision: int | None = None,
        fraction: float | None = None,
        time_elapsed: float | None = None,
        time_remaining: float | None = None,
    ) -> Any:
        self._mock(
            current=current,
            initial=initial,
            target=target,
            name=name,
            unit=unit,
            precision=precision,
            fraction=fraction,
            time_elapsed=time_elapsed,
            time_remaining=time_remaining,
        )
        self._event.set()

    async def wait_for_call(
        self,
        current: T | None = None,
        initial: T | None = None,
        target: T | None = None,
        name: str | None = None,
        unit: str | None = None,
        precision: int | None = None,
        fraction: float | None = None,
        # Any so we can use pytest.approx
        time_elapsed: float | Any = None,
        time_remaining: float | Any = None,
    ):
        await asyncio.wait_for(self._event.wait(), timeout=1)
        assert self._mock.call_count == 1
        assert self._mock.call_args == call(
            current=current,
            initial=initial,
            target=target,
            name=name,
            unit=unit,
            precision=precision,
            fraction=fraction,
            time_elapsed=time_elapsed,
            time_remaining=time_remaining,
        )
        self._mock.reset_mock()
        self._event.clear()
