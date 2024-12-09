from collections.abc import Mapping
from typing import Any

import pytest
from bluesky.protocols import Reading

from ophyd_async.core import AsyncConfigurable, AsyncReadable, SignalDatatypeT, SignalR


def _generate_assert_error_msg(name: str, expected_result, actual_result) -> str:
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    return (
        f"Expected {WARNING}{name}{ENDC} to produce"
        + f"\n{FAIL}{expected_result}{ENDC}"
        + f"\nbut actually got \n{FAIL}{actual_result}{ENDC}"
    )


async def assert_value(signal: SignalR[SignalDatatypeT], value: Any) -> None:
    """Assert a signal's value and compare it an expected signal.

    Parameters
    ----------
    signal:
        signal with get_value.
    value:
        The expected value from the signal.

    Notes
    -----
    Example usage::
        await assert_value(signal, value)

    """
    actual_value = await signal.get_value()
    assert actual_value == value, _generate_assert_error_msg(
        name=signal.name,
        expected_result=value,
        actual_result=actual_value,
    )


def _approx_readable_value(reading: Mapping[str, Reading]) -> Mapping[str, Reading]:
    """Change Reading value to pytest.approx(value)"""
    for i in reading:
        reading[i]["value"] = pytest.approx(reading[i]["value"])
    return reading


async def assert_reading(
    readable: AsyncReadable, expected_reading: Mapping[str, Reading]
) -> None:
    """Assert readings from readable.

    Parameters
    ----------
    readable:
        Callable with readable.read function that generate readings.

    reading:
        The expected readings from the readable.

    Notes
    -----
    Example usage::
        await assert_reading(readable, reading)

    """
    actual_reading = await readable.read()
    assert expected_reading == _approx_readable_value(
        actual_reading
    ), _generate_assert_error_msg(
        name=readable.name,
        expected_result=expected_reading,
        actual_result=actual_reading,
    )


async def assert_configuration(
    configurable: AsyncConfigurable,
    configuration: Mapping[str, Reading],
) -> None:
    """Assert readings from Configurable.

    Parameters
    ----------
    configurable:
        Configurable with Configurable.read function that generate readings.

    configuration:
        The expected readings from configurable.

    Notes
    -----
    Example usage::
        await assert_configuration(configurable configuration)

    """
    actual_configurable = await configurable.read_configuration()
    assert configuration == _approx_readable_value(
        actual_configurable
    ), _generate_assert_error_msg(
        name=configurable.name,
        expected_result=configuration,
        actual_result=actual_configurable,
    )


def assert_emitted(docs: Mapping[str, list[dict]], **numbers: int):
    """Assert emitted document generated by running a Bluesky plan

    Parameters
    ----------
    Doc:
        A dictionary

    numbers:
        expected emission in kwarg from

    Notes
    -----
    Example usage::
        docs = defaultdict(list)
        RE.subscribe(lambda name, doc: docs[name].append(doc))
        RE(my_plan())
        assert_emitted(docs, start=1, descriptor=1, event=1, stop=1)
    """
    assert list(docs) == list(numbers), _generate_assert_error_msg(
        name="documents",
        expected_result=list(numbers),
        actual_result=list(docs),
    )
    actual_numbers = {name: len(d) for name, d in docs.items()}
    assert actual_numbers == numbers, _generate_assert_error_msg(
        name="emitted",
        expected_result=numbers,
        actual_result=actual_numbers,
    )
