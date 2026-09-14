import numpy as np
import pytest

from ophyd_async.core import (
    Array1D,
    EventPageDataProvider,
    soft_signal_r_and_setter,
)


@pytest.fixture
def provider() -> EventPageDataProvider:
    """A 5-collection buffer, filled, with one timestamp per collection."""
    data, _ = soft_signal_r_and_setter(
        Array1D[np.float64], np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    )
    timestamps, _ = soft_signal_r_and_setter(
        Array1D[np.float64], np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    )
    written, _ = soft_signal_r_and_setter(int, 5)
    return EventPageDataProvider({"det-stats": data}, written, timestamps)


@pytest.mark.parametrize(
    "collections_per_event,expected_data,expected_times",
    [
        # step scan: one event holds the whole 5-collection buffer, timed by its
        # last collection
        (5, [[10.0, 11.0, 12.0, 13.0, 14.0]], [104.0]),
        # fly scan: five events of one collection each, timed one by one
        (
            1,
            [[10.0], [11.0], [12.0], [13.0], [14.0]],
            [100.0, 101.0, 102.0, 103.0, 104.0],
        ),
    ],
)
async def test_slices_the_arrays_into_events(
    provider: EventPageDataProvider,
    collections_per_event: int,
    expected_data: list[list[float]],
    expected_times: list[float],
):
    pages = [
        page
        async for page in provider.make_pages(
            collections_written=5, collections_per_event=collections_per_event
        )
    ]
    (page,) = pages
    np.testing.assert_array_equal(page["data"]["det-stats"], expected_data)
    assert page["time"] == expected_times


async def test_datakeys_take_dtype_from_the_signal(provider: EventPageDataProvider):
    datakeys = await provider.make_datakeys(5)
    assert list(datakeys) == ["det-stats"]
    assert datakeys["det-stats"]["shape"] == [5]
    assert datakeys["det-stats"]["dtype"] == "array"
    assert datakeys["det-stats"]["dtype_numpy"] == "<f8"


async def test_only_emits_collections_it_has_not_emitted(
    provider: EventPageDataProvider,
):
    first = [page async for page in provider.make_pages(2, 1)]
    assert [p["time"] for p in first] == [[100.0, 101.0]]
    # Asked again for the same collections, there is nothing new to emit
    assert [page async for page in provider.make_pages(2, 1)] == []
    # Only what has arrived since
    second = [page async for page in provider.make_pages(4, 1)]
    assert [p["time"] for p in second] == [[102.0, 103.0]]
