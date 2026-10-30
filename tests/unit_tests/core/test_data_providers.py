import numpy as np
import pytest

from ophyd_async.core import (
    Array1D,
    EventPageDataProvider,
    soft_signal_r_and_setter,
)


def make_provider(collections_per_event: int) -> EventPageDataProvider:
    """A 5-collection buffer, filled, with one timestamp per collection."""
    data, _ = soft_signal_r_and_setter(
        Array1D[np.float64], np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    )
    timestamps, _ = soft_signal_r_and_setter(
        Array1D[np.float64], np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    )
    return EventPageDataProvider({"det-stats": data}, timestamps, collections_per_event)


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
    collections_per_event: int,
    expected_data: list[list[float]],
    expected_times: list[float],
):
    provider = make_provider(collections_per_event)
    page = await provider.make_page(5 // collections_per_event)
    assert page is not None
    np.testing.assert_array_equal(page["data"]["det-stats"], expected_data)
    assert page["time"] == expected_times


async def test_datakeys_take_dtype_from_the_signal():
    datakeys = await make_provider(5).make_datakeys()
    assert list(datakeys) == ["det-stats"]
    assert datakeys["det-stats"]["shape"] == [5]
    assert datakeys["det-stats"]["dtype"] == "array"
    assert datakeys["det-stats"]["dtype_numpy"] == "<f8"


async def test_pages_carry_on_from_the_last():
    provider = make_provider(1)
    pages = [await provider.make_page(index) for index in (2, 4, 4)]
    assert [page and page["time"] for page in pages] == [
        [100.0, 101.0],
        [102.0, 103.0],
        None,
    ]
