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
    page = await provider.make_page(0, 5, collections_per_event)
    assert page is not None
    np.testing.assert_array_equal(page["data"]["det-stats"], expected_data)
    assert page["time"] == expected_times


async def test_datakeys_take_dtype_from_the_signal(provider: EventPageDataProvider):
    datakeys = await provider.make_datakeys(5)
    assert list(datakeys) == ["det-stats"]
    assert datakeys["det-stats"]["shape"] == [5]
    assert datakeys["det-stats"]["dtype"] == "array"
    assert datakeys["det-stats"]["dtype_numpy"] == "<f8"


@pytest.mark.parametrize(
    "collections_from,collections_to,expected_times",
    [
        (0, 2, [100.0, 101.0]),
        (2, 4, [102.0, 103.0]),
        # The same range again gives the same page: the provider holds no state
        (2, 4, [102.0, 103.0]),
        (3, 3, None),
    ],
)
async def test_pages_the_range_it_is_asked_for(
    provider: EventPageDataProvider,
    collections_from: int,
    collections_to: int,
    expected_times: list[float] | None,
):
    page = await provider.make_page(collections_from, collections_to, 1)
    assert (page and page["time"]) == expected_times


async def test_step_boundary_rearms_only_when_told_how(
    provider: EventPageDataProvider,
):
    # Without a rearm the arrays keep filling, so there is nothing to do
    await provider.step_boundary()
    rearms = []

    async def rearm():
        rearms.append(True)

    provider.rearm = rearm
    await provider.step_boundary()
    assert rearms == [True]
