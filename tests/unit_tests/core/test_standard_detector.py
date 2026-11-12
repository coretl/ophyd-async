import asyncio
import dataclasses
import re
from collections.abc import Sequence
from unittest.mock import ANY

import numpy as np
import pytest
from bluesky.protocols import EventPageCollectable, WritesStreamAssets
from event_model import DataKey
from event_model.documents import PartialEventPage

from ophyd_async.core import (
    Array1D,
    BoundedDataLogic,
    DataProvider,
    DetectorAcquireLogic,
    DetectorLogic,
    DetectorSuppressLogic,
    DetectorTrigger,
    DetectorTriggerLogic,
    PageableDataProvider,
    Settings,
    SignalDict,
    SignalR,
    StandardReadable,
    StreamResourceDataProvider,
    StreamResourceInfo,
    TransitionError,
    TriggerInfo,
    UnboundedDataLogic,
    soft_signal_r_and_setter,
    soft_signal_rw,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.testing import (
    assert_configuration,
    assert_reading,
    wait_for_pending_wakeups,
)

# Test Logic Class Implementations


class JustInternalTriggerLogic(DetectorTriggerLogic):
    """Only supports internal triggering."""

    num: int
    livetime: float
    deadtime: float

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        self.num, self.livetime, self.deadtime = num, livetime, deadtime

    async def default_trigger_info(self) -> TriggerInfo:
        return TriggerInfo()


class AllTriggerTypesLogic(DetectorTriggerLogic):
    """Supports all types of triggering."""

    trigger: DetectorTrigger
    num: int | None = None
    livetime: float | None = None
    deadtime: float | None = None

    def get_deadtime(self, config_values: SignalDict) -> float:
        return 0.001

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        self.trigger = DetectorTrigger.INTERNAL
        self.num, self.livetime, self.deadtime = num, livetime, deadtime

    async def prepare_edge(self, num: int, livetime: float):
        self.trigger = DetectorTrigger.EXTERNAL_EDGE
        self.num, self.livetime = num, livetime

    async def prepare_level(self, num: int):
        self.trigger = DetectorTrigger.EXTERNAL_LEVEL
        self.num = num


class AveragingTriggerLogic(DetectorTriggerLogic):
    """Supports exposures per collection averaging."""

    exposures_per_collection: int
    num: int
    livetime: float
    deadtime: float

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        self.num, self.livetime, self.deadtime = num, livetime, deadtime

    async def prepare_exposures_per_collection(self, exposures_per_collection: int):
        self.exposures_per_collection = exposures_per_collection


class DeadtimeTriggerLogic(DetectorTriggerLogic):
    """Trigger logic that calculates deadtime from a signal."""

    num: int
    livetime: float
    deadtime: float

    def __init__(self, deadtime_signal: SignalR[float]):
        self.deadtime_signal = deadtime_signal

    def get_deadtime(self, config_values: SignalDict) -> float:
        """Return the deadtime from the signal value.

        The signal reaches us because the detector registers it as
        CONFIG_SIGNAL, not because this logic nominates it.
        """
        return config_values[self.deadtime_signal]

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        self.num, self.livetime, self.deadtime = num, livetime, deadtime


class MockAcquireLogic(DetectorAcquireLogic):
    """Mock acquire logic that tracks state."""

    armed = False
    arm_count = 0
    disarm_count = 0

    async def acquire(self):
        self.armed = True
        self.arm_count += 1
        yield
        await asyncio.sleep(0.001)
        self.armed = False

    async def ensure_stopped(self):
        self.armed = False
        self.disarm_count += 1


class StreamableOnlyDataLogic(UnboundedDataLogic[StreamResourceDataProvider]):
    """Produces only streamable (file-based) data."""

    def __init__(
        self,
        tmp_path,
        datakey_suffix: str = "",
        shape: tuple[int, ...] = (10, 15),
        dtype_numpy: str = "|u1",
    ):
        self.collections_written = soft_signal_rw(int)
        self.stop_count = 0
        self.tmp_path = tmp_path
        self.datakey_suffix = datakey_suffix
        self.shape = shape
        self.dtype_numpy = dtype_numpy

    async def start_provider(
        self,
        name: str,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> StreamResourceDataProvider:
        resource = StreamResourceInfo(
            data_key=name + self.datakey_suffix,
            shape=self.shape,
            chunk_shape=(1, *self.shape),
            dtype_numpy=self.dtype_numpy,
            parameters={"dataset": "/data"},
        )
        return StreamResourceDataProvider(
            uri=f"file://localhost/{self.tmp_path.as_posix().lstrip('/')}/test.h5",
            resources=[resource],
            mimetype="application/x-hdf5",
            collections_written=self.collections_written,
            collections_per_event=collections_per_event,
        )

    async def ensure_stopped(self) -> None:
        self.stop_count += 1

    def get_hinted_fields(self, name: str) -> Sequence[str]:
        return [name + self.datakey_suffix]


async def test_stream_resource_data_provider_1d_single_collection_is_array():
    provider = StreamResourceDataProvider(
        uri="file://localhost/tmp/test.h5",
        resources=[
            StreamResourceInfo(
                data_key="det-spectrum",
                shape=(3,),
                chunk_shape=(1, 3),
                dtype_numpy="<f8",
                parameters={"dataset": "/entry/data/spectrum"},
            )
        ],
        mimetype="application/x-hdf5",
        collections_written=soft_signal_rw(int),
        collections_per_event=1,
    )

    assert await provider.make_datakeys() == {
        "det-spectrum": {
            "dtype": "array",
            "dtype_numpy": "<f8",
            "external": "STREAM:",
            "shape": [1, 3],
            "source": "file://localhost/tmp/test.h5",
        }
    }


class MockPageableProvider(PageableDataProvider):
    """A finite buffer emitted as event pages, one value per collection."""

    def __init__(
        self,
        datakey_name: str,
        collections_written: SignalR[int],
        collections_per_event: int,
    ):
        super().__init__(collections_written, collections_per_event)
        self.datakey_name = datakey_name
        self.start_new_resources(0)

    def start_new_resources(self, origin: int) -> None:
        self.published = 0

    async def make_datakeys(self) -> dict[str, DataKey]:
        return {
            self.datakey_name: DataKey(
                source="mock",
                shape=[self.collections_per_event],
                dtype="array",
                dtype_numpy="<i8",
            )
        }

    def _page(self, events: int) -> PartialEventPage:
        new = range(events)
        return {
            "data": {
                self.datakey_name: [[0] * self.collections_per_event for _ in new]
            },
            "time": [0.0 for _ in new],
            "timestamps": {self.datakey_name: [0.0 for _ in new]},
        }

    async def make_page(self, index: int) -> PartialEventPage | None:
        if index <= self.published:
            return None
        page = self._page(index - self.published)
        self.published = index
        return page


class BoundedOnlyDataLogic(BoundedDataLogic[MockPageableProvider]):
    """Produces bounded data held in a finite buffer, sized when it is armed."""

    def __init__(self, datakey_suffix: str = ""):
        self.datakey_suffix = datakey_suffix
        self.collections_written = soft_signal_rw(int)
        self.prepare_calls: list[tuple[int, float]] = []
        self.stop_count = 0
        self.providers: list[MockPageableProvider] = []

    async def start_provider(
        self,
        name: str,
        existing_datakeys: Sequence[str],
        num_collections: int,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> MockPageableProvider | None:
        datakey_name = name + self.datakey_suffix
        if datakey_name in existing_datakeys:
            return None
        self.prepare_calls.append((num_collections, collection_period))
        # A real buffer clears its progress counter when armed
        await self.collections_written.set(0)
        provider = MockPageableProvider(
            datakey_name, self.collections_written, collections_per_event
        )
        self.providers.append(provider)
        return provider

    async def ensure_stopped(self) -> None:
        self.stop_count += 1

    def get_hinted_fields(self, name: str) -> Sequence[str]:
        return [name + self.datakey_suffix]


# Parameterized Tests for Logic Combinations


@pytest.mark.parametrize(
    "trigger_logic,expected_triggers,expected_deadtime",
    [
        (None, {DetectorTrigger.INTERNAL}, None),
        (JustInternalTriggerLogic(), {DetectorTrigger.INTERNAL}, None),
        (
            AllTriggerTypesLogic(),
            {
                DetectorTrigger.INTERNAL,
                DetectorTrigger.EXTERNAL_EDGE,
                DetectorTrigger.EXTERNAL_LEVEL,
            },
            0.001,
        ),
        (
            DeadtimeTriggerLogic(soft_signal_rw(float, 0.02)),
            {DetectorTrigger.INTERNAL},
            0.02,
        ),
    ],
)
async def test_get_trigger_deadtime(
    trigger_logic, expected_triggers, expected_deadtime
):
    """Test get_trigger_deadtime with various trigger logic implementations."""
    det = DetectorLogic(trigger=trigger_logic).with_device()
    await det.stage()
    if isinstance(trigger_logic, DeadtimeTriggerLogic):
        # A logic that needs a signal declares it as configuration; the logic
        # no longer nominates signals separately
        det.set_readable_format(trigger_logic.deadtime_signal, Format.CONFIG_SIGNAL)
    triggers, deadtime = await det.get_trigger_deadtime()
    assert triggers == expected_triggers
    assert deadtime == expected_deadtime


async def test_get_trigger_deadtime_with_settings():
    """Test get_trigger_deadtime using values from a Settings object."""
    # Create a signal for deadtime and set its initial value
    deadtime_signal = soft_signal_rw(float, 0.02)

    # Create detector with DeadtimeTriggerLogic
    det = DetectorLogic(trigger=DeadtimeTriggerLogic(deadtime_signal)).with_device()
    await det.stage()
    det.sig = deadtime_signal
    det.set_readable_format(deadtime_signal, Format.CONFIG_SIGNAL)

    # Verify initial deadtime from signal
    triggers, deadtime = await det.get_trigger_deadtime()
    assert deadtime == 0.02

    # Create settings with a different deadtime value
    settings = Settings(det, {deadtime_signal: 0.05})

    # Verify deadtime from settings overrides signal value
    triggers, deadtime = await det.get_trigger_deadtime(settings)
    assert triggers == {DetectorTrigger.INTERNAL}
    assert deadtime == 0.05

    # Verify signal value hasn't changed
    assert await deadtime_signal.get_value() == 0.02


@pytest.mark.parametrize(
    "trigger_type",
    [
        DetectorTrigger.INTERNAL,
        DetectorTrigger.EXTERNAL_EDGE,
        DetectorTrigger.EXTERNAL_LEVEL,
    ],
)
async def test_prepare_trigger_types(trigger_type):
    """Test each trigger type is properly delegated to trigger logic."""
    trigger_logic = AllTriggerTypesLogic()
    det = DetectorLogic(trigger=trigger_logic).with_device()
    await det.stage()

    trigger_info = TriggerInfo(
        trigger=trigger_type, livetime=0.5, deadtime=0.1, number_of_events=10
    )
    await det.prepare(trigger_info)

    # Verify the right prepare method was called
    assert trigger_logic.trigger == trigger_type
    assert trigger_logic.num == 10


async def test_prepare_unsupported_trigger_type():
    """Test that preparing with unsupported trigger type raises error."""
    det = DetectorLogic(trigger=JustInternalTriggerLogic()).with_device()
    await det.stage()

    with pytest.raises(ValueError, match="Trigger type.*EXTERNAL_EDGE not supported"):
        await det.prepare(TriggerInfo(trigger=DetectorTrigger.EXTERNAL_EDGE))


@pytest.mark.parametrize(
    "exposures_per_collection,collections_per_event,number_of_events,"
    "expected_exposures,expected_number_of_collections",
    [
        (1, 1, 1, 1, 1),
        (2, 1, 1, 2, 1),
        (1, 3, 5, 15, 15),
        (2, 3, 5, 30, 15),
        (4, 2, 10, 80, 20),
    ],
)
async def test_trigger_info_calculations(
    exposures_per_collection,
    collections_per_event,
    number_of_events,
    expected_exposures,
    expected_number_of_collections,
):
    """Verify TriggerInfo correctly computes number_of_exposures."""
    info = TriggerInfo(
        exposures_per_collection=exposures_per_collection,
        collections_per_event=collections_per_event,
        number_of_events=number_of_events,
    )
    assert info.number_of_exposures == expected_exposures
    assert info.number_of_collections == expected_number_of_collections


@pytest.mark.parametrize("exposures_per_collection", [1, 2, 5, 10])
async def test_exposures_per_collection(exposures_per_collection):
    """Test exposure averaging configuration."""
    tl = AveragingTriggerLogic()
    det = DetectorLogic(trigger=tl).with_device()
    await det.stage()

    await det.prepare(
        TriggerInfo(
            exposures_per_collection=exposures_per_collection, number_of_events=5
        )
    )

    assert tl.exposures_per_collection == exposures_per_collection
    # num should be number_of_exposures (events * collections_per_event * exposures)
    assert tl.num == 5 * 1 * exposures_per_collection


async def test_exposures_per_collection_not_supported():
    """Test that exposures_per_collection > 1 fails without supporting logic."""
    det = DetectorLogic(trigger=JustInternalTriggerLogic()).with_device()
    await det.stage()

    with pytest.raises(
        ValueError, match="Multiple exposures per collection not supported"
    ):
        await det.prepare(TriggerInfo(exposures_per_collection=5))


@pytest.mark.parametrize(
    "trigger_type,arm_timing",
    [
        (DetectorTrigger.INTERNAL, "kickoff"),
        (DetectorTrigger.EXTERNAL_EDGE, "prepare"),
        (DetectorTrigger.EXTERNAL_LEVEL, "prepare"),
    ],
)
async def test_arm_timing(trigger_type, arm_timing, tmp_path):
    """Verify detector is armed at the correct time based on trigger type."""
    tl = AllTriggerTypesLogic()
    al = MockAcquireLogic()
    dl = StreamableOnlyDataLogic(tmp_path)
    det = DetectorLogic(trigger=tl, acquire=al, data=[dl]).with_device()
    await det.stage()

    # Prepare the detector
    await det.prepare(TriggerInfo(trigger=trigger_type, number_of_events=2))

    if arm_timing == "prepare":
        # External triggers should arm during prepare
        assert al.arm_count == 1
        assert al.armed is True
    else:
        # Internal triggers should not arm during prepare
        assert al.arm_count == 0
        assert al.armed is False

    # Kickoff
    await det.kickoff()

    if arm_timing == "kickoff":
        # Internal triggers should arm during kickoff
        assert al.arm_count == 1
        assert al.armed is True
    else:
        # External triggers should still be armed from prepare
        assert al.arm_count == 1
        assert al.armed is True
    await det.unstage()


@pytest.mark.parametrize(
    "trigger_type,arms_at_trigger",
    [
        (DetectorTrigger.INTERNAL, True),
        (DetectorTrigger.EXTERNAL_EDGE, False),
        (DetectorTrigger.EXTERNAL_LEVEL, False),
    ],
)
async def test_trigger_arms_only_an_internally_triggered_detector(
    trigger_type, arms_at_trigger, tmp_path
):
    """trigger() arms the detector, unless prepare() already did.

    An externally triggered detector was armed by prepare() and is sitting
    waiting on its trigger source, so arming it again would restart it.
    """
    al = MockAcquireLogic()
    dl = StreamableOnlyDataLogic(tmp_path)
    det = DetectorLogic(
        trigger=AllTriggerTypesLogic(), acquire=al, data=[dl]
    ).with_device()
    await det.stage()

    await det.prepare(TriggerInfo(trigger=trigger_type))
    assert al.arm_count == (0 if arms_at_trigger else 1)
    assert al.armed is not arms_at_trigger

    status = det.trigger()
    # Give it a moment to arm
    await wait_for_pending_wakeups(raise_if_exceeded=False)
    # Armed either way by now, but only ever once
    assert al.armed is True
    assert al.arm_count == 1

    # Complete the trigger
    await dl.collections_written.set(1)
    await status


class NoYieldAcquireLogic(MockAcquireLogic):
    async def acquire(self):
        return
        yield


@pytest.mark.parametrize("entry", ["trigger", "fly", "external_fly"])
async def test_acquire_that_never_yields_raises(tmp_path, entry):
    dl = StreamableOnlyDataLogic(tmp_path)
    det = DetectorLogic(
        acquire=NoYieldAcquireLogic(), data=[dl], trigger=AllTriggerTypesLogic()
    ).with_device()
    await det.stage()
    external = entry == "external_fly"
    info = TriggerInfo(
        trigger=DetectorTrigger.EXTERNAL_EDGE if external else DetectorTrigger.INTERNAL,
        livetime=0.1,
    )
    with pytest.raises(
        RuntimeError,
        match=r"NoYieldAcquireLogic\.acquire\(\) finished without yielding",
    ):
        await det.prepare(info)
        if entry == "trigger":
            status = det.trigger()
        else:
            await det.kickoff()
            status = det.complete()
        await wait_for_pending_wakeups()
        await dl.collections_written.set(1)
        await status


async def test_arm_logic_called_on_stage():
    """Test that acquire logic is stopped on stage."""
    al = MockAcquireLogic()
    det = DetectorLogic(acquire=al).with_device()

    al.armed = True  # Simulate being armed
    await det.stage()

    assert al.disarm_count == 1
    assert al.armed is False


async def test_describe_before_prepare_raises(tmp_path):
    """Test that describe() fails before prepare()."""
    det = DetectorLogic(data=[StreamableOnlyDataLogic(tmp_path)]).with_device()
    await det.stage()

    with pytest.raises(TransitionError, match="not allowed from STAGED"):
        await det.describe()


async def test_describe_collect_before_prepare_raises(tmp_path):
    """Test that describe_collect() fails before prepare()."""
    det = DetectorLogic(data=[StreamableOnlyDataLogic(tmp_path)]).with_device()
    await det.stage()

    with pytest.raises(TransitionError, match="not allowed from STAGED"):
        await det.describe_collect()


async def test_trigger_after_multi_event_prepare_raises():
    """Test that trigger() after prepare with multiple events fails."""
    det = DetectorLogic(trigger=JustInternalTriggerLogic()).with_device()
    await det.stage()

    await det.prepare(TriggerInfo(number_of_events=5))

    with pytest.raises(ValueError, match="trigger\\(\\) is not supported for multiple"):
        await det.trigger()


async def test_preserve_detector_state_requires_default_trigger_info(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test PRESERVE_DETECTOR_STATE=YES errors without default_trigger_info."""
    monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "YES")
    det = DetectorLogic(trigger=AllTriggerTypesLogic()).with_device("mydet")
    # AllTriggerTypesLogic intentionally does not implement default_trigger_info
    await det.stage()

    with pytest.raises(
        RuntimeError,
        match="OPHYD_ASYNC_PRESERVE_DETECTOR_STATE=YES is set but the "
        "trigger logic has no default_trigger_info\\(\\)",
    ):
        await det.trigger()


async def test_preserve_detector_state_no_trigger_logic_falls_back(
    monkeypatch: pytest.MonkeyPatch,
):
    """When OPHYD_ASYNC_PRESERVE_DETECTOR_STATE=YES but there is no trigger logic,
    trigger() silently falls back to a bare TriggerInfo() rather than raising.
    A detector with no trigger logic has no hardware state to preserve."""
    monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "YES")
    det = DetectorLogic().with_device("nodet")
    await det.stage()
    # Should not raise: no trigger logic means nothing to preserve
    await det.trigger()
    # ...and it was prepared, implicitly
    assert await det.describe() == {}


async def test_preserve_detector_state_multi_collection_watcher_and_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    """When OPHYD_ASYNC_PRESERVE_DETECTOR_STATE=YES and default_trigger_info returns
    collections_per_event > 1, trigger() should wait for that many raw collections
    and collect_asset_docs() should emit one datum for the Bluesky event."""

    class MultiCollectionTriggerLogic(DetectorTriggerLogic):
        async def prepare_internal(self, num: int, livetime: float, deadtime: float):
            pass

        async def default_trigger_info(self) -> TriggerInfo:
            return TriggerInfo(collections_per_event=5)

    monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "YES")
    dl = StreamableOnlyDataLogic(tmp_path)
    det = DetectorLogic(trigger=MultiCollectionTriggerLogic(), data=[dl]).with_device(
        "multidet"
    )
    await det.stage()

    # Collect watcher updates via watch() callback
    updates: list[dict] = []
    status = det.trigger()
    status.watch(lambda **kwargs: updates.append(kwargs))
    await wait_for_pending_wakeups(raise_if_exceeded=False)
    assert (await det.describe())["multidet"]["shape"] == [5, 10, 15]

    for collections_written in range(1, 5):
        await dl.collections_written.set(collections_written)
        await wait_for_pending_wakeups(raise_if_exceeded=False)
        assert not status.done

    await dl.collections_written.set(5)
    await status
    docs = [doc async for doc in det.collect_asset_docs()]
    assert [name for name, _ in docs] == ["stream_resource", "stream_datum"]
    assert docs[1][1]["indices"] == {"start": 0, "stop": 1}
    assert [update["current"] for update in updates] == [0, 1, 2, 3, 4, 5]
    assert all(update["target"] == 5 for update in updates)


async def test_collect_asset_docs_uses_trigger_observed_event(
    tmp_path,
):
    """When ``trigger()`` observes that ``dl.collections_written`` hit its target,
    ``collect_asset_docs`` should not check *again* for the same condition.

    It could be the case that ``dl.collections_written`` changes between ``trigger()``
    and ``collect_asset_docs``."""
    dl = StreamableOnlyDataLogic(tmp_path)
    dl.collections_written, set_collections_written = soft_signal_r_and_setter(
        int, getter=lambda: 0
    )
    det = DetectorLogic(trigger=JustInternalTriggerLogic(), data=[dl]).with_device(
        "det"
    )
    await det.stage()
    await det.prepare(TriggerInfo(collections_per_event=5, exposure_timeout=0.1))

    status = det.trigger()
    await wait_for_pending_wakeups(raise_if_exceeded=False)
    set_collections_written(5)
    await status

    docs = [doc async for doc in det.collect_asset_docs()]
    assert [name for name, _ in docs] == ["stream_resource", "stream_datum"]
    assert docs[1][1]["indices"] == {"start": 0, "stop": 1}


async def test_trigger_times_out_on_collection_timeout(tmp_path):
    det = DetectorLogic(
        trigger=JustInternalTriggerLogic(), data=[StreamableOnlyDataLogic(tmp_path)]
    ).with_device("det")
    await det.stage()
    await det.prepare(TriggerInfo(exposure_timeout=100, collection_timeout=0.1))
    with pytest.raises(TimeoutError):
        await det.trigger()


async def test_one_kickoff_per_prepare(tmp_path):
    """A prepare serves one kickoff/complete cycle; the next needs a new prepare."""
    dl = StreamableOnlyDataLogic(tmp_path)
    det = DetectorLogic(trigger=JustInternalTriggerLogic(), data=[dl]).with_device(
        "foo"
    )
    await det.stage()

    await det.prepare(TriggerInfo(number_of_events=5))
    await det.kickoff()

    # A second kickoff without an intervening complete is rejected
    with pytest.raises(RuntimeError, match="cannot go to FLYING from FLYING"):
        await det.kickoff()

    status = det.complete()
    await dl.collections_written.set(5)
    await status

    # ...and so is one after complete, until prepare() runs again
    with pytest.raises(RuntimeError, match="cannot go to FLYING from FLOWN"):
        await det.kickoff()


async def test_restage_resets_state():
    """Test that unstage() then stage() resets detector state."""
    det = DetectorLogic(trigger=JustInternalTriggerLogic()).with_device()
    await det.stage()

    await det.prepare(TriggerInfo(number_of_events=5))

    # Restaging should reset everything
    await det.unstage()
    await det.stage()

    with pytest.raises(TransitionError, match="describe not allowed from STAGED"):
        await det.describe()
    with pytest.raises(TransitionError, match="cannot go to FLYING from STAGED"):
        await det.kickoff()


async def test_hints_from_single_data_logic(tmp_path):
    """Test that hints come from data logic."""
    det = DetectorLogic(data=[StreamableOnlyDataLogic(tmp_path)]).with_device("bar")
    await det.stage()

    await det.prepare(TriggerInfo())

    assert det.hints == {"fields": ["bar"]}


async def test_hints_from_multiple_data_logics(tmp_path):
    """Test that hints are aggregated from multiple data logics."""
    dl1 = StreamableOnlyDataLogic(tmp_path)
    dl2 = StreamableOnlyDataLogic(tmp_path, datakey_suffix="-extra")
    det = DetectorLogic(data=[dl1, dl2]).with_device("bar")
    await det.stage()

    await det.prepare(TriggerInfo())

    # Should include hints from both logics
    hints = det.hints
    assert "fields" in hints
    assert hints["fields"] == ["bar", "bar-extra"]


@pytest.mark.parametrize(
    "signal_type,initial_value,expected_dtype,expected_dtype_numpy,expected_shape",
    [
        (float, 1.5, "number", "<f8", []),
        (str, "test", "string", "|S40", []),
        (Array1D[np.int32], np.array([1, 2, 3], dtype=np.int32), "array", "<i4", [3]),
    ],
)
async def test_config_signals_in_describe_configuration(
    signal_type, initial_value, expected_dtype, expected_dtype_numpy, expected_shape
):
    """Test that added config signals appear in describe_configuration."""
    det = DetectorLogic().with_device()
    await det.stage()
    signal = soft_signal_rw(
        signal_type, initial_value=initial_value, name="test-config"
    )
    det.set_readable_format(signal, Format.CONFIG_SIGNAL)

    # Check describe_configuration
    config_desc = await det.describe_configuration()
    assert config_desc == {
        "test-config": {
            "dtype": expected_dtype,
            "dtype_numpy": expected_dtype_numpy,
            "shape": expected_shape,
            "source": "soft://test-config",
        },
    }

    # Check read_configuration
    await assert_configuration(
        det,
        {
            "test-config": {"value": initial_value},
        },
    )


async def test_streamable_supports_both_step_and_fly(tmp_path):
    """Test that streamable data logic supports both step and fly scanning."""
    tl = JustInternalTriggerLogic()
    dl = StreamableOnlyDataLogic(tmp_path)
    det = DetectorLogic(trigger=tl, data=[dl]).with_device("foo")
    await det.stage()

    # Step scan should work
    status = det.trigger()
    # Yield so detector can get collections written, then set it so we complete
    await wait_for_pending_wakeups(raise_if_exceeded=False)
    await dl.collections_written.set(1)
    await wait_for_pending_wakeups(raise_if_exceeded=False)
    assert status.done
    assert status.success
    docs = [doc async for doc in det.collect_asset_docs()]
    sr = docs[:1]
    assert docs == [
        (
            "stream_resource",
            {
                "data_key": "foo",
                "mimetype": "application/x-hdf5",
                "parameters": {
                    "chunk_shape": (1, 10, 15),
                    "dataset": "/data",
                },
                "uid": ANY,
                "uri": f"file://localhost/{tmp_path.as_posix().lstrip('/')}/test.h5",
            },
        ),
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 0, "stop": 1},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": ANY,
                "uid": ANY,
            },
        ),
    ]
    # Fly scan should also work
    await det.prepare(TriggerInfo(number_of_events=5))
    await det.kickoff()
    status = det.complete()
    # Check that setting collections written will only give the first 4, in a
    # new resource for the new prepare
    await dl.collections_written.set(5)
    docs = [doc async for doc in det.collect_asset_docs()]
    new_sr = docs[:1]
    assert new_sr == [(name, {**doc, "uid": ANY}) for name, doc in sr]
    assert new_sr[0][1]["uid"] != sr[0][1]["uid"]
    assert docs[1:] == [
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 1, "stop": 5},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": new_sr[0][1]["uid"],
                "uid": ANY,
            },
        ),
    ]
    assert not status.done
    # Then one more should complete
    await dl.collections_written.set(6)
    docs = [doc async for doc in det.collect_asset_docs()]
    assert docs == [
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 5, "stop": 6},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": ANY,
                "uid": ANY,
            },
        ),
    ]
    assert status.done


async def test_read_returns_correct_values():
    """Test that read() returns the values of registered signals."""
    det = DetectorLogic().with_device("det")
    await det.stage()
    det.counts, _ = soft_signal_r_and_setter(int, 42, name="det-counts")
    det.set_readable_format(det.counts, Format.HINTED_SIGNAL)

    await det.trigger()
    await assert_reading(det, {"det-counts": {"value": 42}})


async def test_detector_with_no_logics():
    """Test that detector works with no logics for basic internal triggering."""
    det = DetectorLogic().with_device()
    await det.stage()

    # Should support only INTERNAL triggering
    triggers, deadtime = await det.get_trigger_deadtime()
    assert triggers == {DetectorTrigger.INTERNAL}
    assert deadtime is None

    # Can prepare with INTERNAL trigger but not external
    await det.prepare(TriggerInfo())
    await det.trigger()
    with pytest.raises(ValueError, match="Trigger type.*EXTERNAL_EDGE not supported"):
        await det.prepare(TriggerInfo(trigger=DetectorTrigger.EXTERNAL_EDGE))


@pytest.mark.parametrize(
    "info",
    [
        TriggerInfo(livetime=0.5),
        TriggerInfo(deadtime=0.1),
        TriggerInfo(exposures_per_collection=2),
    ],
)
async def test_detector_without_trigger_logic_cannot_set_timing(info: TriggerInfo):
    det = DetectorLogic().with_device("foo")
    await det.stage()
    with pytest.raises(ValueError, match="No trigger logic, so cannot set livetime"):
        await det.prepare(info)


async def test_errors_from_logic_calls_keep_their_message():
    det = DetectorLogic().with_device("foo")
    await det.stage()
    with pytest.raises(ValueError) as exc_info:
        await det.prepare(TriggerInfo(livetime=0.5))
    assert exc_info.value.args == (
        "No trigger logic, so cannot set livetime or deadtime "
        "or exposures_per_collection",
    )


async def test_detector_logic_takes_keywords_only():
    """Each logic's role is named, so none can be given positionally."""
    with pytest.raises(TypeError, match="positional"):
        DetectorLogic(JustInternalTriggerLogic())  # type: ignore[misc]


async def test_replaced_detector_logic_does_not_share_prepared_data(tmp_path):
    """A copy made with dataclasses.replace has its own prepared state."""
    logic = DetectorLogic(
        trigger=JustInternalTriggerLogic(), data=[StreamableOnlyDataLogic(tmp_path)]
    )
    det = logic.with_device("det")
    copy = dataclasses.replace(logic).with_device("copy")
    await det.stage()
    await copy.stage()

    await det.prepare(TriggerInfo())

    assert list(await det.describe()) == ["det"]
    with pytest.raises(TransitionError, match="describe not allowed from STAGED"):
        await copy.describe()


@pytest.mark.parametrize("initial_shutter_closed", [True, False])
async def test_ensure_ready_vs_ensure_stopped_hooks(initial_shutter_closed: bool):
    """ensure_ready and ensure_stopped are separate hooks.

    A detector that needs different behaviour at stage time (ensure_ready) versus
    scan-end (ensure_stopped) can override both independently.  Here a shutter
    must stay open between kickoff/complete cycles, but must close when the scan
    ends (unstage).  stage() must not close the shutter even if it was already
    closed before the scan began.
    """

    class ShutterAcquireLogic(DetectorAcquireLogic):
        def __init__(self, shutter_closed: bool):
            self.shutter_closed = shutter_closed

        async def ensure_ready(self):
            pass  # don't touch the shutter at stage time

        async def acquire(self):
            self.shutter_closed = False  # open shutter when acquiring
            yield

        async def ensure_stopped(self):
            self.shutter_closed = True  # close shutter at end of scan

    al = ShutterAcquireLogic(initial_shutter_closed)
    det = DetectorLogic(acquire=al).with_device()

    await det.stage()
    assert (
        al.shutter_closed is initial_shutter_closed
    )  # stage() must not touch the shutter

    await det.trigger()
    assert al.shutter_closed is False  # acquire() opens the shutter

    await det.unstage()
    assert al.shutter_closed is True  # unstage() must close the shutter


async def test_bounded_step_scan_reads_derived_from_page():
    """A bounded buffer describes as an array and read() derives one reading."""
    dl = BoundedOnlyDataLogic()
    det = DetectorLogic(trigger=JustInternalTriggerLogic(), data=[dl]).with_device(
        "foo"
    )
    await det.stage()

    ti = TriggerInfo(
        number_of_events=1, collections_per_event=5, livetime=0.1, deadtime=0.0
    )
    await det.prepare(ti)
    desc = await det.describe()
    assert desc["foo"]["shape"] == [5]
    # prepare sized the buffer for 5 collections at the 0.1s period
    assert dl.prepare_calls == [(5, pytest.approx(0.1))]

    status = det.trigger()
    await wait_for_pending_wakeups(raise_if_exceeded=False)
    await dl.collections_written.set(5)
    await status
    assert status.success
    # prepare() armed the buffer for this point, so trigger() does not re-arm it
    assert len(dl.providers) == 1
    assert dl.stop_count == 1
    reading = await det.read()
    assert reading["foo"]["value"] == [0, 0, 0, 0, 0]

    # The next point stops, re-makes and re-starts the logic, as the buffer
    # holds the last point's data, and counts from the re-armed buffer's 0
    status = det.trigger()
    await wait_for_pending_wakeups(raise_if_exceeded=False)
    assert len(dl.providers) == 2
    assert dl.stop_count == 2
    assert not status.done
    await dl.collections_written.set(5)
    await status
    assert status.success
    assert dl.prepare_calls == [(5, pytest.approx(0.1))] * 2


async def test_bounded_fly_scan_accumulates_across_kickoffs():
    """A bounded buffer is armed once at prepare and not re-armed per kickoff."""
    dl = BoundedOnlyDataLogic()
    det = DetectorLogic(trigger=JustInternalTriggerLogic(), data=[dl]).with_device(
        "foo"
    )
    await det.stage()

    await det.prepare(
        TriggerInfo(number_of_events=10, collections_per_event=1, livetime=0.1)
    )
    assert dl.prepare_calls == [(10, pytest.approx(0.1))]

    await det.kickoff()
    status = det.complete()
    # The buffer fills as the scan runs, and pages come out as it goes
    await dl.collections_written.set(5)
    pages = [page async for page in det.collect_pages()]  # type: ignore[attr-defined]
    assert pages[0]["data"]["foo"] == [[0]] * 5
    assert not status.done

    await dl.collections_written.set(10)
    pages = [page async for page in det.collect_pages()]  # type: ignore[attr-defined]
    assert pages[0]["data"]["foo"] == [[0]] * 5
    await status
    # kickoff() never re-arms: the buffer was only ever armed once
    assert dl.prepare_calls == [(10, pytest.approx(0.1))]


@pytest.mark.parametrize(
    "requested_livetime,default,expected_period",
    [
        # livetime unset -> resolved from the trigger logic's current state
        (0.0, TriggerInfo(livetime=0.1, deadtime=0.02), 0.12),
        # livetime given -> used as-is, no readback substitution
        (0.05, TriggerInfo(livetime=0.1, deadtime=0.02), 0.05),
        # livetime unset but trigger logic has nothing set -> stays 0
        (0.0, TriggerInfo(), 0.0),
    ],
)
async def test_prepare_resolves_zero_livetime_for_bounded_period(
    requested_livetime, default, expected_period, monkeypatch: pytest.MonkeyPatch
):
    """A livetime of 0 is filled in from the trigger logic before sizing a buffer.

    Only when hardware state is preserved: test_u12 in test_detector_use_cases
    covers the variable being unset.
    """
    monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "YES")

    class PeriodTriggerLogic(DetectorTriggerLogic):
        async def prepare_internal(self, num: int, livetime: float, deadtime: float):
            pass

        async def default_trigger_info(self) -> TriggerInfo:
            return default

    dl = BoundedOnlyDataLogic()
    det = DetectorLogic(trigger=PeriodTriggerLogic(), data=[dl]).with_device("foo")
    await det.stage()

    await det.prepare(
        TriggerInfo(
            number_of_events=1, collections_per_event=3, livetime=requested_livetime
        )
    )
    assert dl.prepare_calls == [(3, pytest.approx(expected_period))]


async def test_data_logic_with_no_start_provider_raises():
    """Test error when a DataLogic does not implement start_provider."""

    class EmptyDataLogic(UnboundedDataLogic):
        async def start_provider(self, *args):
            return await super().start_provider(*args)  # type: ignore[safe-super]

    det = DetectorLogic(data=[EmptyDataLogic()]).with_device()
    await det.stage()

    with pytest.raises(NotImplementedError):
        await det.prepare(TriggerInfo())


async def test_unstage_disarms_detector():
    """Test that unstage() calls disarm on the detector."""
    al = MockAcquireLogic()
    det = DetectorLogic(acquire=al).with_device()
    await det.stage()

    al.armed = True
    await det.unstage()

    # Once on stage(), once on unstage()
    assert al.disarm_count == 2
    assert al.armed is False


@pytest.mark.parametrize(
    "second_collections_per_event,stops", [(2, 0), (3, 1)], ids=["kept", "remade"]
)
async def test_prepare_stops_data_logic_when_recreating_providers(
    tmp_path, second_collections_per_event: int, stops: int
):
    """A streamable provider is kept while it would be made the same, else remade."""
    dl = StreamableOnlyDataLogic(tmp_path)
    det = DetectorLogic(trigger=JustInternalTriggerLogic(), data=[dl]).with_device(
        "det"
    )
    await det.stage()
    staged_stops = dl.stop_count

    await det.prepare(TriggerInfo(collections_per_event=2))
    assert dl.stop_count == staged_stops  # No stop on first prepare
    status = det.trigger()
    await wait_for_pending_wakeups(raise_if_exceeded=False)
    await dl.collections_written.set(2)
    await status

    await det.prepare(TriggerInfo(collections_per_event=second_collections_per_event))
    assert dl.stop_count == staged_stops + stops


async def test_different_collections_written_raises(tmp_path):
    """Test that different collections_written values from providers raises error."""
    dl1 = StreamableOnlyDataLogic(tmp_path)
    dl2 = StreamableOnlyDataLogic(tmp_path, datakey_suffix="-2")
    det = DetectorLogic(
        trigger=JustInternalTriggerLogic(), data=[dl1, dl2]
    ).with_device("det")
    await det.stage()
    # Neither start_provider() resets its count
    await dl1.collections_written.set(3)
    await dl2.collections_written.set(5)

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Differing numbers of collections written: "
            "StreamableOnlyDataLogic[0] has 3, StreamableOnlyDataLogic[1] has 5"
        ),
    ):
        await det.prepare(TriggerInfo(number_of_events=5))


async def test_data_logic_and_registered_signal(tmp_path):
    """A registered signal and a data logic both describe(), only one collects."""
    det = DetectorLogic(
        trigger=JustInternalTriggerLogic(), data=[StreamableOnlyDataLogic(tmp_path)]
    ).with_device("det")
    await det.stage()
    det.counts, _ = soft_signal_r_and_setter(int, 42, name="det-counts")
    det.set_readable_format(det.counts, Format.HINTED_SIGNAL)

    await det.prepare(TriggerInfo(number_of_events=3))

    # Should have data from the data logic and the registered signal
    description = await det.describe()
    assert description == {
        "det": {
            "dtype": "array",
            "dtype_numpy": "|u1",
            "external": "STREAM:",
            "shape": [1, 10, 15],
            "source": f"file://localhost/{tmp_path.as_posix().lstrip('/')}/test.h5",
        },
        "det-counts": {
            "dtype": "integer",
            "dtype_numpy": "<i8",
            "shape": [],
            "source": "soft://det-counts",
        },
    }
    # But collect only has streamable
    collect_description = await det.describe_collect()
    assert collect_description == {
        "det": {
            "dtype": "array",
            "dtype_numpy": "|u1",
            "external": "STREAM:",
            "shape": [1, 10, 15],
            "source": f"file://localhost/{tmp_path.as_posix().lstrip('/')}/test.h5",
        },
    }
    # Should be able to do fly scanning (has streamable logic)
    await det.kickoff()
    await det.unstage()


async def test_collect_asset_docs_with_explicit_index(tmp_path):
    """Test collect_asset_docs() with explicitly provided index."""
    dl = StreamableOnlyDataLogic(tmp_path)
    det = DetectorLogic(trigger=JustInternalTriggerLogic(), data=[dl]).with_device(
        "det"
    )
    await det.stage()

    await det.prepare(TriggerInfo(number_of_events=5, collections_per_event=2))
    await det.kickoff()

    # Collect with explicit index (not relying on get_index)
    docs = [doc async for doc in det.collect_asset_docs(index=3)]
    await det.unstage()

    # Should emit docs for 3 events
    assert docs == [
        (
            "stream_resource",
            {
                "data_key": "det",
                "mimetype": "application/x-hdf5",
                "parameters": {
                    "chunk_shape": (1, 10, 15),
                    "dataset": "/data",
                },
                "uid": ANY,
                "uri": f"file://localhost/{tmp_path.as_posix().lstrip('/')}/test.h5",
            },
        ),
        (
            "stream_datum",
            {
                "descriptor": "",
                "indices": {"start": 0, "stop": 3},
                "seq_nums": {"start": 0, "stop": 0},
                "stream_resource": ANY,
                "uid": ANY,
            },
        ),
    ]


async def test_child_readable_config_signals_in_describe_configuration():
    """Child StandardReadable CONFIG_SIGNALs appear in describe_configuration."""
    child = StandardReadable(name="child")
    config_sig = soft_signal_rw(float, initial_value=1.5, name="child-exposure")
    child.set_readable_format(config_sig, Format.CONFIG_SIGNAL)

    det = DetectorLogic().with_device("det")
    await det.stage()
    det.set_readable_format(child, Format.CHILD)
    await det.prepare(TriggerInfo())

    config = await det.describe_configuration()
    assert "child-exposure" in config
    reading = await det.read_configuration()
    assert "child-exposure" in reading
    assert reading["child-exposure"]["value"] == 1.5


async def test_child_readable_read_signals_in_read(tmp_path):
    """Child StandardReadable HINTED_SIGNALs appear in read/describe."""
    child = StandardReadable(name="child")
    read_sig = soft_signal_rw(int, initial_value=99, name="child-counts")
    child.set_readable_format(read_sig, Format.HINTED_SIGNAL)

    dl = StreamableOnlyDataLogic(tmp_path)
    det = DetectorLogic(data=[dl]).with_device("det")
    await det.stage()
    det.set_readable_format(child, Format.CHILD)
    await det.prepare(TriggerInfo())
    status = det.trigger()
    await wait_for_pending_wakeups(raise_if_exceeded=False)
    await dl.collections_written.set(1)
    await status

    # The data logic describes its stream, the child describes its signal
    desc = await det.describe()
    assert "child-counts" in desc
    assert "det" in desc

    reading = await det.read()
    assert "child-counts" in reading
    assert reading["child-counts"]["value"] == 99


async def test_child_readable_hints_merged(tmp_path):
    """Child StandardReadable hints are merged with data logic hints."""
    child = StandardReadable(name="child")
    hinted_sig = soft_signal_rw(float, name="child-intensity")
    child.set_readable_format(hinted_sig, Format.HINTED_SIGNAL)

    det = DetectorLogic(data=[StreamableOnlyDataLogic(tmp_path)]).with_device("det")
    await det.stage()
    det.set_readable_format(child, Format.CHILD)

    assert "fields" in det.hints
    assert "child-intensity" in det.hints["fields"]
    assert "det" in det.hints["fields"]


async def test_trigger_logic_not_implemented_errors():
    """Test NotImplementedError for unimplemented DetectorTriggerLogic methods."""
    logic = DetectorTriggerLogic()

    with pytest.raises(NotImplementedError):
        logic.get_deadtime(SignalDict())

    with pytest.raises(NotImplementedError):
        await logic.prepare_internal(1, 0.1, 0.01)

    with pytest.raises(NotImplementedError):
        await logic.prepare_edge(1, 0.1)

    with pytest.raises(NotImplementedError):
        await logic.prepare_level(1)

    with pytest.raises(NotImplementedError):
        await logic.prepare_exposures_per_collection(2)


async def test_data_logic_not_implemented_errors():
    """Test NotImplementedError for unimplemented DetectorDataLogic methods."""

    class Bounded(BoundedDataLogic[DataProvider]):
        async def start_provider(self, *args):
            return await super().start_provider(*args)  # type: ignore[safe-super]

    class Unbounded(UnboundedDataLogic[DataProvider]):
        async def start_provider(self, *args):
            return await super().start_provider(*args)  # type: ignore[safe-super]

    with pytest.raises(NotImplementedError):
        await Bounded().start_provider("test", [], 1, 1, 0, 0)
    with pytest.raises(NotImplementedError):
        await Unbounded().start_provider("test", 1, 0, 0)
    logic = Bounded()

    # These have default implementations, so should not raise
    assert logic.get_hinted_fields("test") == []
    await logic.ensure_ready()
    await logic.ensure_stopped()


async def test_detector_readable_format_changes_at_runtime():
    # A StandardDetector is a StandardReadable, so a plugin signal can be moved
    # between configuration and hinted reads without redefining the Device.
    det = DetectorLogic().with_device("det")
    await det.stage()
    det.temperature, _ = soft_signal_r_and_setter(float, 20.0, name="temperature")

    name = det.temperature.name
    det.set_readable_format(det.temperature, Format.CONFIG_SIGNAL)
    assert set(await det.read_configuration()) == {name}
    # Empty hints are {} rather than {"fields": []}, as for any StandardReadable
    assert det.hints == {}

    det.set_readable_format(det.temperature, Format.HINTED_UNCACHED_SIGNAL)
    assert set(await det.read_configuration()) == set()
    assert det.hints == {"fields": [name]}

    det.set_readable_format(det.temperature, None)
    assert set(await det.read_configuration()) == set()
    assert det.hints == {}


async def test_detector_add_config_signals_is_deprecated():
    det = DetectorLogic().with_device("det")
    await det.stage()
    signal, _ = soft_signal_r_and_setter(float, 1.0, name="sig")
    with pytest.deprecated_call():
        det.add_config_signals(signal)
    assert set(await det.read_configuration()) == {signal.name}


async def test_streamable_logic_looks_like_writes_stream_assets(tmp_path):
    """An unbounded logic exposes collect_asset_docs, and only that.

    The bluesky bundler picks up stream assets via a structural isinstance
    against WritesStreamAssets (a hasattr check for collect_asset_docs). A
    detector must look like exactly one of WritesStreamAssets /
    EventPageCollectable so the bundler routes it down a single path.
    """
    det = DetectorLogic(data=[StreamableOnlyDataLogic(tmp_path)]).with_device()
    await det.stage()

    # Which verb applies follows what the logics will produce, so it is not
    # known until they have been asked
    assert not isinstance(det, WritesStreamAssets)
    await det.prepare(TriggerInfo())

    assert isinstance(det, WritesStreamAssets)
    assert not isinstance(det, EventPageCollectable)
    assert hasattr(det, "collect_asset_docs")
    assert not hasattr(det, "collect_pages")


async def test_bounded_logic_looks_like_event_page_collectable():
    """A bounded logic exposes collect_pages, and only that."""
    det = DetectorLogic(data=[BoundedOnlyDataLogic()]).with_device()
    await det.stage()
    await det.prepare(TriggerInfo())

    assert isinstance(det, EventPageCollectable)
    assert not isinstance(det, WritesStreamAssets)
    assert hasattr(det, "collect_pages")
    assert not hasattr(det, "collect_asset_docs")


async def test_no_data_logic_looks_like_neither():
    """A detector with no data logic emits neither stream assets nor event pages."""
    det = DetectorLogic(trigger=JustInternalTriggerLogic()).with_device()
    await det.stage()

    assert not isinstance(det, WritesStreamAssets)
    assert not isinstance(det, EventPageCollectable)
    assert not hasattr(det, "collect_asset_docs")
    assert not hasattr(det, "collect_pages")


async def test_missing_attribute_raises_standard_attribute_error():
    """__getattr__ falls through to a normal AttributeError for other names."""
    det = DetectorLogic().with_device()
    await det.stage()
    with pytest.raises(
        AttributeError,
        match=r"object has no attribute 'does_not_exist'",
    ):
        det.does_not_exist  # noqa: B018


async def test_unsuppressed_bounded_keys_raise(tmp_path):
    """A finite buffer the stream assets do not cover cannot be combined with them."""
    det = DetectorLogic(
        data=[
            BoundedOnlyDataLogic(datakey_suffix="-stats"),
            StreamableOnlyDataLogic(tmp_path),
        ]
    ).with_device("det")
    await det.stage()

    # Only knowable once the logics have said what they would make
    with pytest.raises(
        TypeError,
        match=(
            r"would produce \['det-stats'\] as event pages and the rest of its "
            r"data as stream assets"
        ),
    ):
        await det.prepare(TriggerInfo())


async def test_suppressed_bounded_logic_sits_the_scan_out(tmp_path):
    """A finite buffer whose keys are all written durably gives way to the file."""
    bounded = BoundedOnlyDataLogic()
    # The same quantity as the buffer: one integer per collection
    streamable = StreamableOnlyDataLogic(tmp_path, shape=(), dtype_numpy="<i8")
    det = DetectorLogic(data=[bounded, streamable]).with_device("det")
    await det.stage()

    await det.prepare(TriggerInfo())

    # Both would produce "det", so the durable copy wins and nothing was armed
    assert bounded.prepare_calls == []
    assert isinstance(det, WritesStreamAssets)
    assert not isinstance(det, EventPageCollectable)
    # ...and it stops hinting at data nobody will produce
    assert det.hints == {"fields": ["det"]}


class SetSuppressLogic(DetectorSuppressLogic):
    def __init__(self, hidden: set[str]) -> None:
        self.hidden = hidden

    async def hidden_datakeys(self) -> set[str]:
        return set(self.hidden)


async def test_suppress_logic_hides_readables_until_unstage():
    hidden = soft_signal_rw(int, 1, name="det-hidden")
    also_hidden = soft_signal_rw(int, 2, name="det-also-hidden")
    kept = soft_signal_rw(int, 3, name="det-kept")
    suppress = SetSuppressLogic({"det-hidden", "det-also-hidden"})
    det = DetectorLogic(suppress=suppress).with_device("det")
    for signal in (hidden, also_hidden, kept):
        det.set_readable_format(signal, Format.HINTED_SIGNAL)
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()

    description = await det.describe()

    assert set(description) == {"det-kept"}

    # The same result at a later prepare is no change
    await det.prepare(TriggerInfo())
    await det.trigger()

    # A different result is, until the next stage takes it
    suppress.hidden = {"det-hidden"}
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "det: hidden datakeys changed since the first prepare() after "
            "stage(): ['det-also-hidden']. unstage() and stage() to use the new values."
        ),
    ):
        await det.prepare(TriggerInfo())
    await det.unstage()
    await det.stage()
    await det.prepare(TriggerInfo())
    await det.trigger()
    assert set(await det.describe()) == {"det-also-hidden", "det-kept"}


class _HangingLogic:
    """A data logic whose start_provider never returns, recording if it runs."""

    running = False

    async def start_provider(self, *args):
        self.running = True
        try:
            await asyncio.Event().wait()
        finally:
            self.running = False


class _FailingLogic:
    async def start_provider(self, *args):
        raise ValueError("path does not exist")


class HangingUnboundedLogic(_HangingLogic, UnboundedDataLogic): ...


class FailingUnboundedLogic(_FailingLogic, UnboundedDataLogic): ...


class HangingBoundedLogic(_HangingLogic, BoundedDataLogic): ...


class FailingBoundedLogic(_FailingLogic, BoundedDataLogic): ...


@pytest.mark.parametrize(
    "hanging,failing",
    [
        (HangingUnboundedLogic, FailingUnboundedLogic),
        (HangingBoundedLogic, FailingBoundedLogic),
    ],
    ids=["unbounded", "bounded"],
)
async def test_failed_start_provider_stops_the_others_starting(hanging, failing):
    """Nothing a failed prepare() started is still running after it raises."""
    hanging_logic = hanging()
    det = DetectorLogic(data=[hanging_logic, failing()]).with_device("det")
    await det.stage()

    with pytest.raises(ValueError, match="path does not exist"):
        await det.prepare(TriggerInfo())

    assert not hanging_logic.running
    await det.unstage()


async def test_hidden_datakeys_are_asked_before_the_trigger_logic_is_prepared():
    calls: list[str] = []

    class RecordingTriggerLogic(JustInternalTriggerLogic):
        async def prepare_internal(self, num: int, livetime: float, deadtime: float):
            calls.append("prepare_internal")

    class RecordingSuppressLogic(DetectorSuppressLogic):
        async def hidden_datakeys(self) -> set[str]:
            calls.append("hidden_datakeys")
            return set()

    det = DetectorLogic(
        trigger=RecordingTriggerLogic(), suppress=RecordingSuppressLogic()
    ).with_device("det")
    await det.stage()

    await det.prepare(TriggerInfo())

    assert calls == ["hidden_datakeys", "prepare_internal"]
