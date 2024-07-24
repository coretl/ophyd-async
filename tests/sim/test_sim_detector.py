from collections import defaultdict

import bluesky.plans as bp
import h5py
import numpy as np
import pytest
from bluesky import RunEngine

from ophyd_async.core import DeviceCollector, assert_emitted
from ophyd_async.epics import motor
from ophyd_async.sim.demo import PatternDetector


@pytest.fixture
async def sim_motor():
    async with DeviceCollector(mock=True):
        sim_motor = motor.Motor("test")
    return sim_motor


async def test_sim_pattern_detector_initialization(
    sim_pattern_detector: PatternDetector,
):
    assert (
        sim_pattern_detector.pattern_generator
    ), "PatternGenerator was not initialized correctly."


async def test_detector_creates_controller_and_writer(
    sim_pattern_detector: PatternDetector,
):
    assert sim_pattern_detector.writer
    assert sim_pattern_detector.controller


async def test_writes_pattern_to_file(
    sim_pattern_detector: PatternDetector,
    RE: RunEngine,
):
    # assert that the file contains data in expected dimensions
    docs = defaultdict(list)

    def capture_emitted(name, doc):
        docs[name].append(doc)

    RE(bp.count([sim_pattern_detector]), capture_emitted)
    assert_emitted(
        docs, start=1, descriptor=1, stream_resource=2, stream_datum=2, event=1, stop=1
    )
    path = docs["stream_resource"][0]["uri"].split("://localhost")[-1]
    h5file = h5py.File(path)
    assert list(h5file["/entry"]) == ["data", "sum"]
    assert list(h5file["/entry/sum"]) == [44540.0]
    assert np.sum(h5file["/entry/data/data"]) == 44540.0
