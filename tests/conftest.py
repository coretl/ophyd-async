import asyncio
import os
import pprint
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from bluesky.run_engine import RunEngine, TransitionError
from pytest import FixtureRequest

from ophyd_async.core import (
    DetectorTrigger,
    FilenameProvider,
    StaticFilenameProvider,
    StaticPathProvider,
    TriggerInfo,
)

PANDA_RECORD = str(Path(__file__).parent / "fastcs" / "panda" / "db" / "panda.db")
INCOMPLETE_BLOCK_RECORD = str(
    Path(__file__).parent / "fastcs" / "panda" / "db" / "incomplete_block_panda.db"
)
INCOMPLETE_RECORD = str(
    Path(__file__).parent / "fastcs" / "panda" / "db" / "incomplete_panda.db"
)
EXTRA_BLOCKS_RECORD = str(
    Path(__file__).parent / "fastcs" / "panda" / "db" / "extra_blocks_panda.db"
)

# Prevent pytest from catching exceptions when debugging in vscode so that break on
# exception works correctly (see: https://github.com/pytest-dev/pytest/issues/7409)
if os.getenv("PYTEST_RAISE", "0") == "1":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


# Autouse fixture that will set all EPICS networking env vars to use lo interface
# to avoid false failures caused by things like firewalls blocking EPICS traffic.
@pytest.fixture(scope="session", autouse=True)
def configure_epics_environment():
    os.environ["EPICS_CAS_INTF_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_CAS_BEACON_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_CAS_AUTO_ADDR_LIST"] = "NO"
    os.environ["EPICS_CA_AUTO_BEACON_ADDR_LIST"] = "NO"

    os.environ["EPICS_PVAS_INTF_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_PVAS_BEACON_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_PVA_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_PVAS_AUTO_BEACON_ADDR_LIST"] = "NO"
    os.environ["EPICS_PVA_AUTO_ADDR_LIST"] = "NO"


_ALLOWED_CORO_TASKS = {"async_finalizer", "async_setup", "async_teardown"}


@pytest.fixture(autouse=True, scope="function")
def assert_no_pending_tasks(request: FixtureRequest):
    """
    if "RE" in request.fixturenames:
        # The RE fixture will do the same, as well as some additional loop cleanup
        return
    """

    # There should be no tasks pending after a test has finished
    fail_count = request.session.testsfailed

    def error_and_kill_pending_tasks():
        loop = asyncio.get_event_loop()
        unfinished_tasks = {
            task
            for task in asyncio.all_tasks(loop)
            if task.get_coro().__name__ not in _ALLOWED_CORO_TASKS and not task.done()
        }
        for task in unfinished_tasks:
            task.cancel()

        # If the tasks are still pending because the test failed
        # for other reasons then we don't have to error here
        if unfinished_tasks and request.session.testsfailed == fail_count:
            raise RuntimeError(
                f"Not all tasks closed during test {request.node.name}:\n"
                f"{pprint.pformat(unfinished_tasks)}"
            )

    request.addfinalizer(error_and_kill_pending_tasks)


@pytest.fixture(scope="function")
def RE(request: FixtureRequest):
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, call_returns_result=True, loop=loop)

    fail_count = request.session.testsfailed

    def clean_event_loop():
        if RE.state not in ("idle", "panicked"):
            try:
                RE.halt()
            except TransitionError:
                pass

        unfinished_tasks = {
            task
            for task in asyncio.all_tasks(loop)
            if task.get_coro().__name__ not in _ALLOWED_CORO_TASKS and not task.done()
        }

        # Cancelling the tasks is thread unsafe, but we get a new event loop
        # with each RE test so we don't need to do this
        loop.call_soon_threadsafe(loop.stop)
        RE._th.join()
        for task in unfinished_tasks:
            task.cancel()
        loop.close()

        # If the tasks are still pending because the test failed
        # for other reasons then we don't have to error here
        if unfinished_tasks and request.session.testsfailed == fail_count:
            raise RuntimeError(
                f"Not all tasks closed during test {request.node.name}:\n"
                f"{pprint.pformat(unfinished_tasks)}"
            )

    request.addfinalizer(clean_event_loop)
    return RE


@pytest.fixture(scope="module", params=["pva"])
def panda_pva():
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "epicscorelibs.ioc",
                "-m",
                macros,
                "-d",
                PANDA_RECORD,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        for macros in [
            "INCLUDE_EXTRA_BLOCK=,INCLUDE_EXTRA_SIGNAL=",
            "EXCLUDE_WIDTH=#,IOC_NAME=PANDAQSRVIB",
            "EXCLUDE_PCAP=#,IOC_NAME=PANDAQSRVI",
        ]
    ]
    time.sleep(2)

    for p in processes:
        assert not p.poll(), p.stdout.read()

    yield processes

    for p in processes:
        p.terminate()
        p.communicate()


@pytest.fixture
async def normal_coroutine() -> Callable[[], Any]:
    async def inner_coroutine():
        await asyncio.sleep(0.01)

    return inner_coroutine


@pytest.fixture
async def failing_coroutine() -> Callable[[], Any]:
    async def inner_coroutine():
        await asyncio.sleep(0.01)
        raise ValueError()

    return inner_coroutine


@pytest.fixture
def static_filename_provider():
    return StaticFilenameProvider("ophyd_async_tests")


@pytest.fixture
def static_path_provider_factory(tmp_path: Path):
    def create_static_dir_provider_given_fp(fp: FilenameProvider):
        return StaticPathProvider(fp, tmp_path)

    return create_static_dir_provider_given_fp


@pytest.fixture
def static_path_provider(
    static_path_provider_factory: callable,
    static_filename_provider: FilenameProvider,
):
    return static_path_provider_factory(static_filename_provider)


@pytest.fixture
def one_shot_trigger_info() -> TriggerInfo:
    return TriggerInfo(
        frame_timeout=None,
        number=1,
        trigger=DetectorTrigger.internal,
        deadtime=None,
        livetime=None,
    )
