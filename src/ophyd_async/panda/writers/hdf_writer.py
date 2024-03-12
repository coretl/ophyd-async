import asyncio
import atexit
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from bluesky.protocols import Asset, Descriptor, Hints
from p4p.client.thread import Context

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    DetectorWriter,
    Device,
    DirectoryProvider,
    NameProvider,
    SignalR,
    SignalRW,
    wait_for_value,
)
from ophyd_async.panda.panda import PandA

from .panda_hdf import _HDFDataset, _HDFFile


@dataclass()
class HdfSignals:
    file_path: SignalRW  # This is the directory rather than path. Path is read only
    file_name: SignalRW
    num_capture: SignalRW
    num_captured: SignalR
    capture: SignalRW


class Capture(str, Enum):
    No = "No"
    Value = "Value"
    Diff = "Diff"
    Sum = "Sum"
    Mean = "Mean"
    Min = "Min"
    Max = "Max"
    MinMax = "Min Max"
    MinMaxMean = "Min Max Mean"


def get_capture_signals(
    panda: Device, path_prefix: Optional[str] = ""
) -> Dict[str, SignalR]:
    """Get dict mapping a capture signal's name to the signal itself"""
    # TODO makecommon with code from device_save_loader?
    if not path_prefix:
        path_prefix = ""
    signals: Dict[str, SignalR[Any]] = {}
    for attr_name, attr in panda.children():
        dot_path = f"{path_prefix}{attr_name}"
        if isinstance(attr, SignalR) and attr_name.endswith("_capture"):
            signals[dot_path] = attr
        attr_signals = get_capture_signals(attr, path_prefix=dot_path + ".")
        signals.update(attr_signals)
    return signals


# This should return a dictionary which maps to a dict which contains the Capture
# signal object,
# and the value of that signal
async def get_signals_marked_for_capture(
    capture_signals: Dict[str, SignalR]
) -> Dict[str, Union[SignalR, Capture]]:
    # Read signals to see if they should be captured

    do_read = [signal.get_value() for signal in capture_signals.values()]

    signal_values = await asyncio.gather(*do_read)

    assert len(signal_values) == len(
        capture_signals.keys()
    ), "Length of read signals are different to length of signals"
    signals_to_capture: Dict[str, Dict[str, SignalR | Capture]] = {}
    for signal_path, signal_object, signal_value in zip(
        capture_signals.keys(), capture_signals.values(), signal_values
    ):
        if (signal_value.value in iter(Capture)) and (signal_value.value != Capture.No):
            signals_to_capture[signal_path] = {
                "signal": signal_object,
                "capture_type": signal_value.value,
            }

    return signals_to_capture


class PandaHDFWriter(DetectorWriter):
    # hdf: DataBlock
    _ctxt: Optional[Context] = None

    @property
    def ctxt(self) -> Context:
        if PandaHDFWriter._ctxt is None:
            PandaHDFWriter._ctxt = Context("pva", nt=False)

            @atexit.register
            def _del_ctxt():
                # If we don't do this we get messages like this on close:
                #   Error in sys.excepthook:
                #   Original exception was:
                PandaHDFWriter._ctxt = None

        return PandaHDFWriter._ctxt

    def __init__(
        self,
        prefix: str,
        directory_provider: DirectoryProvider,
        name_provider: NameProvider,
        panda_device: PandA,
    ) -> None:
        self.panda_device = panda_device
        self._prefix = prefix
        self._directory_provider = directory_provider
        self._name_provider = name_provider
        self._datasets: List[_HDFDataset] = []
        self._file: Optional[_HDFFile] = None

        # TODO add panda.data to the Panda device instead, as a typed block
        self.hdf = HdfSignals(
            panda_device.data.hdfdirectory,
            panda_device.data.hdffilename,
            panda_device.data.numcapture,
            panda_device.data.numcaptured,
            panda_device.data.capture,
        )

        # Get capture PVs by looking at panda. Gives mapping of dotted attribute path
        # to Signal object
        self.capture_signals = get_capture_signals(self.panda_device)

    # Triggered on PCAP arm
    async def open(self, multiplier: int = 1) -> Dict[str, Descriptor]:
        """Retrieve and get descriptor of all PandA signals marked for capture"""

        # Ensure flushes are immediate
        await self.panda_device.data.flushperiod.set(0)

        self.to_capture = await get_signals_marked_for_capture(self.capture_signals)
        self._file = None
        info = self._directory_provider()
        await asyncio.gather(
            self.hdf.file_path.set(info.directory_path),
            self.hdf.file_name.set(f"{info.filename_prefix}.h5"),
        )

        # TODO confirm all missing functionality from AD writer isn't needed here

        await self.hdf.num_capture.set(0)
        # Wait for it to start, stashing the status that tells us when it finishes
        await self.hdf.capture.set(True)
        name = self._name_provider()
        if multiplier > 1:
            raise ValueError(
                "All PandA datasets should be scalar, multiplier should be 1"
            )
        self._datasets = []
        for attribute_path, value in self.to_capture.items():
            # TODO check that a 'abc_capture' signal always records an 'abc_val' signal
            signal_name = attribute_path.split(".")[-1]
            block_name = attribute_path.split(".")[-2]

            # Get block names from numbered blocks, eg INENC[1]
            if block_name.isnumeric():
                actual_block = attribute_path.split(".")[-3]
                block_name = f"{actual_block}.{block_name}"

            if value["capture_type"] == Capture.MinMaxMean:
                shape = [3]
            elif value["capture_type"] == Capture.MinMax:
                shape = [2]
            else:
                shape = [1]
                # TODO check if this is correct format

            self._datasets.append(
                _HDFDataset(
                    name,
                    block_name,
                    f"{name}.{block_name}.{signal_name}",
                    f"{block_name}:{signal_name}".upper(),
                    shape,
                    multiplier=1,
                )
            )

        describe = {
            ds.name: Descriptor(
                source=self.hdf.file_path.source,
                shape=ds.shape,
                dtype="array" if ds.shape != [1] else "number",
                external="STREAM:",
            )
            for ds in self._datasets
        }
        return describe

    # Next two functions are exactly the same as AD writer. Could move as default
    # StandardDetector behavior
    async def wait_for_index(
        self, index: int, timeout: Optional[float] = DEFAULT_TIMEOUT
    ):
        def matcher(value: int) -> bool:
            return value >= index

        matcher.__name__ = f"index_at_least_{index}"
        await wait_for_value(self.hdf.num_captured, matcher, timeout=timeout)

    async def get_indices_written(self) -> int:
        return await self.hdf.num_captured.get_value()

    async def collect_stream_docs(self, indices_written: int) -> AsyncIterator[Asset]:
        # TODO: fail if we get dropped frames
        if indices_written:
            if not self._file:
                self._file = _HDFFile(
                    await self.hdf.file_name.get_value(), self._datasets
                )
                for doc in self._file.stream_resources():
                    yield "stream_resource", doc
            for doc in self._file.stream_data(indices_written):
                yield "stream_datum", doc

    # Could put this function as default for StandardDetector
    async def close(self):
        await self.hdf.capture.set(False, wait=True, timeout=DEFAULT_TIMEOUT)

    @property
    def hints(self) -> Hints:
        return {"fields": [self._name_provider()]}
