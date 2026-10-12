from abc import abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any

import numpy as np
from bluesky.protocols import StreamAsset
from event_model import ComposeStreamResource, DataKey, StreamRange
from event_model.documents import PartialEventPage

from ._datatypes import Array1D
from ._signal import SignalR, SignalW
from ._utils import ConfinedModel, error_if_none, gather_dict, get_dtype


class DataProvider:
    """What a [](#DetectorDataLogic) makes to publish one scan's data.

    The two kinds, [](#StreamableDataProvider) and [](#PageableDataProvider),
    differ in how the data is published; what they share is here.
    """

    #: Counts the collections written so far
    collections_written_signal: SignalR[int]

    @abstractmethod
    async def make_datakeys(self, collections_per_event: int) -> dict[str, DataKey]:
        """Return a DataKey for each field this provider produces.

        Called before the first exposure is taken.

        :param collections_per_event: this should appear in the shape of each DataKey
        """

    async def step_boundary(self) -> None:
        """Get ready for the next point of a step scan.

        Called at each step point after the first, before the detector reads
        `collections_written_signal` as that point's starting count. A finite
        buffer re-arms itself here; a source that keeps counting does nothing,
        which is the default.
        """


class StreamableDataProvider(DataProvider):
    """Publishes its data as stream assets: a StreamResource and StreamDatums."""

    async def make_stream_docs(
        self, collections_written: int, collections_per_event: int
    ) -> AsyncIterator[StreamAsset]:
        """Make StreamAsset documents up to the given index.

        Default implementation is a no-op. Subclasses should override this
        to emit actual StreamAsset documents.
        """
        while False:
            yield


class PageableDataProvider(DataProvider):
    """Publishes its data as event pages.

    For a source read back as arrays rather than written to a file: an
    areaDetector stats time series, or a scaler's MCA arrays. The detector
    keeps track of what it has already published, so `make_page` is asked for
    a range of collections and holds no state of its own.
    """

    @abstractmethod
    async def make_page(
        self, collections_from: int, collections_to: int, collections_per_event: int
    ) -> PartialEventPage | None:
        """Return one page of the events in a range of collections.

        :param collections_from: the first collection of the first event
        :param collections_to: the collection after the last one to include
        :param collections_per_event: how many collections make up one event
        :return: a page of `(collections_to - collections_from) //
            collections_per_event` events, or `None` if that is 0
        """


class EventPageDataProvider(PageableDataProvider):
    """Emits a fixed-length array of collections as event pages.

    For a device that fills one array per field as it acquires, alongside an
    array of per-collection timestamps and a signal counting how many
    collections it has filled: an areaDetector stats time series, or a scaler's
    MCA arrays. Each array is sliced into `collections_per_event`-length chunks,
    one per event, and the last collection of an event supplies that event's
    timestamp.

    :param data: datakey (already suffixed) to the array signal that backs it
    :param collections_written_signal: counts the collections filled so far
    :param timestamps: the acquisition time of each collection
    :param rearm: clears the arrays and restarts filling them, called at each
        step point after the first; `None` for arrays that keep filling
    """

    def __init__(
        self,
        data: Mapping[str, SignalR[Array1D[Any]]],
        collections_written_signal: SignalR[int],
        timestamps: SignalR[Array1D[np.float64]],
        rearm: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.data = dict(data)
        self.collections_written_signal = collections_written_signal
        self.timestamps = timestamps
        self.rearm = rearm

    async def make_datakeys(self, collections_per_event: int) -> dict[str, DataKey]:
        return {
            datakey: DataKey(
                source=signal.source,
                shape=[collections_per_event],
                dtype="array",
                dtype_numpy=get_dtype(
                    error_if_none(signal.datatype, f"{signal.source} has no datatype")
                ).str,
            )
            for datakey, signal in self.data.items()
        }

    async def step_boundary(self) -> None:
        if self.rearm:
            await self.rearm()

    async def make_page(
        self, collections_from: int, collections_to: int, collections_per_event: int
    ) -> PartialEventPage | None:
        events = range((collections_to - collections_from) // collections_per_event)
        if not events:
            return None
        # Read every array and the per-collection timestamps in one parallel batch
        read: dict[SignalR[Array1D[Any]], Array1D[Any]] = await gather_dict(
            {
                signal: signal.get_value()
                for signal in (*self.data.values(), self.timestamps)
            }
        )
        starts = [collections_from + e * collections_per_event for e in events]
        stamps = read[self.timestamps]
        # One timestamp per event: the acquisition time of its last collection
        event_times = [
            float(stamps[start + collections_per_event - 1]) for start in starts
        ]
        return {
            # TODO: numpy arrays are not in PartialEventPage's value type, but
            # they round trip through the bundler fine; fix upstream in
            # event-model rather than converting to lists here
            "data": {  # type: ignore[typeddict-item]
                datakey: [
                    read[signal][start : start + collections_per_event]
                    for start in starts
                ]
                for datakey, signal in self.data.items()
            },
            "time": event_times,
            "timestamps": dict.fromkeys(self.data, event_times),
        }


class StreamResourceInfo(ConfinedModel):
    """A description of a single StreamResource that should be emitted."""

    data_key: str
    """The data_key that will appear in the event descriptor,
    e.g. det or det.data"""

    shape: tuple[int | None, ...]
    """The shape of a single collection's data in the HDF file,
    e.g. (768, 1024) for arrays or () for scalars"""

    chunk_shape: tuple[int, ...]
    """The explicit chunk size written to disk"""

    dtype_numpy: str
    """The numpy dtype for this field,
    e.g. <i2 or <f8"""

    parameters: dict[str, Any]
    """Any other parameters that should be included in the StreamResource,
    e.g. dataset path"""

    source: str = ""
    """The source string that should appear in the event descriptor, blank means use uri
    e.g. ca://HDF:FullFileName_RBV"""


class StreamResourceDataProvider(StreamableDataProvider):
    """A helper class to make stream resource and datums for HDF datasets.

    :param full_file_name: Absolute path to the file that has been written
    :param datasets: Descriptions of each of the datasets that will appear in the file
    """

    def __init__(
        self,
        uri: str,
        resources: Sequence[StreamResourceInfo],
        mimetype: str,
        collections_written_signal: SignalR[int],
        flush_signal: SignalW[bool] | None = None,
    ) -> None:
        self.uri = uri
        self.resources = list(resources)
        self.collections_written_signal = collections_written_signal
        self.flush_signal = flush_signal
        self.last_emitted = 0
        bundler_composer = ComposeStreamResource()
        self.bundles = [
            bundler_composer(
                mimetype=mimetype,
                uri=uri,
                data_key=resource.data_key,
                parameters={
                    "chunk_shape": resource.chunk_shape,
                    **resource.parameters,
                },
                uid=None,
                validate=True,
            )
            for resource in self.resources
        ]

    async def make_datakeys(self, collections_per_event: int) -> dict[str, DataKey]:
        describe = {
            resource.data_key: DataKey(
                source=resource.source or self.uri,
                shape=[collections_per_event, *resource.shape],
                dtype="array"
                if collections_per_event > 1 or len(resource.shape) > 0
                else "number",
                dtype_numpy=resource.dtype_numpy,
                external="STREAM:",
            )
            for resource in self.resources
        }
        return describe

    async def make_stream_docs(
        self, collections_written: int, collections_per_event: int
    ) -> AsyncIterator[StreamAsset]:
        if self.flush_signal:
            await self.flush_signal.set(True)
        # TODO: fail if we get dropped frames
        indices_written = collections_written // collections_per_event
        if indices_written and not self.last_emitted:
            for bundle in self.bundles:
                yield "stream_resource", bundle.stream_resource_doc

        if indices_written > self.last_emitted:
            indices: StreamRange = {
                "start": self.last_emitted,
                "stop": indices_written,
            }
            self.last_emitted = indices_written
            for bundle in self.bundles:
                yield "stream_datum", bundle.compose_stream_datum(indices)
