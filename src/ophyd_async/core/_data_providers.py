import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import numpy as np
from bluesky.protocols import StreamAsset
from event_model import ComposeStreamResource, DataKey, StreamRange
from event_model.documents import PartialEventPage

from ._datatypes import Array1D
from ._signal import SignalR, SignalW
from ._utils import ConfinedModel, error_if_none, gather_dict, get_dtype


class DataProvider(ABC):
    """What a [](#DetectorDataLogic) returns to publish the data it is writing.

    The two kinds, [](#StreamableDataProvider) and [](#PageableDataProvider),
    differ in how the data is published; what they share is here. A provider
    counts its events from an origin: collection 0 of its data logic's
    `collections_written` unless `start_new_resources` says otherwise.

    :param collections_per_event: how many collections make up one event
    """

    def __init__(self, collections_per_event: int) -> None:
        self.collections_per_event = collections_per_event

    @abstractmethod
    async def make_datakeys(self) -> dict[str, DataKey]:
        """Return a DataKey for each field, of shape `[collections_per_event, ...]`.

        Called before the first exposure is taken.
        """

    @abstractmethod
    def start_new_resources(self, origin: int) -> None:
        """Count events from collection `origin`, with nothing yet published.

        Called on an unbounded provider at each `prepare()`, so each run has its
        own stream resources.
        """


class StreamableDataProvider(DataProvider):
    """Publishes its data as stream assets: a StreamResource and StreamDatums."""

    @abstractmethod
    def make_stream_docs(self, index: int) -> AsyncIterator[StreamAsset]:
        """Make the StreamAsset documents for the events up to `index`.

        Emits the StreamResources the first time, then StreamDatums from where
        the last call got to.

        :param index: the event to publish up to, counted from the origin
        """


class PageableDataProvider(DataProvider):
    """Publishes its data as event pages.

    For a source read back as arrays rather than written to a file: an
    areaDetector stats time series, or a scaler's MCA arrays.
    """

    @abstractmethod
    async def make_page(self, index: int) -> PartialEventPage | None:
        """Return one page of the events from where the last call got to.

        :param index: the event to publish up to
        :return: the page, or `None` if there are no new events
        """


class EventPageDataProvider(PageableDataProvider):
    """Emits a fixed-length array of collections as event pages.

    For a device that fills one array per field as it acquires, alongside an
    array of per-collection timestamps: an areaDetector stats time series, or a
    scaler's MCA arrays. Each array is sliced into `collections_per_event`-length
    chunks, one per event, and the last collection of an event supplies that
    event's timestamp.

    :param data: datakey (already suffixed) to the array signal that backs it
    :param timestamps: the acquisition time of each collection
    :param collections_per_event: how many collections make up one event
    """

    def __init__(
        self,
        data: Mapping[str, SignalR[Array1D[Any]]],
        timestamps: SignalR[Array1D[np.float64]],
        collections_per_event: int,
    ) -> None:
        super().__init__(collections_per_event)
        self.data = dict(data)
        self.timestamps = timestamps
        self.start_new_resources(0)

    def start_new_resources(self, origin: int) -> None:
        self._origin = origin
        # The events published by make_page
        self._published = 0

    async def make_datakeys(self) -> dict[str, DataKey]:
        return {
            datakey: DataKey(
                source=signal.source,
                shape=[self.collections_per_event],
                dtype="array",
                dtype_numpy=get_dtype(
                    error_if_none(signal.datatype, f"{signal.source} has no datatype")
                ).str,
            )
            for datakey, signal in self.data.items()
        }

    async def make_page(self, index: int) -> PartialEventPage | None:
        if index <= self._published:
            return None
        cpe = self.collections_per_event
        # Read every array and the per-collection timestamps in one parallel batch
        arrays, stamps = await asyncio.gather(
            gather_dict(
                {datakey: signal.get_value() for datakey, signal in self.data.items()}
            ),
            self.timestamps.get_value(),
        )
        starts = [self._origin + event * cpe for event in range(self._published, index)]
        self._published = index
        # One timestamp per event: the acquisition time of its last collection
        event_times = [float(stamps[start + cpe - 1]) for start in starts]
        data = {
            datakey: [array[start : start + cpe] for start in starts]
            for datakey, array in arrays.items()
        }
        return {
            # TODO: numpy arrays are not in PartialEventPage's value type, but
            # they round trip through the bundler fine; fix upstream in
            # event-model rather than converting to lists here
            "data": data,  # type: ignore[typeddict-item]
            "time": event_times,
            "timestamps": dict.fromkeys(data, event_times),
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

    :param uri: the file that is being written
    :param resources: descriptions of each of the datasets that will appear in it
    :param mimetype: the mimetype of the file
    :param collections_per_event: how many collections make up one event
    :param flush_signal: set before making stream docs, to flush the file
    """

    def __init__(
        self,
        uri: str,
        resources: Sequence[StreamResourceInfo],
        mimetype: str,
        collections_per_event: int,
        flush_signal: SignalW[bool] | None = None,
    ) -> None:
        super().__init__(collections_per_event)
        self.uri = uri
        self.resources = list(resources)
        self.mimetype = mimetype
        self.flush_signal = flush_signal
        self.start_new_resources(0)

    def start_new_resources(self, origin: int) -> None:
        bundler_composer = ComposeStreamResource()
        self.bundles = [
            bundler_composer(
                mimetype=self.mimetype,
                uri=self.uri,
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
        self._resources_emitted = False
        self._origin = origin
        # The collections in the file that StreamDatums have been made for
        self._published = origin

    async def make_datakeys(self) -> dict[str, DataKey]:
        cpe = self.collections_per_event
        return {
            resource.data_key: DataKey(
                source=resource.source or self.uri,
                shape=[cpe, *resource.shape],
                dtype="array" if cpe > 1 or len(resource.shape) > 0 else "number",
                dtype_numpy=resource.dtype_numpy,
                external="STREAM:",
            )
            for resource in self.resources
        }

    async def make_stream_docs(self, index: int) -> AsyncIterator[StreamAsset]:
        if self.flush_signal:
            await self.flush_signal.set(True)
        # TODO: fail if we get dropped frames
        cpe = self.collections_per_event
        written = self._origin + index * cpe
        # StreamDatum indices are events in the file, which may hold earlier runs
        indices: StreamRange = {"start": self._published // cpe, "stop": written // cpe}
        if indices["stop"] > indices["start"]:
            if not self._resources_emitted:
                self._resources_emitted = True
                for bundle in self.bundles:
                    yield "stream_resource", bundle.stream_resource_doc
            for bundle in self.bundles:
                yield "stream_datum", bundle.compose_stream_datum(indices)
            self._published = written
