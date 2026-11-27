from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from bluesky.protocols import StreamAsset
from event_model import ComposeStreamResource, DataKey, StreamRange

from ._signal import SignalR, SignalW
from ._utils import ConfinedModel


class DataProvider(ABC):
    """What a [](#DetectorDataLogic) returns to publish the data it is writing.

    A provider counts its events from an origin: collection 0 of
    `collections_written` unless `start_new_resources` says otherwise.

    :param collections_written: counts the collections written so far, reset to
        0 by the data logic before it returns the provider
    :param collections_per_event: how many collections make up one event
    """

    def __init__(
        self, collections_written: SignalR[int], collections_per_event: int
    ) -> None:
        self.collections_written = collections_written
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
    :param collections_written: counts the collections written so far
    :param collections_per_event: how many collections make up one event
    :param flush_signal: set before making stream docs, to flush the file
    """

    def __init__(
        self,
        uri: str,
        resources: Sequence[StreamResourceInfo],
        mimetype: str,
        collections_written: SignalR[int],
        collections_per_event: int,
        flush_signal: SignalW[bool] | None = None,
    ) -> None:
        super().__init__(collections_written, collections_per_event)
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
