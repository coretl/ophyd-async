from collections.abc import Sequence

import numpy as np

from ophyd_async.core import (
    PathProvider,
    StreamResourceDataProvider,
    StreamResourceInfo,
    UnboundedDataLogic,
)

from ._pattern_generator import DATA_PATH, SUM_PATH, PatternGenerator

WIDTH = 320
HEIGHT = 240


class BlobDataLogic(UnboundedDataLogic[StreamResourceDataProvider]):
    def __init__(
        self,
        path_provider: PathProvider,
        pattern_generator: PatternGenerator,
    ):
        self.path_provider = path_provider
        self.pattern_generator = pattern_generator
        self.collections_written = self.pattern_generator.images_written

    async def start_provider(
        self,
        name: str,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> StreamResourceDataProvider:
        # The sim blob writer uses a fixed chunk shape and writes for as long as
        # it is told to, so neither period is needed
        path_info = self.path_provider(name)
        self.pattern_generator.open_file(
            path_info.directory_path / f"{path_info.filename}.h5", WIDTH, HEIGHT
        )
        data_resource = StreamResourceInfo(
            data_key=name,
            shape=(HEIGHT, WIDTH),
            # NDAttributes appear to always be configured with
            # this chunk size
            chunk_shape=(1, HEIGHT, WIDTH),
            dtype_numpy=np.dtype(np.uint8).str,
            parameters={"dataset": DATA_PATH},
        )
        sum_resource = StreamResourceInfo(
            data_key=f"{name}-sum",
            shape=(),
            # NDAttributes appear to always be configured with
            # this chunk size
            chunk_shape=(1024,),
            dtype_numpy=np.dtype(np.int64).str,
            parameters={"dataset": SUM_PATH},
        )
        return StreamResourceDataProvider(
            uri=f"{path_info.directory_uri}{path_info.filename}.h5",
            resources=[data_resource, sum_resource],
            mimetype="application/x-hdf5",
            collections_per_event=collections_per_event,
        )

    async def ensure_stopped(self) -> None:
        self.pattern_generator.close_file()

    def get_hinted_fields(self, name: str) -> Sequence[str]:
        # The main dataset is always hinted
        return [name]
