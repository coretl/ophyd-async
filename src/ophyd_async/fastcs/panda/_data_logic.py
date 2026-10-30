import asyncio
import logging

import numpy as np

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    PathProvider,
    StreamResourceDataProvider,
    StreamResourceInfo,
    UnboundedDataLogic,
    wait_for_value,
)

from ._block import DataBlock, PandaCaptureMode

logger = logging.getLogger("ophyd_async")


class PandaHDFDataLogic(UnboundedDataLogic[StreamResourceDataProvider]):
    def __init__(
        self,
        path_provider: PathProvider,
        data_block: DataBlock,
    ):
        self.path_provider = path_provider
        self.data_block = data_block

    async def start_provider(
        self,
        name: str,
        collections_per_event: int,
        collection_period: float,
        flush_period: float,
    ) -> StreamResourceDataProvider:
        # TODO: size the chunk from `collection_period` once the IOC exposes the
        # chunk-size signal (see the chunk_shape TODO below).
        path_info = self.path_provider(name)
        # Set create dir depth first to guarantee that callback when setting
        # directory path has correct value
        await self.data_block.create_directory.set(path_info.create_dir_depth)
        # Setup the HDF writer. The PandA flushes on a timer rather than by
        # chunk size, so it takes the flush period directly; 0 means it flushes
        # only when its buffer fills.
        await asyncio.gather(
            self.data_block.flush_period.set(flush_period),
            self.data_block.hdf_directory.set(str(path_info.directory_path)),
            self.data_block.hdf_file_name.set(
                f"{path_info.filename}.h5",
            ),
            self.data_block.capture_mode.set(PandaCaptureMode.FOREVER),
        )
        # Make sure that directory exists or has been created.
        if not await self.data_block.directory_exists.get_value() == 1:
            raise OSError(
                f"Directory {path_info.directory_path} does not exist or "
                "is not writable by the PandABlocks-ioc!"
            )
        # Start capturing
        await self.data_block.capture.set(True)
        # Capturing resets NumCaptured, which the detector reads as its baseline
        await wait_for_value(self.data_block.num_captured, 0, timeout=DEFAULT_TIMEOUT)
        # The datasets the PandA writes: PandABlocks-ioc only fills the table
        # once capturing has started
        capture_table = await self.data_block.datasets.get_value()
        if len(capture_table) == 0:
            logger.warning(
                f"PandA {name} DATASETS table is empty! "
                "No stream resource docs will be generated. "
                "Make sure captured positions have their corresponding "
                "*:DATASET PV set to a scientifically relevant name."
            )
        resources = [
            StreamResourceInfo(
                data_key=dataset_name,
                shape=(),
                dtype_numpy=np.dtype(np.float64).str,
                # TODO: Update chunk size to read signal once available in IOC
                # Currently PandA IOC sets chunk size to 1024 points per chunk
                chunk_shape=(1024,),
                parameters={"dataset": f"/{dataset_name}"},
            )
            for dataset_name in capture_table.name
        ]
        return StreamResourceDataProvider(
            uri=f"{path_info.directory_uri}{path_info.filename}.h5",
            resources=resources,
            mimetype="application/x-hdf5",
            collections_written=self.data_block.num_captured,
            collections_per_event=collections_per_event,
        )

    async def ensure_stopped(self) -> None:
        await self.data_block.capture.set(False)
