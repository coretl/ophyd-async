from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, Dict, Optional

import h5py
import numpy as np
from bluesky.protocols import DataKey, StreamAsset

from ophyd_async.core import PathInfo, PathProvider
from ophyd_async.core.mock_signal_backend import MockSignalBackend
from ophyd_async.core.signal import observe_value, soft_signal_r_and_setter
from ophyd_async.core.utils import DEFAULT_TIMEOUT
from ophyd_async.epics.areadetector.writers.general_hdffile import _HDFDataset, _HDFFile

# raw data path
DATA_PATH = "/entry/data/data"

# pixel sum path
SUM_PATH = "/entry/sum"

MAX_UINT8_VALUE = np.iinfo(np.uint8).max


def generate_gaussian_blob(height: int, width: int) -> np.ndarray:
    """Make a Gaussian Blob with float values in range 0..1"""
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    d = np.sqrt(x * x + y * y)
    blob = np.exp(-(d**2))
    return blob


def generate_interesting_pattern(x: float, y: float) -> float:
    """This function is interesting in x and y in range -10..10, returning
    a float value in range 0..1
    """
    z = 0.5 + (np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)) / 2
    return z


<<<<<<< HEAD
=======
class HdfStreamProvider:
    def __init__(
        self,
        path_info: PathInfo,
        full_file_name: Path,
        datasets: List[DatasetConfig],
    ) -> None:
        self._last_emitted = 0
        self._bundles: List[ComposeStreamResourceBundle] = self._compose_bundles(
            path_info, full_file_name, datasets
        )

    def _compose_bundles(
        self,
        path_info: PathInfo,
        full_file_name: Path,
        datasets: List[DatasetConfig],
    ) -> List[StreamAsset]:
        # path = str(full_file_name.relative_to(path_info.root))
        # root = str(path_info.root)
        bundler_composer = ComposeStreamResource()

        bundles: List[ComposeStreamResourceBundle] = []

        bundles = [
            bundler_composer(
                mimetype="application/x-hdf5",
                uri=f"file://{full_file_name}",
                # spec=SLICE_NAME,
                # root=root,
                # resource_path=path,
                data_key=d.name.replace("/", "_"),
                parameters={
                    "path": d.path,
                    "multiplier": d.multiplier,
                    "timestamps": "/entry/instrument/NDAttributes/NDArrayTimeStamp",
                },
            )
            for d in datasets
        ]
        return bundles

    def stream_resources(self) -> Iterator[StreamResource]:
        for bundle in self._bundles:
            yield bundle.stream_resource_doc

    def stream_data(self, indices_written: int) -> Iterator[StreamDatum]:
        # Indices are relative to resource
        if indices_written > self._last_emitted:
            updated_stream_range = StreamRange(
                start=self._last_emitted,
                stop=indices_written,
            )
            self._last_emitted = indices_written
            for bundle in self._bundles:
                yield bundle.compose_stream_datum(indices=updated_stream_range)
        return None

    def close(self) -> None:
        for bundle in self._bundles:
            bundle.close()


>>>>>>> 588006e6 (Fix panda's  to be absolute path)
class PatternGenerator:
    def __init__(
        self,
        saturation_exposure_time: float = 0.1,
        detector_width: int = 320,
        detector_height: int = 240,
    ) -> None:
        self.saturation_exposure_time = saturation_exposure_time
        self.exposure = saturation_exposure_time
        self.x = 0.0
        self.y = 0.0
        self.height = detector_height
        self.width = detector_width
        self.image_counter: int = 0

        # it automatically initializes to 0
        self.counter_signal, self._set_counter_signal = soft_signal_r_and_setter(int)
        self._full_intensity_blob = (
            generate_gaussian_blob(width=detector_width, height=detector_height)
            * MAX_UINT8_VALUE
        )
        self._hdf_stream_provider: Optional[_HDFFile] = None
        self._handle_for_h5_file: Optional[h5py.File] = None
        self.target_path: Optional[Path] = None

    async def write_image_to_file(self) -> None:
        assert self._handle_for_h5_file, "no file has been opened!"
        # prepare - resize the fixed hdf5 data structure
        # so that the new image can be written
        self._handle_for_h5_file[DATA_PATH].resize(
            (self.image_counter + 1, self.height, self.width)
        )
        self._handle_for_h5_file[SUM_PATH].resize((self.image_counter + 1,))

        # generate the simulated data
        intensity: float = generate_interesting_pattern(self.x, self.y)
        detector_data = (
            self._full_intensity_blob
            * intensity
            * self.exposure
            / self.saturation_exposure_time
        ).astype(np.uint8)

        # write data to disc (intermediate step)
        self._handle_for_h5_file[DATA_PATH][self.image_counter] = detector_data
        sum = np.sum(detector_data)
        self._handle_for_h5_file[SUM_PATH][self.image_counter] = sum

        # save metadata - so that it's discoverable
        self._handle_for_h5_file[DATA_PATH].flush()
        self._handle_for_h5_file[SUM_PATH].flush()

        # counter increment is last
        # as only at this point the new data is visible from the outside
        self.image_counter += 1
        self._set_counter_signal(self.image_counter)

    def set_exposure(self, value: float) -> None:
        self.exposure = value

    def set_x(self, value: float) -> None:
        self.x = value

    def set_y(self, value: float) -> None:
        self.y = value

    async def open_file(
        self, path_provider: PathProvider, name: str, multiplier: int = 1
    ) -> Dict[str, DataKey]:
        await self.counter_signal.connect()

        self.target_path = self._get_new_path(path_provider)
        self._path_provider = path_provider

        self._handle_for_h5_file = h5py.File(self.target_path, "w", libver="latest")

        assert self._handle_for_h5_file, "not loaded the file right"

        self._handle_for_h5_file.create_dataset(
            name=DATA_PATH,
            shape=(0, self.height, self.width),
            dtype=np.uint8,
            maxshape=(None, self.height, self.width),
        )
        self._handle_for_h5_file.create_dataset(
            name=SUM_PATH,
            shape=(0,),
            dtype=np.float64,
            maxshape=(None,),
        )

        # once datasets written, can switch the model to single writer multiple reader
        self._handle_for_h5_file.swmr_mode = True
        self.multiplier = multiplier

        outer_shape = (multiplier,) if multiplier > 1 else ()

        # cache state to self
        # Add the main data
        self._datasets = [
            _HDFDataset(
                data_key=name,
                dataset=DATA_PATH,
                shape=(self.height, self.width),
                multiplier=multiplier,
            ),
            _HDFDataset(
                f"{name}-sum",
                dataset=SUM_PATH,
                shape=(),
                multiplier=multiplier,
            ),
        ]

        describe = {
            ds.data_key: DataKey(
                source="sim://pattern-generator-hdf-file",
                shape=outer_shape + tuple(ds.shape),
                dtype="array" if ds.shape else "number",
                external="STREAM:",
            )
            for ds in self._datasets
        }
        return describe

    def _get_new_path(self, path_provider: PathProvider) -> Path:
        info = path_provider(device_name="pattern")
        filename = info.filename
        new_path: Path = info.root / info.resource_dir / filename
        return new_path

    async def collect_stream_docs(
        self, indices_written: int
    ) -> AsyncIterator[StreamAsset]:
        """
        stream resource says "here is a dataset",
        stream datum says "here are N frames in that stream resource",
        you get one stream resource and many stream datums per scan
        """
        if self._handle_for_h5_file:
            self._handle_for_h5_file.flush()
        # when already something was written to the file
        if indices_written:
            # if no frames arrived yet, there's no file to speak of
            # cannot get the full filename the HDF writer will write
            # until the first frame comes in
            if not self._hdf_stream_provider:
                assert self.target_path, "open file has not been called"
                self._datasets = self._get_datasets()
                self._hdf_stream_provider = _HDFFile(
                    self._path_provider(),
                    self.target_path,
                    self._datasets,
                )
                for doc in self._hdf_stream_provider.stream_resources():
                    yield "stream_resource", doc
            if self._hdf_stream_provider:
                for doc in self._hdf_stream_provider.stream_data(indices_written):
                    yield "stream_datum", doc

    def close(self) -> None:
        if self._handle_for_h5_file:
            self._handle_for_h5_file.close()
            print("file closed")
            self._handle_for_h5_file = None

    async def observe_indices_written(
        self, timeout=DEFAULT_TIMEOUT
    ) -> AsyncGenerator[int, None]:
        async for num_captured in observe_value(self.counter_signal, timeout=timeout):
            yield num_captured // self.multiplier
