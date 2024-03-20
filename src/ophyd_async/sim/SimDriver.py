import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

import h5py
import numpy as np
from bluesky.protocols import Descriptor, StreamAsset
from event_model import (
    ComposeStreamResource,
    ComposeStreamResourceBundle,
    StreamDatum,
    StreamResource,
)

from ophyd_async.core import DirectoryInfo, DirectoryProvider
from ophyd_async.core.signal import SignalR, observe_value
from ophyd_async.core.sim_signal_backend import SimSignalBackend
from ophyd_async.core.utils import DEFAULT_TIMEOUT

# raw data path
DATA_PATH = "/entry/data/data"
# pixel sum path
SUM_PATH = "/entry/sum"

MAX_UINT8_VALUE = np.iinfo(np.uint8).max

SLICE_NAME = "AD_HDF5_SWMR_SLICE"


@dataclass
class DatasetConfig:
    name: str
    shape: Sequence[int]
    maxshape: tuple[Union[None, int], ...] = (None, None, None)
    path: Optional[str] = None
    multiplier: Optional[int] = 1
    dtype: Optional[Any] = None
    fillvalue: Optional[int] = None


def get_full_file_description(
    datasets: List[DatasetConfig], outer_shape: tuple[int, ...]
):
    full_file_description: Dict[str, Descriptor] = {}
    for d in datasets:
        shape = outer_shape + tuple(d.shape)
        source = (f"sim://{d.name}",)
        dtype = ("number" if d.shape == [1] else "array",)
        value = Descriptor(source=source, shape=shape, dtype=dtype, external="stream:")
        full_file_description[d.name] = value
    return full_file_description


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


class HdfStreamProvider:
    def __init__(
        self,
        directory_info: DirectoryInfo,
        full_file_name: Path,
        datasets: List[DatasetConfig],
    ) -> None:
        self._last_emitted = 0
        self._bundles: List[ComposeStreamResourceBundle] = self._compose_bundles(
            directory_info, full_file_name, datasets
        )

    def _compose_bundles(
        self,
        directory_info: DirectoryInfo,
        full_file_name: Path,
        datasets: List[DatasetConfig],
    ) -> List[StreamAsset]:
        path = str(full_file_name.relative_to(directory_info.root))
        root = str(directory_info.root)
        bundler_composer = ComposeStreamResource()

        bundles: List[ComposeStreamResourceBundle] = []

        for d in datasets:
            bundle: ComposeStreamResourceBundle = bundler_composer(
                spec=SLICE_NAME,
                root=root,
                resource_path=path,
                data_key=d.name,
                resource_kwargs={
                    "path": d.path,
                    "multiplier": d.multiplier,
                    "timestamps": "/entry/instrument/NDAttributes/NDArrayTimeStamp",
                },
            )
            bundles += bundle
        return bundles

    def stream_resources(self) -> Iterator[StreamResource]:
        for bundle in self._bundles:
            yield bundle.stream_resource_doc

    def stream_data(self, indices_written: int) -> Iterator[StreamDatum]:
        # Indices are relative to resource
        if indices_written > self._last_emitted:
            indices = dict(
                start=self._last_emitted,
                stop=indices_written,
            )
            self._last_emitted = indices_written
            for bundle in self._bundles:
                yield bundle.compose_stream_datum(indices)
        return None

    def close(self) -> None:
        for bundle in self._bundles:
            bundle.close()


class SimDriver:
    def __init__(
        self,
        saturation_exposure_time: float = 1,
        detector_width: int = 320,
        detector_height: int = 240,
    ) -> None:
        self.saturation_exposure_time = saturation_exposure_time
        self.exposure = saturation_exposure_time
        self.x = 0.0
        self.y = 0.0
        self.height = detector_height
        self.width = detector_width
        self.written_images_counter: int = 0

        self.sim_signal = SignalR(
            SimSignalBackend(int, str(self.written_images_counter))
        )
        self.STARTING_BLOB = (
            generate_gaussian_blob(width=detector_width, height=detector_height)
            * MAX_UINT8_VALUE
        )
        self._hdf_stream_provider: Optional[HdfStreamProvider] = None
        self._handle_for_h5_file: Optional[h5py.File] = None

    async def write_image_to_file(self) -> None:
        assert self._handle_for_h5_file, "no file has been opened!"
        self._handle_for_h5_file.create_dataset(
            name=f"pattern-generator-file-{self.written_images_counter}",
            dtype=np.ndarray,
        )

        await self.sim_signal.connect(sim=True)
        # prepare - resize the fixed hdf5 data structure
        # so that the new image can be written
        new_layer = self.written_images_counter + 1
        target_dimensions = (new_layer, self.height, self.width)

        self._handle_for_h5_file[DATA_PATH].resize(target_dimensions)
        self._handle_for_h5_file[SUM_PATH].resize(new_layer)

        # generate the simulated data
        intensity: float = generate_interesting_pattern(self.x, self.y)
        detector_data: np.uint8 = np.uint8(
            self.STARTING_BLOB
            * intensity
            * self.exposure
            / self.saturation_exposure_time
        )

        # write data to disc (intermediate step)
        self._handle_for_h5_file[DATA_PATH][self.written_images_counter] = detector_data
        self._handle_for_h5_file[SUM_PATH][self.written_images_counter] = np.sum(
            detector_data
        )

        # save metadata - so that it's discoverable
        self._handle_for_h5_file[DATA_PATH].flush()
        self._handle_for_h5_file[SUM_PATH].flush()

        # counter increment is last
        # as only at this point the new data is visible from the outside
        self.written_images_counter += 1

    def set_exposure(self, value: float) -> None:
        self.exposure = value

    def set_x(self, value: float) -> None:
        self.x = value

    def set_y(self, value: float) -> None:
        self.y = value

    async def open_file(
        self, directory: DirectoryProvider, multiplier: int = 1
    ) -> Dict[str, Descriptor]:
        file_ref_object = self._get_file_ref_object(directory)
        self.multiplier = multiplier
        self._directory_provider = directory

        datasets = self._get_datasets()

        for d in datasets:
            file_ref_object.create_dataset(
                name=d.name,
                shape=d.shape,
                dtype=d.dtype,
            )
        file_ref_object.swmr_mode = True

        self._handle_for_h5_file = file_ref_object
        self._hdf_stream_provider = HdfStreamProvider(
            directory(),
            self._handle_for_h5_file,
            datasets,
        )

        outer_shape = (multiplier,) if multiplier > 1 else ()
        print("file opened")
        full_file_description = get_full_file_description(datasets, outer_shape)
        self._datasets = datasets
        return full_file_description

    def _get_datasets(self) -> List[DatasetConfig]:
        raw_dataset = DatasetConfig(
            name=DATA_PATH,
            dtype=np.uint8,
            shape=(1, self.height, self.width),
            maxshape=(None, self.height, self.width),
        )

        sum_dataset = DatasetConfig(
            name=SUM_PATH,
            dtype=np.float64,
            shape=(1,),
            maxshape=(None, None, None),
            fillvalue=-1,
        )

        datasets: List[DatasetConfig] = [raw_dataset, sum_dataset]
        return datasets

    def _get_file_ref_object(self, dir: DirectoryProvider) -> h5py.File:
        info = dir()
        filename = f"{info.prefix}pattern{info.suffix}.h5"
        new_path: Path = info.root / info.resource_dir / filename
        h5py_file_ref_object = h5py.File(new_path, "w")
        return h5py_file_ref_object

    async def collect_stream_docs(
        self, indices_written: int
    ) -> AsyncIterator[StreamAsset]:
        if self._handle_for_h5_file:
            self._handle_for_h5_file.flush()
        # when already something was written to the file
        if indices_written:
            # if no frames arrived yet, there's no file to speak of
            # cannot get the full filename the HDF writer will write
            # until the first frame comes in
            if not self._hdf_stream_provider:
                assert not self._datasets, "datasets not initialized"
                datasets = self._get_datasets()
                self._datasets = datasets
                self._hdf_stream_provider = HdfStreamProvider(
                    self._directory_provider(),
                    self._handle_for_h5_file,
                    self._datasets,
                )
                for doc in self._hdf_stream_provider.stream_resources():
                    yield "stream_resource", doc
        for doc in self._handle_for_h5_file.stream_data(indices_written):
            yield "stream_datum", doc

    def close(self) -> None:
        # self._handle_for_h5_file.close()
        print("file closed")
        self._handle_for_h5_file = None

    async def observe_indices_written(
        self, timeout=DEFAULT_TIMEOUT
    ) -> AsyncGenerator[int, None]:
        async for num_captured in observe_value(self.sim_signal, timeout=timeout):
            yield num_captured // self.multiplier
