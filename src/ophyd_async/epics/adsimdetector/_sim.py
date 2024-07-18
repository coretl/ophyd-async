from typing import Sequence

from ophyd_async.core import PathProvider, SignalR, StandardDetector
from ophyd_async.epics import adcore

from ._sim_controller import ADSimController


class DemoADSimDetector(StandardDetector):
    _controller: ADSimController
    _writer: adcore.HDFWriter

    def __init__(
        self,
        drv: adcore.ADBase,
        hdf: adcore.NDFileHDF,
        path_provider: PathProvider,
        name: str = "",
        config_sigs: Sequence[SignalR] = (),
    ):
        self.drv = drv
        self.hdf = hdf

        super().__init__(
            ADSimController(self.drv),
            adcore.HDFWriter(
                self.hdf,
                path_provider,
                lambda: self.name,
                adcore.ADBaseShapeProvider(self.drv),
            ),
            config_sigs=config_sigs,
            name=name,
        )
