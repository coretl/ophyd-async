from collections.abc import Sequence

from ophyd_async.core import PathProvider, SignalR
from ophyd_async.epics.adcore import (
    ADHDFWriter,
    ADWriter,
    AreaDetector,
    NDPluginBaseIO,
)

from ._kinetix_controller import KinetixController


class KinetixDetector(AreaDetector[KinetixController, ADWriter]):
    """
    Ophyd-async implementation of an ADKinetix Detector.
    https://github.com/NSLS-II/ADKinetix
    """

    def __init__(
        self,
        prefix: str,
        path_provider: PathProvider,
        drv_suffix: str = "cam1:",
        writer_cls: type[ADWriter] = ADHDFWriter,
        fileio_suffix: str | None = None,
        name: str = "",
        config_sigs: Sequence[SignalR] = (),
        plugins: dict[str, NDPluginBaseIO] | None = None,
        use_fileio_for_ds_describer: bool = False,
    ):
        controller, driver = KinetixController.controller_and_drv(
            prefix + drv_suffix, name=name
        )
        writer, fileio = writer_cls.writer_and_io(
            prefix,
            path_provider,
            lambda: name,
            ds_describer_source=driver if not use_fileio_for_ds_describer else None,
            fileio_suffix=fileio_suffix,
            plugins=plugins,
        )

        super().__init__(
            driver=driver,
            controller=controller,
            fileio=fileio,
            writer=writer,
            plugins=plugins,
            name=name,
            config_sigs=config_sigs,
        )
        self.drv = driver
        self.fileio = fileio
