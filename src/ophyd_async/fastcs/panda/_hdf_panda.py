from __future__ import annotations

from collections.abc import Sequence

from ophyd_async.core import PathProvider, SignalR, StandardDetector
from ophyd_async.epics.pvi import PviDeviceConnector

from ._block import CommonPandaBlocks
from ._control import PandaPcapController
from ._writer import PandaHDFWriter


class HDFPanda(CommonPandaBlocks, StandardDetector):
    def __init__(
        self,
        uri: str,
        path_provider: PathProvider,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ):
        # TODO: add Tango support
        # This has to be first so we make self.pcap
        connector = PviDeviceConnector(uri + "PVI")
        connector.create_children_from_annotations(self)
        controller = PandaPcapController(pcap=self.pcap)
        writer = PandaHDFWriter(
            path_provider=path_provider,
            name_provider=lambda: name,
            panda_data_block=self.data,
        )
        super().__init__(
            controller=controller,
            writer=writer,
            config_sigs=config_sigs,
            name=name,
            connector=connector,
        )
