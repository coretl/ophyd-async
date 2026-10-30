from collections.abc import Sequence

import bluesky.plan_stubs as bps
from bluesky.utils import plan

from ._io import NDArrayBaseIO
from ._ndattribute import (
    NDAttributeParam,
    NDAttributePv,
    ndattributes_to_xml,
)


@plan
def setup_ndattributes(
    device: NDArrayBaseIO,
    ndattributes: Sequence[NDAttributeParam | NDAttributePv],
):
    xml = ndattributes_to_xml(ndattributes)
    yield from bps.abs_set(
        device.nd_attributes_file,
        xml,
        wait=True,
    )
