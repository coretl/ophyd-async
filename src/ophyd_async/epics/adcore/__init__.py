"""Core components of the areaDetector software.

https://github.com/areaDetector/ADCore
"""

from ._acquire_logic import ADAcquireLogic, ADContAcqAcquireLogic
from ._data_logic import (
    ADDataFactory,
    ADHDFDataLogic,
    ADMultipartDataLogic,
    ADWriterFactory,
    NDArrayDescription,
    TimeSeriesDataLogic,
)
from ._detector import AreaDetector, ContAcqDetector, plugins_with_proc
from ._io import (
    NDROIIO,
    ADBaseColorMode,
    ADBaseDataType,
    ADBaseIO,
    ADBloscCompressor,
    ADBloscShuffle,
    ADCompressMode,
    ADCompressor,
    ADFileWriteMode,
    ADImageMode,
    ADState,
    NDArrayBaseIO,
    NDCBFlushOnSoftTrgMode,
    NDCircularBuffIO,
    NDCodecIO,
    NDCodecStatus,
    NDFileHDF5Compression,
    NDFileHDF5IO,
    NDFileIO,
    NDParam,
    NDPluginBaseIO,
    NDPluginFileIO,
    NDProcessFilterCallbacks,
    NDProcessFilterType,
    NDProcessIO,
    NDROIStatIO,
    NDROIStatNIO,
    NDROIStatTSControl,
    NDStatsIO,
    NDStatsTimeSeriesIO,
    NDStatsTSAcquireMode,
)
from ._ndattribute import (
    NDAttributeDataType,
    NDAttributeParam,
    NDAttributePv,
    NDAttributePvDbrType,
    ndattributes_to_xml,
)
from ._plan_stubs import setup_ndattributes
from ._trigger_logic import (
    ADContAcqTriggerLogic,
    prepare_exposures,
    prepare_filter,
    trigger_info_from_driver,
)

__all__ = [
    # ADCore IOs
    "ADBaseDataType",
    "ADBaseColorMode",
    "NDArrayBaseIO",
    "ADImageMode",
    "ADState",
    "ADBaseIO",
    "NDPluginBaseIO",
    "NDROIIO",
    "NDParam",
    "NDStatsIO",
    "NDStatsTimeSeriesIO",
    "NDStatsTSAcquireMode",
    "NDROIStatNIO",
    "NDROIStatIO",
    "NDROIStatTSControl",
    "NDCBFlushOnSoftTrgMode",
    "NDCircularBuffIO",
    "ADFileWriteMode",
    "NDFileIO",
    "NDPluginFileIO",
    "NDFileHDF5Compression",
    "NDFileHDF5IO",
    "NDCodecIO",
    "NDCodecStatus",
    "NDProcessIO",
    "NDProcessFilterType",
    "NDProcessFilterCallbacks",
    "ADCompressor",
    "ADCompressMode",
    "ADBloscCompressor",
    "ADBloscShuffle",
    # TriggerLogic
    "prepare_exposures",
    "prepare_filter",
    "ADContAcqTriggerLogic",
    "trigger_info_from_driver",
    # AcquireLogic
    "ADAcquireLogic",
    "ADContAcqAcquireLogic",
    # DataLogic
    "NDArrayDescription",
    "ADHDFDataLogic",
    "ADMultipartDataLogic",
    "ADDataFactory",
    "ADWriterFactory",
    "TimeSeriesDataLogic",
    # Detector
    "AreaDetector",
    "ContAcqDetector",
    "plugins_with_proc",
    # NDAttributes
    "NDAttributeDataType",
    "NDAttributePvDbrType",
    "NDAttributePv",
    "NDAttributeParam",
    "ndattributes_to_xml",
    "setup_ndattributes",
]
