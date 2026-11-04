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
)
from ._detector import AreaDetector, ContAcqDetector
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
    NDPluginBaseIO,
    NDPluginFileIO,
    NDProcessFilterCallbacks,
    NDProcessFilterType,
    NDProcessIO,
    NDROIStatIO,
    NDROIStatNIO,
    NDROIStatTSControl,
    NDStatistic,
    NDStatsIO,
    NDStatsTimeSeriesIO,
    NDStatsTSAcquireMode,
)
from ._ndattribute import (
    NDAttributeDataType,
    NDAttributeParam,
    NDAttributePv,
    NDAttributePvDbrType,
    NDAttributeSource,
    ndattributes_to_xml,
)
from ._plan_stubs import setup_ndattributes
from ._stats_time_series import ROIStatTimeSeriesDataLogic, StatsTimeSeriesDataLogic
from ._trigger_logic import (
    ADContAcqTriggerLogic,
    exposures_per_collection_from_filter,
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
    "NDStatistic",
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
    "exposures_per_collection_from_filter",
    # AcquireLogic
    "ADAcquireLogic",
    "ADContAcqAcquireLogic",
    # DataLogic
    "NDArrayDescription",
    "ADHDFDataLogic",
    "ADMultipartDataLogic",
    "ADDataFactory",
    "ADWriterFactory",
    "StatsTimeSeriesDataLogic",
    "ROIStatTimeSeriesDataLogic",
    # Detector
    "AreaDetector",
    "ContAcqDetector",
    # NDAttributes
    "NDAttributeDataType",
    "NDAttributePvDbrType",
    "NDAttributePv",
    "NDAttributeParam",
    "NDAttributeSource",
    "ndattributes_to_xml",
    "setup_ndattributes",
]
