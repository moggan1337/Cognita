"""Model compression methods for federated learning."""

from cognita.compression.compressor import GradientCompressor
from cognita.compression.quantization import Quantizer
from cognita.compression.sparsification import TopKCompressor, RandomKCompressor
from cognita.compression.coding import EntropyCoder

__all__ = [
    "GradientCompressor",
    "Quantizer",
    "TopKCompressor",
    "RandomKCompressor",
    "EntropyCoder",
]
