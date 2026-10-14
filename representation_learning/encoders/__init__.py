"""Visual encoders (Phase 2). Heavy dependencies are imported lazily."""

from .base import VisualEncoder
from .clip import ClipEncoder
from .datatypes import (BackendInfo, EncodeFailure, EncoderConfig, EncoderSpec,
                        PreprocessSpec, VisualRepresentation)
from .dinov2 import Dinov2Encoder
from .errors import (EmbeddingValidationError, EncoderError, EncoderIntegrityError,
                     EncoderLoadError, IncompatibleSpaceError, InvalidInputError,
                     UnsupportedEvidenceError)

__all__ = [
    "VisualEncoder", "Dinov2Encoder", "ClipEncoder", "EncoderConfig", "EncoderSpec",
    "PreprocessSpec", "BackendInfo", "VisualRepresentation", "EncodeFailure",
    "EncoderError", "InvalidInputError", "UnsupportedEvidenceError", "EncoderLoadError",
    "EncoderIntegrityError", "EmbeddingValidationError", "IncompatibleSpaceError",
]
