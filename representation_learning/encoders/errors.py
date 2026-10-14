"""Typed errors. Names follow docs/architecture.md section 7 (Phase 1)."""


class EncoderError(Exception):
    """Base class for all visual-encoder errors."""


class InvalidInputError(EncoderError):
    """Image/crop/bbox is missing, empty, wrongly typed or otherwise unusable."""


class UnsupportedEvidenceError(InvalidInputError):
    """Evidence is outside the Phase 2 scope (e.g. person category) or violates a
    Phase 1 invariant (e.g. annotated video as a source)."""


class EncoderLoadError(EncoderError):
    """Weights/config missing or unreadable, or the requested device is unavailable."""


class EncoderIntegrityError(EncoderLoadError):
    """Weights SHA-256 missing/mismatched, or checkpoint preprocessing disagrees
    with the preprocessing this module applies."""


class EmbeddingValidationError(EncoderError):
    """Model output violates the declared spec (wrong dim, NaN/Inf, zero norm)."""


class IncompatibleSpaceError(EncoderError):
    """Two embeddings from different spaces were compared."""
