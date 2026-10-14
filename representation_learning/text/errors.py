"""Text-side errors (Phase 3, G2).

All error types of the visual side are reused unchanged (`representation_learning.encoders.errors`,
docs/architecture.md section 7); only the one condition that is specific to text gets its own
subclass. It derives from `InvalidInputError`, so callers that already handle that type keep working.
"""

from __future__ import annotations

from ..encoders.errors import (EmbeddingValidationError, EncoderError, EncoderIntegrityError,
                               EncoderLoadError, InvalidInputError)

__all__ = ["EmbeddingValidationError", "EncoderError", "EncoderIntegrityError", "EncoderLoadError",
           "InvalidInputError", "TextTooLongError"]


class TextTooLongError(InvalidInputError):
    """The model input (role prefix + text + special tokens) exceeds the encoder's token limit.
    Text is never truncated to fit; the item is rejected."""

    def __init__(self, message: str, *, token_count: int, limit: int):
        super().__init__(message)
        self.token_count = token_count
        self.limit = limit
