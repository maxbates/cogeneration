from enum import Enum


class PositionalEmbeddingMethod(Enum):
    """Positional embedding methods."""

    rotary = "rotary"
    sine_cosine = "sine_cosine"
