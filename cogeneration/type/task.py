from cogeneration.type.str_enum import StrEnum


class DataTaskEnum(StrEnum):
    """task for training"""

    hallucination = "hallucination"
    inpainting = "inpainting"  # aka `scaffolding`


class InferenceTaskEnum(StrEnum):
    """task for inference"""

    unconditional = "unconditional"
    inpainting = "inpainting"  # aka `scaffolding`
    forward_folding = "forward_folding"
    inverse_folding = "inverse_folding"
