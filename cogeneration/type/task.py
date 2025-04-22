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

    @staticmethod
    def from_data_task(task: DataTaskEnum) -> "InferenceTaskEnum":
        """Get InferenceTaskEnum corresponding to DataTaskEnum"""
        if task == DataTaskEnum.hallucination:
            return InferenceTaskEnum.unconditional
        elif task == DataTaskEnum.inpainting:
            return InferenceTaskEnum.inpainting
        else:
            # default
            return InferenceTaskEnum.unconditional

    @staticmethod
    def to_data_task(task: "InferenceTaskEnum") -> DataTaskEnum:
        """Get DataTaskEnum corresponding to InferenceTaskEnum"""
        if task == InferenceTaskEnum.unconditional:
            return DataTaskEnum.hallucination
        elif task == InferenceTaskEnum.inpainting:
            return DataTaskEnum.inpainting
        else:
            # default
            return DataTaskEnum.hallucination
