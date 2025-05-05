from cogeneration.type.str_enum import StrEnum


class DataTask(StrEnum):
    """task for training"""

    hallucination = "hallucination"
    inpainting = "inpainting"  # aka `scaffolding`


class InferenceTask(StrEnum):
    """task for inference"""

    unconditional = "unconditional"
    inpainting = "inpainting"  # aka `scaffolding`
    forward_folding = "forward_folding"
    inverse_folding = "inverse_folding"

    @staticmethod
    def from_data_task(task: DataTask) -> "InferenceTask":
        """Get InferenceTask corresponding to DataTask"""
        if task == DataTask.hallucination:
            return InferenceTask.unconditional
        elif task == DataTask.inpainting:
            return InferenceTask.inpainting
        else:
            # default
            return InferenceTask.unconditional

    @staticmethod
    def to_data_task(task: "InferenceTask") -> DataTask:
        """Get DataTask corresponding to InferenceTask"""
        if task == InferenceTask.unconditional:
            return DataTask.hallucination
        elif task == InferenceTask.inpainting:
            return DataTask.inpainting
        else:
            # default
            return DataTask.hallucination
