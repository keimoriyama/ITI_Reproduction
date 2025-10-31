from typing import Literal, Optional

from pydantic import BaseModel, Field


class ITIConfig(BaseModel):
    model_name: str = Field(
        default="meta-llama/Llama-2-7b-chat-hf",
        description="model name for the experiment",
    )
    model_prefix: str = Field(default="", description="prefix for the result")
    dataset_name: str = Field(default="tqa_mc2", description="dataset name")
    activation_dataset: Optional[str] = Field(
        default=None,
        description="feature bank for calculating std along direction",
    )

    alpha: float = Field(15, description="alpha, intervention strength")
    num_fold: int = Field(2, description="number of folds")
    val_ratio: float = Field(
        0.2, description="ratio of validation set size to development set size"
    )
    use_center_of_mass: bool = Field(False, description="use center of mass direction")
    use_random_dir: bool = Field(False, description="use random direction")

    device: Literal["cpu", "cuda"] = Field("cpu", description="device")
    debug: bool = Field(False, description="debug mode")
    seed: int = Field(42, description="seed")
    judge_name: Optional[str] = Field(None)
    info_name: Optional[str] = Field(None)
    instruction_prompt: Optional[str] = Field(None)
