from pydantic import BaseModel, Field


class ActivationConfig(BaseModel):
    model_name: str = Field(default="meta-llama/Llama-2-7b-chat-hf")
    model_prefix: str = Field(default="")
    dataset_name: str = Field(default="tqa_mc2")
    device: str = "cpu"
    debug: bool = False
