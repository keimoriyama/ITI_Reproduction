from pydantic import Field
from pydantic_settings import BaseSettings

# , SettingsConfigDict


class Configs(BaseSettings):
    model_name: str = Field(default="meta-llama/Llama-2-7b-chat-hf")
    model_prefix: str = Field(default="")
    dataset_name: str = Field(default="tqa_mc2")


def main():
    print("Hello from iti-reproduction!")
    print(Configs().model_dump())


if __name__ == "__main__":
    main()
