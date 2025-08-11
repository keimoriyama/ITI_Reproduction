from iti_reproduction.cli import setup_cli
from iti_reproduction.get_activation import get_activation
from iti_reproduction.scheme import ActivationConfig


def activation(cfg: ActivationConfig):
    print("hello from function")
    get_activation(cfg)


if __name__ == "__main__":
    cfg = setup_cli(ActivationConfig, commands={"activation": activation})
