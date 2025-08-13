from iti_reproduction.cli import setup_cli
from iti_reproduction.get_activation import get_activation
from iti_reproduction.intervention import intervene
from iti_reproduction.scheme import ITIConfig


def activation(cfg: ITIConfig):
    print("hello from function")
    get_activation(cfg)


def intervention(cfg: ITIConfig):
    intervene(cfg)


if __name__ == "__main__":
    cfg = setup_cli(
        ITIConfig, commands={"activation": activation, "intervention": intervention}
    )
