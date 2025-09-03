# README

This repository is a reimplementation of [honest_llama](https://github.com/likenneth/honest_llama).

# Table of Contents

## Installation

Please use [Docker](https://www.docker.com/) or [uv](https://docs.astral.sh/uv/) for the environmental setup.

If you use uv, please install above link and run this command.
```sh
uv sync
```

If you use Docker, you can setup environment by docker-compose.

```sh
docker compose up -d
```

## Run Experiments

### Get Activations 

```
uv run main.py activation --config config/config.yaml 
```

### Interventions

```
uv run main.py intervention --config config/config.yaml 
```

## Results

TBD
