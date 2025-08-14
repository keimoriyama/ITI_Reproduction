FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ARG USER_ID=1022 
ARG GROUP_ID=1022 
ARG DEBIAN_FRONTEND=noninteractive 
ARG PYTHON_VERSION=3.11

ARG CACHE_DIR="/app/.cache/"
ARG APP_DIR="/app"
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Tokyo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    DEB_PYTHON_INSTALL_LAYOUT=deb \
    APP_DIR=${APP_DIR} \
    PYTHONPATH=${APP_DIR}/src:$PYTHONPATH \
    WANDB_API_KEY='' \
    WANDB_CACHE_DIR=$CACHE_DIR/wandb \
    WANDB_DATA_DIR=$CACHE_DIR/data \
    HF_HOME=$CACHE_DIR/transformer \
    HF_DATASETS_CACHE=$CACHE_DIR/datasets \
    UV_CACHE_DIR=$CACHE_DIR/uv/ \
    UV_NO_CACHE=1 \
    UV_INSTALL_DIR=/usr/local/bin \
    HYDRA_FULL_ERROR=1
    #    UV_PROJECT_ENVIRONMENT=/app/

RUN groupadd -g ${GROUP_ID} appgroup \
 && useradd -u ${USER_ID} -g appgroup -m appuser \
 && chown -R appuser:appgroup /home/appuser \
 && mkdir -p /app \
 && chown -R appuser:appgroup /app \
 && mkdir -p ./.cache \
 && chmod -R 777 ./.cache \
 && chown -R appuser:appgroup ./.cache
 
RUN apt update \
 && apt install -y --no-install-recommends \
    curl \
    wget \
    software-properties-common \
    git


ADD https://astral.sh/uv/install.sh ./uv-installer.sh

# Run the installer then remove it
RUN sh ./uv-installer.sh && rm ./uv-installer.sh

USER appuser

WORKDIR ${APP_DIR}

COPY --chmod=777 README.md pyproject.toml uv.lock .python-version src/ ./

RUN uv sync

ENV PATH="/app/.venv/bin:/usr/local/bin:$PATH"

RUN echo $PATH

CMD ["/bin/bash"]
