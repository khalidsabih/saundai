# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS runtime

ARG SAUND_BUILD_COMMIT=
ARG SAUND_BUILD_TIME_UTC=
ARG SAUND_IMAGE_TAG=local

ENV SAUND_BUILD_COMMIT=${SAUND_BUILD_COMMIT} \
    SAUND_BUILD_TIME_UTC=${SAUND_BUILD_TIME_UTC} \
    SAUND_IMAGE_TAG=${SAUND_IMAGE_TAG} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/djai
COPY pyproject.toml constraints.txt README.md LICENSE VERSION ./
COPY src ./src
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -c constraints.txt .

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "djai"]
CMD ["--help"]

FROM runtime AS stems
WORKDIR /opt/djai
RUN python -m pip install -c constraints.txt ".[stems]"
WORKDIR /workspace

FROM runtime AS brain
WORKDIR /opt/djai
RUN python -m pip install -c constraints.txt ".[stems,neural]"
WORKDIR /workspace

FROM brain AS hosted
WORKDIR /opt/djai
RUN python -m pip install -c constraints.txt ".[hosted]"
COPY config/model-registry.json ./default-config/model-registry.json
COPY docker/hosted-entrypoint.sh /usr/local/bin/saund-hosted-entrypoint
RUN chmod 0755 /usr/local/bin/saund-hosted-entrypoint
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/saund-hosted-entrypoint"]

FROM runtime AS cue
WORKDIR /opt/djai
# The research CUE-DETR extra intentionally pins the public reference stack.
RUN python -m pip install ".[cue]"
WORKDIR /workspace

FROM runtime AS test
WORKDIR /opt/djai
COPY tests ./tests
RUN python -m pip install -c constraints.txt ".[dev]"
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q"]
