FROM nvidia/cuda:13.0.1-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
