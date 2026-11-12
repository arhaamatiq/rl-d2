FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    git curl ca-certificates libegl-dev libgl1 libglvnd0 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    MUJOCO_GL=egl

RUN uv python install 3.13

WORKDIR /app

# Dependency layer: only rebuilt when the lockfile or pyproject change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --extra cu128

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cu128

# Bake the frozen sim-to-sim model into the image (regenerates bar/sim2sim/g1_flat.mjb
# and checks its pinned sha256). CPU only; takes a few minutes.
RUN .venv/bin/python -m babysteps.eval.export_bundle --verify --device cpu

ENTRYPOINT ["bash", "scripts/reproduce.sh"]
CMD ["headline"]
