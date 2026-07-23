FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY popeye_config.env ./.env
COPY packages/core/ ./packages/core/
COPY packages/server/ ./packages/server/

RUN uv sync --frozen --no-dev --package talltable-server --package talltable

RUN useradd --no-create-home --uid 1000 talltable
USER talltable

ENV HOME=/tmp
WORKDIR /tmp

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "talltable_server.server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=192.168.0.0/16"]
