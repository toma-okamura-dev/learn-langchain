# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 依存定義をコピーして先にインストール
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 仮想環境をPATHに通す
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# アプリ本体をコピー
COPY main.py .

CMD ["python", "main.py"]