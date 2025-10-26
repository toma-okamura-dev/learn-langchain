# uv + Python 同梱の軽量ベースイメージ（高速・再現性◎）
FROM ghcr.io/astral-sh/uv:python3.11-bookworm

ENV UV_SYSTEM_PYTHON=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 依存関係レイヤをキャッシュさせるために先に定義ファイルだけコピー
COPY pyproject.toml uv.lock ./

# uv.lock に固定されたバージョンで依存を system にインストール
RUN uv sync --frozen --no-dev


# アプリ本体をコピー（秘密情報はイメージに含めない）
COPY main.py .

# 実行コマンド
CMD ["python", "main.py"]