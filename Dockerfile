FROM --platform=linux/amd64 python:3.12 AS base
ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

RUN pip install uv

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

# Copy source code
COPY . /app
RUN uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "clustering_api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
