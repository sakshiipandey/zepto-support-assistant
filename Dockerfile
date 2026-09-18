FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Compile Python bytecode for faster startup
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install exact pinned dependencies from uv.lock
RUN uv sync --frozen --no-install-project

# Copy project files
COPY . .

EXPOSE 5000

# Start Flask application using uv
CMD ["uv", "run", "python", "app.py"]