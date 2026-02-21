FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry with extended timeout
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 poetry

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies via Poetry with retries and extended timeout
RUN poetry config virtualenvs.create false && \
    poetry config installer.max-workers 10 && \
    poetry install --no-interaction --no-ansi --no-root || \
    (echo "Retry 1..." && poetry install --no-interaction --no-ansi --no-root) || \
    (echo "Retry 2..." && poetry install --no-interaction --no-ansi --no-root)

# Copy application code
COPY . .

EXPOSE 8005

CMD ["python", "main.py"]
