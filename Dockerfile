FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# FFmpeg enables keyframe probing and broader codec support.
# libgl1/libglib2 are common OpenCV runtime dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY framehunter ./framehunter

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

# Use /data for host-mounted inputs/outputs.
WORKDIR /data

ENTRYPOINT ["framehunter"]
