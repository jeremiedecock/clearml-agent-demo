FROM python:3.12-slim

# Git is required to log git repositories in ClearML ; ffmpeg may be used to render videos (e.g. for RL agents)
RUN apt update -y && apt install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
