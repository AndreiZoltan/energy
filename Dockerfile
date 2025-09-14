FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Chisinau

RUN apt-get update && apt-get install -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy your module and CLI
COPY energy_pred/ /app/energy_pred/
COPY main.py /app/main.py

# Keep module importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["python", "-u", "/app/main.py"]
# Users will pass args like: --data-dir /data/GicaHack --model-path /models/model.cbm
