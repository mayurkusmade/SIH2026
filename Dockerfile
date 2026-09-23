# syntax=docker/dockerfile:1
#
# AnantaNetra / IBVAP edge node image.
#
# Two roles run from this one image, which is deliberate: the analytics must be
# byte-identical between an outpost and Sector HQ, so there is no second build to
# drift out of sync.
#
#   edge  - headless agent (run_edge_daemon.py): analytics + uplink, no browser
#   c2    - the operator dashboard (app.py) for a Sector HQ station
#
# Select with IBVAP_ROLE (see docker-compose.yml).
#
# Build:
#   docker build -t ibvap-edge:1.0 .
# Run an outpost:
#   docker run --rm -e IBVAP_TELEMETRY=mqtt -e IBVAP_MQTT_HOST=broker \
#     -v ibvap-a-data:/app/data ibvap-edge:1.0

FROM python:3.11-slim AS base

# OpenCV's shared-library prerequisites. Without these, `import cv2` fails at
# runtime with a cryptic libGL error - the single most common containerization
# failure for an OpenCV workload.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    IBVAP_DATA_DIR=/app/data \
    IBVAP_ROLE=edge

WORKDIR /app

# Dependency layer first, so application edits do not invalidate a 1 GB install.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY . .

# A container writes alerts.csv and the biometric index into the mounted volume,
# never into the image layer.
RUN mkdir -p /app/data \
    && useradd --create-home --shell /usr/sbin/nologin ibvap \
    && chown -R ibvap:ibvap /app
USER ibvap

# A field appliance must come up on its own. These are the demo-safe defaults:
# broker-free simulated link, local video sources, no tensor acceleration.
ENV IBVAP_TELEMETRY=simulated \
    IBVAP_LINK_PROFILE=SATELLITE \
    IBVAP_PACKET_BUDGET_KB=10

VOLUME ["/app/data"]

# Edge role is the default: an unattended outpost box. Override CMD for the C2 UI.
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,sys,urllib.request; \
role=os.environ.get('IBVAP_ROLE','edge'); \
url='http://127.0.0.1:8501/_stcore/health' if role=='c2' else None; \
sys.exit(0 if url is None else (0 if urllib.request.urlopen(url,timeout=3).status==200 else 1))"

CMD ["python", "run_edge_daemon.py"]
