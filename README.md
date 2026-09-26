# 🛡️ IBVAP — Intelligent Border Video Analytics Platform
### Smart India Hackathon 2026 | Problem Statement: SIH26187
**Organization:** Ministry of Home Affairs — Sashastra Seema Bal (SSB), Police II Division  
**Category:** Software | **Theme:** Smart Automation

---

## 📌 Executive Summary

Conventional CCTV infrastructure deployed at Border Out Posts (BOPs), border check posts, and ingress roads primarily provides passive video recording and live monitoring, demanding continuous human vigilance. Proprietary smart camera hardware and commercial FRS/ANPR equipment are prohibitively expensive and difficult to maintain across remote, rugged border sectors.

**IBVAP (Intelligent Border Video Analytics Platform)** is an AI-driven, software-defined surveillance platform that transforms standard IP-based CCTV infrastructure into an active, intelligent surveillance network. It operates 100% locally on edge hardware (CPU/GPU) without requiring specialized smart cameras or cloud connectivity.

---

## 🚀 Key Capabilities (Delivered MVP)

1. **Human Detection & Multi-Object Tracking:**
   - Stock YOLOv8 nano inference filtered to persons.
   - Real-time Centroid Tracking with persistent track IDs and 30-frame trajectory history.
2. **Virtual Fence & Perimeter Intrusion Detection:**
   - Digital tripwire and restricted perimeter boundary crossing detection.
   - Segment-intersection trajectory analysis triggering instant `INTRUSION_ALERT` alarms with exact geographic coordinates.
3. **Cascaded Automatic Number Plate Recognition (ANPR):**
   - Cascaded pipeline triggered **strictly on detected vehicle classes** (`car`, `truck`, `bus`, `motorcycle`).
   - License plate localization via YOLOv8 plate detector + contrast-normalized EasyOCR.
   - **Temporal Best-per-Track Smoothing:** Aggregates multi-frame plate readings, locking in the highest confidence read as the vehicle approaches.
   - **Operational Integrity Fallback:** Distant, occluded, or unreadable plates are automatically flagged as `FLAGGED_FOR_MANUAL_REVIEW` rather than outputting erroneous readings.
4. **Multi-Channel Defense Command Dashboard (Streamlit):**
   - **Channel 1 — BOP Sector 4 Perimeter:** Real-time pedestrian tracking and boundary breach alerting.
   - **Channel 2 — Checkpost Charlie Ingress:** Real-time vehicular classification, license plate detection, and ANPR.
   - **Channel 3 — Pre-recorded Insurance Demo:** Pre-rendered annotated stream ensuring zero-lag presentations.
   - **Historical Incident Register:** Real-time event log with interactive filtering and one-click CSV audit export.
   - **Threaded Multi-Camera Grid:** Every sector gets its own grab thread (I/O) plus inference thread (model), so the dashboard repaints on a timer instead of blocking on YOLO. A 2x2 mode shows all sectors at once.
   - **Camera Health Telemetry:** Per-camera ingest state (`LIVE` / `STALLED` / `RECONNECTING`), input FPS, analysis latency, skipped frames, reconnect count and watchdog recovery actions.
5. **GIS Command Map & Containerized Field Deployment:**
   - **Offline Tactical Picture:** Outposts, camera coverage cones, incident clusters, area of operations, relay distances, a graticule and a truthful scale bar, rendered as one self-contained inline SVG — **no basemap tiles, no CDN, no API key, no network access of any kind**, so it still renders at an air-gapped outpost with the satellite link down.
   - **Coverage-Gap Awareness:** Any incident falling outside every camera cone is flagged as a perimeter surveillance gap rather than quietly plotted as if the sector were watched.
   - **Interception Planning:** Ranked responders with distance, bearing/compass and an ETA — explicitly labelled a **straight-line lower bound**, because with no road topology nothing more honest can be claimed.
   - **GeoJSON Export (RFC 7946):** Outposts, coverage polygons and incidents export with correct `[lon, lat]` axis order for QGIS or any Sector HQ staff map.
   - **Containerized Deployment:** A single image serving two roles — a **headless edge agent** (`run_edge_daemon.py`; no browser, no display at all) and the Sector HQ dashboard — orchestrated by `docker-compose.yml` with a Mosquitto broker and three outposts on modelled field links. Provisioned entirely through `IBVAP_*` environment variables, running as a non-root user with a HEALTHCHECK.
   - **Liveness Beacons:** Every outpost emits a periodic heartbeat, so "quiet sector" and "dead outpost" are distinguishable at HQ — silence is exactly what a successful intrusion looks like.
   - **One Vision Implementation:** The dashboard and the headless agent execute the *same* `modules/pipeline.py`; only the presence of a UI differs, so an alert can never be "dashboard-only".

6. **Live RTSP/ONVIF Ingestion & Multi-Camera Threading:**
   - **Legacy CCTV Retrofit:** Decodes real `rtsp://` / `rtsps://` / ONVIF device streams, RTMP/UDP, USB capture devices and local files, so existing IP cameras become analytics sources with zero hardware change.
   - **Field-Link Hardening:** RTSP is forced over TCP (UDP shreds packets on marginal links), with open/read timeouts and a depth-1 capture buffer so latency can never accumulate.
   - **Depth-1 Drop-Old Buffering:** The analysis stage always processes the NEWEST frame and discards what it missed — on a live border feed, a stale frame is worse than a skipped one.
   - **Self-Healing Cameras:** Dial failures and mid-stream drops reconnect with exponential backoff; a feed that goes silent without erroring is detected by a watchdog and force-reconnected.
   - **Bounded Edge CPU:** A shared permit pool caps simultaneous model passes, so a 4-camera grid cannot oversubscribe a 15W edge box.
   - **Truthful Attribution:** Every alert carries the `camera_id` + `node_id` of the sector that produced it, even with five sectors running at once.

7. **Biometric Facial Recognition & Identity Verification (FRS):**
   - **Real 1:N Biometric Matching:** Pluggable stack — RetinaFace/YuNet/Haar detection with ArcFace (MobileFaceNet, insightface or ONNX runtime) embeddings, matched against a **local** SQLite + numpy vector index. Fully air-gapped; no face image or embedding leaves the outpost.
   - **Exact Search at Roster Scale:** Measured **~2 ms worst case** for an exact cosine sweep over **10,000 identities x 512-d** on one CPU core — inside the <15 ms claim in the pitch, with no lossy ANN index in the path.
   - **Temporal Identity Smoothing:** An identity is promoted only after cross-frame agreement (or one high-confidence read), so a single blurred frame cannot flip a border guard into a suspect. Unconfirmed matches surface as `REVIEW` for a human, never auto-authorized.
   - **Honest Capability Reporting:** The UI and every alert state the mode in use — `BIOMETRIC`, `DEGRADED_NON_BIOMETRIC`, or `UNAVAILABLE`. A non-biometric appearance descriptor can **never** authorize a person unless an operator explicitly opts in, and then it is labelled DEGRADED.
   - **Roster Separability & Calibration:** `capacity_report()` measures the headroom between the match threshold and the observed impostor-similarity tail; `calibrate()` derives a threshold from roster statistics and can only ever *raise* the biometric gate.
   - **Identity Provenance in the Audit Trail:** Every alert records `ID:BIOMETRIC`, `ID:BIOMETRIC_UNCONFIRMED`, `ID:SIMULATED_DEMO_ID` or `ID:NONE`, and a confirmed watchlist match raises a `CRITICAL` `WATCHLIST_HIT`.

8. **Low-Bandwidth Telemetry Uplink (Sector HQ Sync):**
   - **Store-and-Forward Alerts:** every detection leaves the outpost as ONE structured packet (alert metadata + compressed evidence crop), capped at a hard **10 KB byte budget**. Raw video never crosses the link.
   - **Automatic Snapshot Degradation:** the evidence crop is shrunk (resolution first, then JPEG quality) until the whole packet fits; below the metadata floor the alert ships as a ~430 B metadata-only warning instead of overflowing the link.
   - **Field Link Modelling:** `FIBER` / `4G` / `SATELLITE` / `DEGRADED_SATCOM` profiles impose real latency and packet loss. A total outage buffers telemetry locally and retransmits in **FIFO order with exponential backoff** — zero loss across a blackout.
   - **Optional Real MQTT Publish:** `paho-mqtt` when a broker is reachable, with automatic fallback to the broker-free link simulator. `CRITICAL` alerts are routed to a priority topic.
   - **Operational Expiry:** buffered telemetry older than 15 minutes is dead-lettered rather than delivered as stale noise; a bounded buffer sheds the oldest packets, never the newest.

10. **Suspicious-Behaviour Analytics (Pose):**
   - **Conduct, Not Just Crossings:** Fence-climb attempts, falls / person down, low crawls under the tripwire, loitering, suspicious running, held-arm signalling and group convergence — none of which a tripwire can see.
   - **Scale-Independent Calibration:** Every threshold is expressed in units of the person's own body height (body-heights/second, fraction-of-height), so one calibration holds for a figure 4 m from the camera and one 40 m away. Pixel thresholds are why naive behaviour analytics collapse on a long-range perimeter camera.
   - **Temporal Confirmation:** A behaviour must persist across a time window before it is reported, so one noisy keypoint frame cannot raise an alert; per-track cooldowns stop a five-minute loiter from producing three hundred alerts.
   - **Precision Guards (tested):** Stooping at the waist is not a crawl, a moving crawl is not a fall, and one physical event never produces a chain of overlapping alerts (a fall supersedes the crawl that precedes it).
   - **Honesty Guard:** With no pose backend loaded the engine reports `UNAVAILABLE` and infers **nothing at all** — it never fabricates a behaviour. When active, every alert is provenance-stamped `POSE_HEURISTIC` and is never written into the audit trail as a classified fact. There is **no liveness / anti-spoofing**, stated as a known limitation rather than implied away.
   - **Operator Sensitivity Presets:** Conservative / Balanced / Aggressive — because a border post that cries wolf gets switched off by its own operators, which is the real failure mode of behaviour analytics.

11. **SSB Operational Scaling Roadmap (Phase 6):**
   - Thermal / Low-Light Night Vision (KAIST/FLIR multispectral sensor integration).
   - Ruggedized Edge Box Deployment (NVIDIA Jetson AGX / Orin Nano @ 15W, TensorRT INT8).
   - Face **liveness / anti-spoofing** (print-replay and mask detection) before FRS authorizations are trusted at an unattended gate.
   - Multi-Camera Spatial Re-Identification (Cross-camera OSNet appearance matching).
   - Long-range face capture (PTZ auto-zoom on perimeter approach) to raise usable face resolution at 50 m+.

---

## 🏗️ Project Architecture

```
SIH26187/
├── app.py                       # Streamlit Multi-Channel Surveillance Station
├── run_edge_daemon.py           # Headless BOP edge agent (no browser) - same pipeline as app.py
├── run_cli.py                   # Standalone CLI Detection Pipeline Runner
├── Dockerfile                   # One image, two roles: `edge` agent or `c2` dashboard
├── docker-compose.yml           # Field stack: Mosquitto broker + Sector HQ + 3 headless outposts
├── .dockerignore                # Keeps local logs/rosters/media out of the image
├── deploy/
│   └── mosquitto.conf           # Lab broker config (with the production security checklist)
├── test_phase2.py               # Centroid Tracker + Virtual Fence Verifier
├── test_phase3.py               # Cascaded ANPR & Throughput Benchmark
├── test_phase4.py               # Low-Bandwidth Telemetry Link Verifier (budget, offline queue, expiry)
├── test_phase5.py               # Live Ingest Grid Verifier (RTSP tuning, reconnect, watchdog, concurrency)
├── test_phase6.py               # FRS Verifier (1:N accuracy, false-accept rate, smoothing, calibration)
├── test_phase7.py               # GIS Verifier (geodesy, coverage, cluster, offline render, packaging)
├── test_phase8.py               # Behaviour Verifier (pose geometry, precision guards, pipeline wiring)
├── frs_index.sqlite             # Local biometric roster index (generated at runtime, gitignored)
├── render_backup_video.py       # Hackathon Insurance Pre-render Script
├── requirements.txt             # Project Python Dependencies
├── .gitignore                   # Clean Git tracking configuration
├── models/
│   ├── yolov8n.pt               # Pretrained YOLOv8 Nano COCO weights
│   └── licensePlateDetector.pt  # Trained YOLOv8 License Plate Detector
├── modules/
│   ├── __init__.py
│   ├── detector.py              # YOLOv8 Person & Vehicle Detection Module
│   ├── tracker.py               # Centroid Tracker with Trajectory History
│   ├── fence.py                 # Virtual Fence Line-Crossing Intrusion Engine (Directional)
│   ├── anpr.py                  # Cascaded Plate Detector + EasyOCR + Cache
│   ├── enhancer.py              # CLAHE Low-Light / Fog Restoration
│   ├── watchlist.py             # Authorized Personnel & Vehicle Roster (Offline Match)
│   ├── nodes.py                 # Edge Node / Camera Registry (geo placement + video source)
│   ├── ingest.py                # RTSP/ONVIF Live Ingestion: per-camera grab + analysis threads
│   ├── frs.py                   # Biometric FRS: detection + ArcFace embeddings + local 1:N index
│   ├── telemetry.py             # Low-Bandwidth Store-and-Forward Uplink (MQTT / Simulated Link)
│   └── logger.py                # In-memory Event Logger & CSV Exporter
├── sample_videos/
│   ├── bop_perimeter.mp4        # Channel 1: Sector A Long-Range Border Perimeter
│   ├── pedestrian_crossing.mp4  # Channel 2: Sector B Tactical Human Line Crossing (vedio-sih26.mp4)
│   ├── checkpost_traffic.mp4    # Channel 3: Sector C Vehicle Checkpoint & ANPR
│   └── backup_annotated_run.mp4 # Channel 4: Pre-rendered Annotated Backup Run
└── vedio-sih26.mp4               # High-density pedestrian line crossing surveillance feed
```

---

## ⚡ Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/mayurkusmade/SIH2026.git
cd SIH2026

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Weights (auto-loaded if present)

| File | Purpose | Size | Source |
|---|---|---|---|
| `models/yolov8n.pt` | person/vehicle detection | 6.5 MB | in repo |
| `models/licensePlateDetector.pt` | ANPR plate detector | 22 MB | in repo |
| `models/yolov8n-pose.pt` | behaviour analytics (pose) | 6.8 MB | `python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"` then move into `models/` |
| `models/face_detection_yunet.onnx` | FRS face detector | 230 KB | OpenCV zoo `face_detection_yunet_2023mar.onnx` |
| `models/arcface_w600k_r50.onnx` | FRS 512-d biometric embeddings | 174 MB | insightface `buffalo_l` pack (`w600k_r50.onnx`) |

All five loaded ⇒ the FRS banner reports **BIOMETRIC** and pose analytics reports **ACTIVE**. Missing ones degrade honestly (`DEGRADED_NON_BIOMETRIC` / `UNAVAILABLE`) — nothing fails silently.

### 3. Launch Command Center (Streamlit)
```bash
streamlit run app.py
```
*Open http://localhost:8501 in your browser to view all 4 surveillance channels with interactive virtual fence calibration.*

### 4. Headless / CLI Verification
```bash
# Test Core Multi-Channel Detection Pipeline
python run_cli.py

# Test Virtual Fence Intrusion on Sector B (vedio-sih26.mp4)
python test_phase2.py --video vedio-sih26.mp4 --frames 220

# Test Virtual Fence Intrusion across ALL sectors (Sector A + Sector B)
python test_phase2.py --all

# Test Cascaded ANPR & Vehicle Throughput
python test_phase3.py

# Test Low-Bandwidth Telemetry Uplink (byte budget, store-and-forward, expiry)
python test_phase4.py

# Test Live Ingest Grid (RTSP tuning, self-healing cameras, concurrency, attribution)
python test_phase5.py

# Test Facial Recognition (1:N accuracy, false-accept rate, smoothing, calibration)
python test_phase6.py

# Test the GIS command map (geodesy, coverage cones, response ranking, offline render)
python test_phase7.py

# Test behaviour analytics (pose geometry, precision guards, pipeline wiring)
python test_phase8.py

# Test throughput & UI-payload contracts
python test_phase9.py
```

### 5. Headless Edge Agent (no browser)
```bash
# Validate the deployment wiring on any machine - no camera, no model weights needed
python run_edge_daemon.py --dry-run --duration 10

# Run one outpost unattended (writes alerts.csv, publishes telemetry)
python run_edge_daemon.py --channels "Channel 1" --telemetry mqtt --mqtt-host localhost
```

### 6. Containerized Field Deployment
```bash
# Broker + Sector HQ dashboard + three unattended outposts
docker compose up --build

# Sector HQ command station:  http://localhost:8501
# MQTT broker:                localhost:1883
```
Each service is provisioned purely by environment variables (`IBVAP_NODE_ID`,
`IBVAP_CHANNELS`, `IBVAP_TELEMETRY`, `IBVAP_LINK_PROFILE`, `IBVAP_PACKET_BUDGET_KB`, ...),
so a new outpost needs no code change. Model weights are mounted read-only
(`./models:/app/models:ro`) rather than baked into the image, and to point a sector
at a real camera you only edit `video_source` in `modules/nodes.py` to its
RTSP/ONVIF URL.

---

## 📊 Evaluation & Verification Summary

| Feature | Benchmark Metric | Result |
| :--- | :--- | :--- |
| **Detection Engine** | YOLOv8n on CPU | 8.5 FPS (720p), 6.2 FPS (478p), 5.0 FPS (1080p) |
| **Intrusion Detection (Sector A)** | 1080p Tripwire | 7 verified breach events (Inbound & Outbound) |
| **Intrusion Detection (Sector B)** | Tactical Tripwire (`vedio-sih26.mp4`) | 3-5 verified line crossing events detected & logged |
| **ANPR Engine** | EasyOCR on vehicle crops | Verified read (`K433ZR` @ 91.8% confidence) |
| **Telemetry Uplink** | Packet byte budget (54 automated checks) | Every packet ≤ 10 KB (avg ~8.0 KB incl. JPEG evidence crop); metadata-only floor ~430 B |
| **Telemetry Uplink** | Total link outage | 0 packets lost; full FIFO backlog delivered on link recovery |
| **Live Ingest Grid** | 4-camera threaded run (76 automated checks) | Ingest + inference concurrent; UI state reads stay under ~10 ms while 4 models run |
| **Live Ingest Grid** | Camera failure handling | Dial failure and mid-stream drop both self-heal; silent stall caught by watchdog |
| **FRS (Biometrics)** | 1:N search, 10,000 identities x 512-d (87 checks) | Worst case **2.2 ms** exact cosine search; 50/50 lookups correct |
| **FRS (Biometrics)** | 1:N accuracy, 300 identities, noisy probes | **100%** top-1 accuracy at 0.96 mean similarity |
| **FRS (Biometrics)** | False-accept rate, crowded 64-d space | 3.2% at the default 0.45 threshold -> flagged TIGHT, calibration reduces it to **0%** with recall preserved |
| **GIS Command Map** | Geodesy, coverage cones, response ranking (123 checks) | Round-trip distance/bearing within 0.2°, 10 km at 40 km/h → 15 min ETA, coverage gaps flagged |
| **GIS Command Map** | Air-gap guarantee | Rendered map fetches **nothing** — zero tile/CDN/API references, inline SVG only |
| **GIS Command Map** | GeoJSON export | Valid RFC 7946 `[lon, lat]` features; polygon rings explicitly closed |
| **Behaviour Analytics** | Synthetic COCO-17 scenarios (102 checks) | Walking raises **0** alerts; stooping ≠ crawl; moving crawl ≠ fall; fall supersedes crawl |
| **Behaviour Analytics** | Climb rule specificity | Fires only with real fence proximity + rising hips; silent with no fence calibration |
| **Behaviour Analytics** | No-backend honesty | `UNAVAILABLE` mode emits **0** events — nothing is ever inferred without a model |
| **Pipeline Integration** | Dashboard + edge agent share one pipeline | Behaviour + fence events travel together; a crashing pose backend cannot stop the video |
| **Packaging** | Compose ↔ code env contract | Every `IBVAP_*` variable in `docker-compose.yml` is read by the application |
| **Low-Confidence Handling**| Blurry / distant plates | Flagged for manual review (0% false positives) |

---
*Built for the Smart India Hackathon 2026.*
