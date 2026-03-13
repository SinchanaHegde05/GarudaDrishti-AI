# GARUDA DRISHTI— Autonomous Satellite Fault Detection System

## Architecture Overview

```
User Browser  ←→  Flask API (app.py)  ←→  TensorFlow LSTM Autoencoder
                         ↕
              NASA SMAP Dataset (fetched online)
```

---

## Step-by-Step Implementation

### Step 1 — Dataset (Online, No Local Download)
- **Source**: NASA SMAP (Soil Moisture Active Passive) P-1 channel
- **URL**: `github.com/khundman/telemanom` (NASA's official anomaly benchmark)
- **Method**: `fetch_npy_online()` — downloads `.npy` into memory using `io.BytesIO`, never touches disk
- **Shape**: ~8000 timesteps × 25 sensor channels

### Step 2 — Preprocessing
- **MinMax Scaling**: Each of 25 channels scaled to [0,1] range
- **Sequence Creation**: Rolling windows of length 30 → shape (N, 30, 25)
- **Train/Test Split**: Uses NASA's pre-split train/test files

### Step 3 — LSTM Autoencoder Model
```
Input (30, 25)
  → LSTM(64, return_sequences=True)
  → LSTM(32, return_sequences=False)        ← Encoder
  → Dense(16, relu)                         ← Bottleneck
  → RepeatVector(30)
  → LSTM(32, return_sequences=True)
  → LSTM(64, return_sequences=True)         ← Decoder
  → TimeDistributed(Dense(25))              ← Reconstruction
Output (30, 25)
```
- **Loss**: MSE (Mean Squared Error)
- **Optimizer**: Adam
- **Epochs**: 15, Batch size: 64

### Step 4 — Anomaly Detection Logic
- After training, compute reconstruction error on ALL training sequences
- **Threshold** = mean(train_errors) + 2 × std(train_errors)
- At inference: error > threshold → ANOMALY
- **Severity**:
  - error < threshold → NOMINAL ✓
  - threshold < error ≤ 2×threshold → WARNING ⚠
  - error > 2×threshold → CRITICAL ✕

### Step 5 — Flask API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the frontend |
| `/api/train` | POST | Start background training thread |
| `/api/status` | GET | Poll training progress |
| `/api/predict` | POST | Analyze sensor readings |
| `/api/demo` | GET | Get sample normal/anomalous data |

### Step 6 — Interactive Frontend
- **Space-themed dark UI** with real-time animations
- 25 editable sensor input channels
- Timestep slider for multi-frame telemetry
- Load normal/anomalous demo samples
- Real-time scan history chart (canvas)
- System log with timestamps
- Result: verdict + reconstruction error bar + metrics

---

## How to Run

### Prerequisites
```bash
pip install flask tensorflow numpy requests
```

### Start Server
```bash
python app.py
```

### Open Browser
```
http://localhost:5000
```

### Usage Flow
1. Click **"Initialize & Train Neural Network"**
2. Wait ~2-3 minutes for training (progress shown live)
3. Click **"Load Normal Sample"** or **"Load Anomaly Sample"** (or enter custom values)
4. Click **"Analyze Telemetry"**
5. View result: NOMINAL / WARNING / CRITICAL

---

## Project Structure
```
orion/
├── app.py              # Flask backend + ML model
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── static/
    └── index.html      # Frontend (single-file, no build step)
```

---

## Key Design Decisions

**Why LSTM Autoencoder?**
- Satellites produce time-series data — LSTM captures temporal dependencies
- Autoencoders learn "normal" patterns unsupervised
- High reconstruction error = anomalous behavior

**Why no dataset download?**
- Uses `io.BytesIO` to stream `.npy` files directly into NumPy arrays
- Zero disk writes, works in any environment

**Why threshold = mean + 2σ?**
- Statistically, ~95% of normal data falls within 2 standard deviations
- Robust to noise while sensitive to true anomalies
