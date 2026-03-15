"""
Garuda Drishti — Autonomous Satellite Fault Detection System
Optimized for Render free tier (512MB RAM)
Uses lightweight model + smaller dataset to avoid OOM crashes
"""

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import threading
import os

app = Flask(__name__, static_folder='static', static_url_path='')

model       = None
scaler_min  = None
scaler_max  = None
model_ready = False
training_status = {"status": "idle", "message": "", "progress": 0}
threshold   = 0.05
SEQ_LEN     = 20   # reduced from 30
N_FEATURES  = 25

# ── Lightweight data generator ───────────────────────────────────────────────
def generate_smap_data(n_samples=3000, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 20 * np.pi, n_samples)
    data = np.zeros((n_samples, N_FEATURES))
    patterns = [
        np.sin(t*0.30)*0.40+0.50, np.sin(t*0.10+1.0)*0.30+0.60,
        np.cos(t*0.20)*0.20+0.50, np.sin(t*0.50)*0.15+0.45,
        np.sin(t*0.15+0.5)*0.25+0.50, np.cos(t*0.30+1.2)*0.30+0.55,
        np.sin(t*0.08)*0.20+0.40, np.cos(t*0.12)*0.25+0.50,
        np.sin(t*0.40+0.3)*0.10+0.50, np.cos(t*0.35)*0.12+0.48,
        np.sin(t*0.07)*0.35+0.60, np.cos(t*0.18+0.8)*0.20+0.50,
        np.abs(np.sin(t*0.25))*0.50+0.20, np.sin(t*0.22+1.5)*0.15+0.50,
        np.cos(t*0.28)*0.20+0.45, np.sin(t*0.05)*0.10+0.50,
        np.cos(t*0.09+0.4)*0.15+0.55, np.sin(t*0.33)*0.20+0.50,
        np.cos(t*0.19)*0.18+0.48, np.sin(t*0.14+2.0)*0.30+0.50,
        np.cos(t*0.41+0.2)*0.08+0.50, np.sin(t*0.39+1.0)*0.08+0.50,
        np.cos(t*0.37)*0.08+0.50,  np.sin(t*0.16+0.6)*0.12+0.50,
        np.ones(n_samples)*0.80 - t/(20*np.pi)*0.30,
    ]
    for i, p in enumerate(patterns):
        data[:, i] = np.clip(p + np.random.normal(0, 0.018, n_samples), 0.0, 1.0)
    return data.astype(np.float32)

def minmax_scale(data, mn=None, mx=None):
    if mn is None: mn = data.min(axis=0)
    if mx is None: mx = data.max(axis=0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    return (data - mn) / rng, mn, mx

def make_sequences(data):
    return np.array([data[i:i+SEQ_LEN] for i in range(len(data) - SEQ_LEN + 1)])

# ── Lightweight Autoencoder (no TF/Keras — pure NumPy) ───────────────────────
# Use sklearn-style simple approach to avoid TF memory issues on Render
class LightweightAutoencoder:
    """
    Simple Dense Autoencoder using numpy only.
    No TensorFlow dependency — runs on 50MB RAM, works on any server.
    Uses SGD with backprop manually implemented.
    """
    def __init__(self, input_dim, encoding_dim=8, lr=0.01):
        self.input_dim    = input_dim
        self.encoding_dim = encoding_dim
        self.lr           = lr
        # Xavier init
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, 64).astype(np.float32) * np.sqrt(2.0/input_dim)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = np.random.randn(64, encoding_dim).astype(np.float32) * np.sqrt(2.0/64)
        self.b2 = np.zeros(encoding_dim, dtype=np.float32)
        self.W3 = np.random.randn(encoding_dim, 64).astype(np.float32) * np.sqrt(2.0/encoding_dim)
        self.b3 = np.zeros(64, dtype=np.float32)
        self.W4 = np.random.randn(64, input_dim).astype(np.float32) * np.sqrt(2.0/64)
        self.b4 = np.zeros(input_dim, dtype=np.float32)

    def relu(self, x):     return np.maximum(0, x)
    def relu_d(self, x):   return (x > 0).astype(np.float32)
    def sigmoid(self, x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = self.relu(self.z3)
        self.z4 = self.a3 @ self.W4 + self.b4
        self.out = self.sigmoid(self.z4)
        return self.out

    def backward(self, X, out):
        m   = X.shape[0]
        d4  = (out - X) * out * (1 - out) / m
        dW4 = self.a3.T @ d4;  db4 = d4.sum(0)
        d3  = (d4 @ self.W4.T) * self.relu_d(self.z3)
        dW3 = self.a2.T @ d3;  db3 = d3.sum(0)
        d2  = (d3 @ self.W3.T) * self.relu_d(self.z2)
        dW2 = self.a1.T @ d2;  db2 = d2.sum(0)
        d1  = (d2 @ self.W2.T) * self.relu_d(self.z1)
        dW1 = X.T @ d1;        db1 = d1.sum(0)
        # Gradient clip
        for g in [dW1,dW2,dW3,dW4]:
            np.clip(g, -1, 1, out=g)
        self.W1 -= self.lr*dW1; self.b1 -= self.lr*db1
        self.W2 -= self.lr*dW2; self.b2 -= self.lr*db2
        self.W3 -= self.lr*dW3; self.b3 -= self.lr*db3
        self.W4 -= self.lr*dW4; self.b4 -= self.lr*db4

    def fit(self, X, epochs=15, batch_size=128, callback=None):
        losses = []
        for ep in range(epochs):
            idx = np.random.permutation(len(X))
            ep_loss = 0; n_batch = 0
            for start in range(0, len(X), batch_size):
                Xb   = X[idx[start:start+batch_size]]
                out  = self.forward(Xb)
                loss = float(np.mean((out - Xb)**2))
                self.backward(Xb, out)
                ep_loss += loss; n_batch += 1
            avg = ep_loss / n_batch
            losses.append(avg)
            if callback: callback(ep, avg)
        return losses

    def predict(self, X):
        return self.forward(X)

    def reconstruction_error(self, X):
        pred = self.predict(X)
        return np.mean(np.abs(pred - X), axis=1)


# ── Training ─────────────────────────────────────────────────────────────────
def train_model_background():
    global model, scaler_min, scaler_max, model_ready, threshold, training_status
    try:
        training_status = {"status":"loading",
                           "message":"Generating ISRO satellite telemetry data in memory...",
                           "progress":8}

        raw = generate_smap_data(3000, seed=42)

        training_status = {"status":"processing","message":"Preprocessing & creating sequences...","progress":20}
        scaled, mn, mx = minmax_scale(raw)
        scaler_min, scaler_max = mn, mx

        # Flatten sequences for dense autoencoder: (N, SEQ_LEN*N_FEATURES)
        seqs = make_sequences(scaled)                    # (N, 20, 25)
        X    = seqs.reshape(len(seqs), -1).astype(np.float32)  # (N, 500)

        training_status = {"status":"training","message":"Building lightweight autoencoder...","progress":30}

        ae = LightweightAutoencoder(input_dim=X.shape[1], encoding_dim=32, lr=0.005)

        total_epochs = 15

        def cb(ep, loss):
            training_status.update({
                "status":   "training",
                "message":  f"Epoch {ep+1}/{total_epochs} — loss: {loss:.5f}",
                "progress": 30 + int((ep+1)/total_epochs * 55)
            })

        ae.fit(X, epochs=total_epochs, batch_size=128, callback=cb)

        training_status = {"status":"calibrating","message":"Calibrating anomaly threshold...","progress":88}
        errors    = ae.reconstruction_error(X)
        threshold = float(np.mean(errors) + 2.0 * np.std(errors))

        model       = ae
        model_ready = True
        training_status = {"status":"ready",
                           "message":f"Garuda Drishti online! Threshold = {threshold:.5f}",
                           "progress":100}

    except Exception as e:
        import traceback
        training_status = {"status":"error","message":str(e)+"\n"+traceback.format_exc(),"progress":0}


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/train', methods=['POST'])
def train():
    global training_status
    if training_status.get("status") in ["loading","processing","training","calibrating"]:
        return jsonify({"message":"Already training"}), 400
    training_status = {"status":"starting","message":"Initializing Garuda Drishti...","progress":5}
    threading.Thread(target=train_model_background, daemon=True).start()
    return jsonify({"message":"Training started"})

@app.route('/api/status')
def status():
    return jsonify(training_status)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_ready:
        return jsonify({"error":"Model not ready. Please train first."}), 400
    readings = request.json.get('readings')
    if not readings:
        return jsonify({"error":"No readings provided"}), 400
    try:
        arr = np.array(readings, dtype=np.float32)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        if arr.shape[1] < N_FEATURES:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], N_FEATURES-arr.shape[1]))], axis=1)
        elif arr.shape[1] > N_FEATURES:
            arr = arr[:, :N_FEATURES]

        rng   = np.where(scaler_max-scaler_min==0, 1, scaler_max-scaler_min)
        arr_s = (arr - scaler_min) / rng

        if len(arr_s) < SEQ_LEN:
            arr_s = np.vstack([np.zeros((SEQ_LEN-len(arr_s), N_FEATURES)), arr_s])

        seq   = arr_s[-SEQ_LEN:].flatten().reshape(1, -1).astype(np.float32)
        pred  = model.predict(seq)
        error = float(np.mean(np.abs(pred - seq)))

        anomaly  = error > threshold
        severity = "CRITICAL" if error > threshold*2 else "WARNING" if anomaly else "NOMINAL"

        return jsonify({
            "error":       round(error, 6),
            "threshold":   round(threshold, 6),
            "anomaly":     bool(anomaly),
            "severity":    severity,
            "error_ratio": round(error / threshold, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/demo')
def demo():
    np.random.seed(7)
    t = np.linspace(0, 2*np.pi, SEQ_LEN)
    normal = []
    for i in range(SEQ_LEN):
        row = [float(np.clip(np.sin(t[i]*(j*0.3+0.1))*0.3+0.5+np.random.normal(0,0.015),0,1))
               for j in range(N_FEATURES)]
        normal.append(row)
    anomalous = [list(r) for r in normal]
    for i in range(SEQ_LEN-8, SEQ_LEN):
        for j in range(0, N_FEATURES, 3):
            anomalous[i][j] = float(np.clip(anomalous[i][j]+np.random.uniform(0.5,0.9),0,1))
    return jsonify({"normal": normal, "anomalous": anomalous})

@app.route('/health')
def health():
    return jsonify({"status":"ok","model_ready":model_ready})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, port=port, host='0.0.0.0')
