"""
Autonomous Satellite Fault Detection System
Backend: Flask + TensorFlow LSTM Autoencoder
Dataset: Synthetic NASA SMAP-style satellite telemetry (in-memory, no download)
"""

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import threading

app = Flask(__name__, static_folder='static')

model       = None
scaler_min  = None
scaler_max  = None
model_ready = False
training_status = {"status": "idle", "message": "", "progress": 0}
threshold   = 0.05
SEQ_LEN     = 30
N_FEATURES  = 25

def generate_smap_data(n_samples=8000, seed=42):
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
        np.cos(t*0.37)*0.08+0.50, np.sin(t*0.16+0.6)*0.12+0.50,
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

def build_autoencoder():
    inp = keras.Input(shape=(SEQ_LEN, N_FEATURES))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.RepeatVector(SEQ_LEN)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(N_FEATURES))(x)
    return keras.Model(inp, out)

def train_model_background():
    global model, scaler_min, scaler_max, model_ready, threshold, training_status
    try:
        training_status = {"status":"loading","message":"Generating NASA SMAP-style satellite telemetry in memory...","progress":10}
        train_raw = generate_smap_data(7000, seed=42)

        training_status = {"status":"processing","message":"Preprocessing sensor sequences...","progress":22}
        train_scaled, mn, mx = minmax_scale(train_raw)
        scaler_min, scaler_max = mn, mx
        X_train = make_sequences(train_scaled)

        training_status = {"status":"training","message":"Building LSTM Autoencoder...","progress":35}
        ae = build_autoencoder()
        ae.compile(optimizer='adam', loss='mse')

        def on_epoch(epoch, logs):
            training_status.update({
                "status":"training",
                "message":f"Epoch {epoch+1}/15 — loss: {logs['loss']:.5f}  val: {logs.get('val_loss',0):.5f}",
                "progress": 35 + int((epoch+1)/15 * 45)
            })

        ae.fit(X_train, X_train, epochs=15, batch_size=64, validation_split=0.1,
               verbose=0, callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=on_epoch)])

        training_status = {"status":"calibrating","message":"Calibrating anomaly threshold...","progress":88}
        X_pred = ae.predict(X_train, batch_size=64, verbose=0)
        errors = np.mean(np.abs(X_pred - X_train), axis=(1,2))
        threshold = float(np.mean(errors) + 2.0 * np.std(errors))

        model = ae
        model_ready = True
        training_status = {"status":"ready","message":f"Model ready! Threshold = {threshold:.5f}","progress":100}

    except Exception as e:
        import traceback
        training_status = {"status":"error","message":str(e),"progress":0}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/train', methods=['POST'])
def train():
    global training_status
    if training_status.get("status") in ["loading","processing","training","calibrating"]:
        return jsonify({"message":"Already training"}), 400
    training_status = {"status":"starting","message":"Initializing...","progress":5}
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
        if arr.ndim == 1: arr = arr.reshape(1,-1)
        if arr.shape[1] < N_FEATURES:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], N_FEATURES-arr.shape[1]))], axis=1)
        elif arr.shape[1] > N_FEATURES:
            arr = arr[:, :N_FEATURES]
        rng = np.where(scaler_max-scaler_min==0, 1, scaler_max-scaler_min)
        arr_s = (arr - scaler_min) / rng
        if len(arr_s) < SEQ_LEN:
            arr_s = np.vstack([np.zeros((SEQ_LEN-len(arr_s), N_FEATURES)), arr_s])
        seq  = arr_s[-SEQ_LEN:].reshape(1, SEQ_LEN, N_FEATURES)
        pred = model.predict(seq, verbose=0)
        error = float(np.mean(np.abs(pred - seq)))
        anomaly  = error > threshold
        severity = "CRITICAL" if error > threshold*2 else "WARNING" if anomaly else "NOMINAL"
        return jsonify({"error":round(error,6),"threshold":round(threshold,6),
                        "anomaly":bool(anomaly),"severity":severity,
                        "error_ratio":round(error/threshold,3)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

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
    return jsonify({"normal":normal,"anomalous":anomalous})

if __name__ == '__main__':
    app.run(debug=False, port=10000, host='0.0.0.0')