"""
dashboard.py — Real-Time Acoustic Threat Classification Web Dashboard
Flask + Flask-SocketIO backend with real-time mic classification.
Run: python dashboard.py
Open: http://localhost:5000
"""
import numpy as np
import librosa
import joblib
import sounddevice as sd
import threading
import time
import os
from flask import Flask, render_template
from flask_socketio import SocketIO

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\avyuk\datasets"
SAMPLE_RATE = 22050
DURATION    = 2.0       # seconds per classification window
N_MFCC      = 4
INTERVAL    = 2.0       # seconds between classifications

LABELS = {0: "BACKGROUND", 1: "GUNSHOT", 2: "DRONE"}
THREAT = {0: False, 1: True, 2: True}
COLORS = {0: "#2ecc71", 1: "#e74c3c", 2: "#f39c12"}  # green, red, yellow

# ── Load Model ────────────────────────────────────────────────────────────────
print("Loading model and scaler...")
model  = joblib.load(os.path.join(DATA_DIR, "mlp_model.pkl"))
scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
print("  Model loaded successfully!")

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'acoustic-threat-classifier'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── State ─────────────────────────────────────────────────────────────────────
is_listening = False
listen_thread = None

@app.route('/')
def index():
    return render_template('dashboard.html')

def classify_audio(audio_data, sr):
    """Extract features and classify an audio chunk."""
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=N_MFCC)
        features = np.mean(mfccs, axis=1).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        proba = model.predict_proba(features_scaled)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class]) * 100
        
        return {
            'class': pred_class,
            'label': LABELS[pred_class],
            'threat': THREAT[pred_class],
            'color': COLORS[pred_class],
            'confidence': round(confidence, 1),
            'probabilities': {
                'background': round(float(proba[0]) * 100, 1),
                'gunshot': round(float(proba[1]) * 100, 1),
                'drone': round(float(proba[2]) * 100, 1)
            },
            'timestamp': time.strftime('%H:%M:%S')
        }
    except Exception as e:
        print(f"  Classification error: {e}")
        return None

def listen_loop():
    """Background thread: capture mic audio and classify continuously."""
    global is_listening
    print("  Mic listener started!")
    
    while is_listening:
        try:
            # Record audio from default mic
            n_samples = int(SAMPLE_RATE * DURATION)
            audio = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished
            audio = audio.flatten()
            
            # Send waveform data (subsampled for visualization)
            subsample = 200
            waveform = audio[::max(1, len(audio)//subsample)].tolist()
            socketio.emit('waveform', {'data': waveform})
            
            # Classify
            result = classify_audio(audio, SAMPLE_RATE)
            if result:
                socketio.emit('classification', result)
                status = "🚨 THREAT" if result['threat'] else "✅ SAFE"
                print(f"  [{result['timestamp']}] {result['label']} ({result['confidence']:.0f}%) — {status}")
            
        except Exception as e:
            print(f"  Mic error: {e}")
            socketio.emit('error', {'message': str(e)})
            time.sleep(1)
    
    print("  Mic listener stopped.")

@socketio.on('start_listening')
def handle_start(data=None):
    global is_listening, listen_thread
    if not is_listening:
        is_listening = True
        listen_thread = threading.Thread(target=listen_loop, daemon=True)
        listen_thread.start()
        socketio.emit('status', {'listening': True})
        print("  → Listening started via WebSocket")

@socketio.on('stop_listening')
def handle_stop(data=None):
    global is_listening
    is_listening = False
    socketio.emit('status', {'listening': False})
    print("  → Listening stopped via WebSocket")

@socketio.on('connect')
def handle_connect():
    print("  → Client connected")
    socketio.emit('status', {'listening': is_listening})

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print(" 🎯 Acoustic Threat Classifier — Web Dashboard")
    print("="*55)
    print(f"  Model:      {DATA_DIR}\\mlp_model.pkl")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Window:      {DURATION}s")
    print(f"  Dashboard:   http://localhost:5000")
    print("="*55 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
