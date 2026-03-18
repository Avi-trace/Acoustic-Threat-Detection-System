import librosa
import numpy as np
import joblib
import sys

# Load model + scaler
model  = joblib.load(r"C:\Users\avyuk\datasets\mlp_model.pkl")
scaler = joblib.load(r"C:\Users\avyuk\datasets\scaler.pkl")

LABELS = {0: "BACKGROUND", 1: "GUNSHOT", 2: "DRONE"}
ALERT  = {0: "✅ NO THREAT", 1: "🚨 THREAT DETECTED", 2: "⚠️  THREAT DETECTED"}

def classify(file_path):
    audio, sr = librosa.load(file_path, duration=2.0, sr=22050)
    mfccs     = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=4)
    features  = np.mean(mfccs, axis=1).reshape(1, -1)
    features  = scaler.transform(features)

    proba      = model.predict_proba(features)[0]
    pred_class = np.argmax(proba)
    confidence = proba[pred_class] * 100

    print(f"\n{'='*40}")
    print(f"  File     : {file_path}")
    print(f"  Result   : {LABELS[pred_class]}")
    print(f"  {ALERT[pred_class]}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"{'='*40}\n")

    if confidence < 70:
        print("  ⚠️  Low confidence — signal may be degraded or distant\n")

# Run on a file passed as argument, or test file
file = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\avyuk\datasets\AK-47\1 (1).wav"
classify(file)
