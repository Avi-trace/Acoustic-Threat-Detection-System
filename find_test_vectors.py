import numpy as np
import joblib

X = np.load(r"C:\Users\avyuk\datasets\X.npy")
y = np.load(r"C:\Users\avyuk\datasets\y.npy")
scaler = joblib.load(r"C:\Users\avyuk\datasets\scaler.pkl")
model  = joblib.load(r"C:\Users\avyuk\datasets\mlp_model.pkl")

SCALE = 256
W1 = np.round(model.coefs_[0] * SCALE).astype(np.int64)
b1 = np.round(model.intercepts_[0] * SCALE).astype(np.int64)
W2 = np.round(model.coefs_[1] * SCALE).astype(np.int64)
b2 = np.round(model.intercepts_[1] * SCALE).astype(np.int64)

def hw_predict(x_scaled):
    x = np.round(x_scaled * SCALE).astype(np.int64)
    net1 = (x @ W1 >> 8) + b1
    h    = np.maximum(0, net1)
    out  = (h @ W2 >> 8) + b2
    return np.argmax(out), np.round(x_scaled * SCALE).astype(int)

LABELS = {0:"Background", 1:"Gunshot", 2:"Drone"}
print("=== HARDWARE-VERIFIED TEST VECTORS ===\n")

for cls in [0, 1, 2]:
    idxs = np.where(y == cls)[0]
    for idx in idxs:
        raw    = X[idx]
        scaled = scaler.transform(raw.reshape(1,-1))[0]
        hw_cls, q8 = hw_predict(scaled)
        if hw_cls == cls:
            print(f"Class {cls} ({LABELS[cls]}): x0={q8[0]}, x1={q8[1]}, x2={q8[2]}, x3={q8[3]}")
            break
