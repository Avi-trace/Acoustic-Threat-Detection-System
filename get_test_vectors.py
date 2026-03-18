"""
get_test_vectors.py
Generates hardware-verified test vectors for mlp_tb_pipelined.v
Picks one real sample per class, runs Q8 fixed-point math in Python
to confirm what the Verilog should output, then prints ready-to-paste
apply_and_check() calls.
"""
import numpy as np
import joblib, os

DATASET_DIR = r"C:\Users\avyuk\datasets"
SCALE = 256

model  = joblib.load(os.path.join(DATASET_DIR, "mlp_model.pkl"))
scaler = joblib.load(os.path.join(DATASET_DIR, "scaler.pkl"))
X      = np.load(os.path.join(DATASET_DIR, "X.npy"))
y      = np.load(os.path.join(DATASET_DIR, "y.npy"))

CLASS_NAMES = {0: "Background", 1: "GUNSHOT   ", 2: "DRONE     "}
VERILOG_CLASS = {0: "2'b00", 1: "2'b01", 2: "2'b10"}

def q8_mlp(x_raw, model, scaler):
    """Run exact Q8 integer forward pass matching the Verilog pipeline."""
    x_scaled = scaler.transform(x_raw.reshape(1, -1))[0]
    x_q8 = np.round(x_scaled * SCALE).astype(int)

    h = x_q8
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        b_q8 = np.round(b * SCALE).astype(int)
        net = np.zeros(W.shape[1], dtype=np.int64)
        for n in range(W.shape[1]):
            W_q8 = np.round(W[:, n] * SCALE).astype(int)
            mac = int(np.sum(h.astype(np.int64) * W_q8.astype(np.int64)))
            net[n] = (mac >> 8) + b_q8[n]
        # ReLU on all layers except last
        if i < len(model.coefs_) - 1:
            h = np.maximum(0, net)
        else:
            h = net

    winner = int(np.argmax(h))
    sorted_scores = np.sort(h)[::-1]
    conf_delta = int(sorted_scores[0] - sorted_scores[1])
    return x_q8, winner, conf_delta, h

print("Finding one real sample per class...")
print()

found = {}
for idx in range(len(X)):
    cls = int(y[idx])
    if cls not in found:
        x_q8, pred, conf, scores = q8_mlp(X[idx], model, scaler)
        if pred == cls:  # only use samples the model gets right
            found[cls] = (x_q8, pred, conf, scores)
    if len(found) == 3:
        break

print("=" * 70)
print("COPY THESE INTO mlp_tb_pipelined.v — replace the 3 apply_and_check calls")
print("=" * 70)
print()

for cls in [0, 1, 2]:
    if cls not in found:
        print(f"WARNING: No correctly-classified sample found for class {cls}")
        continue
    x_q8, pred, conf, scores = found[cls]
    name = CLASS_NAMES[cls]
    vclass = VERILOG_CLASS[cls]

    # Check for overflow
    overflow = any(abs(v) > 32767 for v in x_q8)
    if overflow:
        print(f"WARNING: Some features overflow 16-bit for class {cls}!")

    print(f"        // ── TEST: {name} ─────────────────────────────")
    print(f"        apply_and_check(")
    print(f"            16'sd{x_q8[0]}, 16'sd{x_q8[1]}, 16'sd{x_q8[2]}, 16'sd{x_q8[3]},   // x0-x3")
    print(f"            16'sd{x_q8[4]}, 16'sd{x_q8[5]}, 16'sd{x_q8[6]}, 16'sd{x_q8[7]},   // x4-x7")
    print(f"            16'sd{x_q8[8]}, 16'sd{x_q8[9]}, 16'sd{x_q8[10]}, 16'sd{x_q8[11]},  // x8-x11")
    print(f"            16'sd{x_q8[12]}, 16'sd{x_q8[13]}, 16'sd{x_q8[14]}, 16'sd{x_q8[15]}, // x12-x15")
    print(f"            {vclass}, \"{name}\", {cls+1}")
    print(f"        );")
    print()
    print(f"        // Python Q8 prediction: class={pred} ({name.strip()})  conf_delta={conf}")
    print(f"        // Raw scores: bg={scores[0]}  gun={scores[1]}  drone={scores[2]}")
    print()

print("=" * 70)
print("Also replace the throughput test block with these same x values.")
print("=" * 70)