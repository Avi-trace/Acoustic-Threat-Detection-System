"""
export_verilog_weights.py — Auto-generate Verilog weight parameters
Handles any MLP architecture (16→32→16→3 or other).
Reads trained sklearn MLPClassifier and outputs Q8 fixed-point weight
declarations ready to paste into a Verilog module.
"""
import numpy as np
import joblib
import os

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\avyuk\datasets"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALE      = 256   # Q8: multiply by 2^8

# ── Load Model ────────────────────────────────────────────────────────────────
model = joblib.load(os.path.join(DATA_DIR, "mlp_model.pkl"))

n_layers = len(model.coefs_)
print(f"Model architecture:")
print(f"  Input  : {model.coefs_[0].shape[0]} features")
for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
    print(f"  Layer {i+1}: {W.shape[0]} → {W.shape[1]}  (W shape {W.shape}, b shape {b.shape})")
print()

# ── Quantize all layers to Q8 ─────────────────────────────────────────────────
W_q = [np.round(W * SCALE).astype(int) for W in model.coefs_]
b_q = [np.round(b * SCALE).astype(int) for b in model.intercepts_]

# ── Check for overflow (16-bit signed: -32768 to 32767) ──────────────────────
for i, (W, b) in enumerate(zip(W_q, b_q)):
    wmax = np.max(np.abs(W))
    bmax = np.max(np.abs(b))
    if wmax > 32767 or bmax > 32767:
        print(f"  WARNING: Layer {i+1} has values exceeding 16-bit signed range!")
        print(f"           Max weight={wmax}, Max bias={bmax}")
        print(f"           Consider reducing SCALE or using [31:0] parameters.")
    else:
        print(f"  Layer {i+1}: max_weight={wmax}, max_bias={bmax} — fits in 16 bits OK")

print()

# ── Generate Verilog ──────────────────────────────────────────────────────────
lines = []
lines.append("// " + "="*77)
lines.append("// AUTO-GENERATED WEIGHTS — Do not edit manually!")
lines.append(f"// Source     : {os.path.join(DATA_DIR, 'mlp_model.pkl')}")
lines.append(f"// Q8 format  : scale = {SCALE} (multiply float weights by {SCALE}, round to int)")
lines.append(f"// Architecture: {model.coefs_[0].shape[0]} inputs → " +
             " → ".join(str(W.shape[1]) for W in model.coefs_) + " outputs")
lines.append("// " + "="*77)
lines.append("")

for layer_idx, (W, b) in enumerate(zip(W_q, b_q)):
    n_in, n_out = W.shape
    layer_num   = layer_idx + 1

    lines.append(f"// {'='*77}")
    lines.append(f"// LAYER {layer_num} WEIGHTS — W{layer_num}[input_idx][neuron_idx]"
                 f" — shape ({n_in},{n_out}) — Q8")
    lines.append(f"// {'='*77}")

    for j in range(n_out):
        parts  = [f"W{layer_num}_{i}_{j} = {W[i, j]}" for i in range(n_in)]
        prefix = f"// Neuron {j}\nlocalparam signed [15:0] "
        # wrap long lines at 4 params each
        chunks = [parts[k:k+4] for k in range(0, len(parts), 4)]
        first_line = prefix + ",  ".join(chunks[0])
        if len(chunks) == 1:
            lines.append(first_line + ";")
        else:
            lines.append(first_line + ",")
            for chunk in chunks[1:-1]:
                lines.append("                          " + ",  ".join(chunk) + ",")
            lines.append("                          " + ",  ".join(chunks[-1]) + ";")

    lines.append("")
    lines.append(f"// LAYER {layer_num} BIASES — b{layer_num}[neuron_idx] — Q8")
    # wrap biases 4 per line
    bias_parts = [f"B{layer_num}_{j} = {b[j]}" for j in range(n_out)]
    for k in range(0, len(bias_parts), 4):
        chunk = bias_parts[k:k+4]
        lines.append("localparam signed [15:0] " + ",  ".join(chunk) + ";")
    lines.append("")

verilog_code = "\n".join(lines)

# ── Print ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print(" VERILOG WEIGHT PARAMETERS (Q8 Fixed-Point)")
print("=" * 60)
print(verilog_code)

# ── Save ──────────────────────────────────────────────────────────────────────
output_path = os.path.join(OUTPUT_DIR, "verilog_weights.v")
with open(output_path, 'w') as f:
    f.write(verilog_code + "\n")

print("=" * 60)
print(f" Saved to : {output_path}")
print(f" Layers   : {n_layers}")
print(f" Note     : Update your Verilog top module to match the new")
print(f"            architecture ({model.coefs_[0].shape[0]}→"
      + "→".join(str(W.shape[1]) for W in model.coefs_) + ")")
print("=" * 60)