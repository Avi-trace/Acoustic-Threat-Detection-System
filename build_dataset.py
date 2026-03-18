"""
build_dataset.py — Feature Extraction Pipeline (16 features)
Features:
  [0-9]  10 MFCCs
  [10]   Spectral Centroid (mean)
  [11]   Spectral Rolloff (mean)
  [12]   Zero Crossing Rate (mean)
  [13]   Low-band energy  (0–500 Hz)   — drone rotor fundamental
  [14]   Mid-band energy  (500–2000 Hz) — drone harmonics
  [15]   High-band energy (2000–8000 Hz) — gunshot transient energy
"""
import os
import numpy as np
import librosa

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR  = r"C:\Users\avyuk\datasets"
SR        = 22050
DURATION  = 2.0
N_MFCC    = 10
N_FEATURES = 16

# Frequency band boundaries (Hz) for drone/gunshot discrimination
BAND_LOW  = (0,    500)
BAND_MID  = (500,  2000)
BAND_HIGH = (2000, 8000)

def band_energy(y, sr, f_low, f_high):
    """RMS energy of the signal filtered to [f_low, f_high] Hz."""
    stft   = np.abs(librosa.stft(y))
    freqs  = librosa.fft_frequencies(sr=sr)
    mask   = (freqs >= f_low) & (freqs < f_high)
    band   = stft[mask, :]
    return float(np.mean(band ** 2)) if band.size > 0 else 0.0

def extract_features(file_path):
    """
    Returns a 16-element numpy array or None on failure.
    """
    try:
        y, sr = librosa.load(file_path, duration=DURATION, sr=SR, mono=True)

        # ── 10 MFCCs ────────────────────────────────────────────────────────
        mfccs   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        f_mfcc  = np.mean(mfccs, axis=1)                          # shape (10,)

        # ── Spectral Centroid ────────────────────────────────────────────────
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        f_cent   = np.mean(centroid)                               # scalar

        # ── Spectral Rolloff ─────────────────────────────────────────────────
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        f_roll   = np.mean(rolloff)                                # scalar

        # ── Zero Crossing Rate ───────────────────────────────────────────────
        zcr      = librosa.feature.zero_crossing_rate(y)
        f_zcr    = np.mean(zcr)                                    # scalar

        # ── Custom Frequency Band Energies ───────────────────────────────────
        e_low    = band_energy(y, sr, *BAND_LOW)
        e_mid    = band_energy(y, sr, *BAND_MID)
        e_high   = band_energy(y, sr, *BAND_HIGH)

        features = np.concatenate([
            f_mfcc,
            [f_cent, f_roll, f_zcr, e_low, e_mid, e_high]
        ])                                                         # shape (16,)

        assert features.shape == (N_FEATURES,), f"Bad shape: {features.shape}"
        return features

    except Exception as e:
        print(f"  [SKIP] {file_path}: {e}")
        return None

# ── Dataset assembly ──────────────────────────────────────────────────────────
features_list = []
labels_list   = []

# --- GUNSHOT FILES (label = 1) ---
gunshot_folders = [
    "AK-12", "AK-47", "IMI Desert Eagle", "M4", "M16",
    "M249", "MG-42", "MP5", "Zastava M92"
]
for folder in gunshot_folders:
    path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(path):
        print(f"  [WARN] Missing folder: {path}")
        continue
    for fname in os.listdir(path):
        if fname.lower().endswith(".wav"):
            f = extract_features(os.path.join(path, fname))
            if f is not None:
                features_list.append(f)
                labels_list.append(1)

print(f"Gunshots loaded : {labels_list.count(1)}")

# --- DRONE FILES (label = 2) ---
drone_path = os.path.join(DATA_DIR, "Binary_Drone_Audio", "yes_drone")
for fname in os.listdir(drone_path):
    if fname.lower().endswith(".wav"):
        f = extract_features(os.path.join(drone_path, fname))
        if f is not None:
            features_list.append(f)
            labels_list.append(2)

print(f"Drones loaded   : {labels_list.count(2)}")

# --- BACKGROUND FILES (label = 0) ---
bg_path = os.path.join(DATA_DIR, "Binary_Drone_Audio", "unknown")
for fname in os.listdir(bg_path):
    if fname.lower().endswith(".wav"):
        f = extract_features(os.path.join(bg_path, fname))
        if f is not None:
            features_list.append(f)
            labels_list.append(0)

print(f"Background loaded: {labels_list.count(0)}")

# ── Save ──────────────────────────────────────────────────────────────────────
X = np.array(features_list, dtype=np.float32)
y = np.array(labels_list,   dtype=np.int32)

print(f"\nTotal samples : {len(X)}")
print(f"Feature shape : {X.shape}   ← should be (N, 16)")
print(f"Classes       : {np.unique(y)}")

np.save(os.path.join(DATA_DIR, "X.npy"), X)
np.save(os.path.join(DATA_DIR, "y.npy"), y)
print("Dataset saved!")