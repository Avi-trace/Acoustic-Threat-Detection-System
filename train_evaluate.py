"""
train_evaluate.py — Training & Evaluation Pipeline (16-feature MLP)
Architecture: 16 inputs → 32 hidden → 16 hidden → 3 outputs
Outputs: confusion_matrix.png, roc_curves.png, classification report
"""
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import os

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR     = r"C:\Users\avyuk\datasets"
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
LABELS       = {0: "Background", 1: "Gunshot", 2: "Drone"}
CLASS_NAMES  = list(LABELS.values())
RANDOM_STATE = 42
N_FEATURES   = 16

FEATURE_NAMES = [
    "MFCC_1",  "MFCC_2",  "MFCC_3",  "MFCC_4",  "MFCC_5",
    "MFCC_6",  "MFCC_7",  "MFCC_8",  "MFCC_9",  "MFCC_10",
    "Spec_Centroid", "Spec_Rolloff", "ZCR",
    "E_Low(0-500Hz)", "E_Mid(500-2kHz)", "E_High(2-8kHz)"
]

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

assert X.shape[1] == N_FEATURES, \
    f"Expected {N_FEATURES} features, got {X.shape[1]}. Re-run build_dataset.py first."

print(f"  Total samples : {len(X)}")
print(f"  Feature shape : {X.shape}")
print(f"  Classes       : {np.unique(y)} → {[LABELS[c] for c in np.unique(y)]}")
print()

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ── Scale Features ────────────────────────────────────────────────────────────
scaler     = StandardScaler()
X_train_s  = scaler.fit_transform(X_train)
X_test_s   = scaler.transform(X_test)

# ── Train MLP ─────────────────────────────────────────────────────────────────
# Architecture: 16 → 32 → 16 → 3
# Larger than original (4→8→3) to handle richer 16-dim feature space
print("\nTraining MLPClassifier (16 → 32 → 16 → 3)...")
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    verbose=False
)
model.fit(X_train_s, y_train)

train_acc = model.score(X_train_s, y_train)
test_acc  = model.score(X_test_s,  y_test)
print(f"  Train accuracy : {train_acc*100:.1f}%")
print(f"  Test  accuracy : {test_acc*100:.1f}%")
print(f"  Layers         : {[N_FEATURES] + list(model.hidden_layer_sizes) + [3]}")
print(f"  Iterations     : {model.n_iter_}")

# ── Save Model ────────────────────────────────────────────────────────────────
joblib.dump(model,  os.path.join(DATA_DIR, "mlp_model.pkl"))
joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.pkl"))
print(f"\n  Model + scaler saved to: {DATA_DIR}")

# ── Classification Report ─────────────────────────────────────────────────────
y_pred = model.predict(X_test_s)
print("\n" + "="*55)
print(" CLASSIFICATION REPORT")
print("="*55)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
print("Generating confusion matrix...")
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    linewidths=0.5, linecolor='gray',
    annot_kws={"size": 16, "weight": "bold"},
    ax=ax
)
ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label',      fontsize=13, fontweight='bold')
ax.set_title('MLP Acoustic Threat Classifier — Confusion Matrix (16 features)',
             fontsize=13, fontweight='bold')
ax.text(0.5, -0.15, f"Test Accuracy: {test_acc*100:.1f}%",
        transform=ax.transAxes, ha='center', fontsize=12,
        style='italic', color='darkgreen')
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {cm_path}")

# ── ROC Curves ────────────────────────────────────────────────────────────────
print("Generating ROC curves...")
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score    = model.predict_proba(X_test_s)
colors     = ['#2ecc71', '#e74c3c', '#f39c12']

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2.5,
            label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random (AUC = 0.500)')
ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate',  fontsize=13, fontweight='bold')
ax.set_title('MLP Acoustic Threat Classifier — ROC Curves (16 features)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {roc_path}")

print("\n" + "="*55)
print(" DONE — All outputs generated!")
print("="*55)