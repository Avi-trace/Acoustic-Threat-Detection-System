import librosa
import numpy as np

# generate a fake audio signal (we'll replace with real data tomorrow)
fake_audio = np.random.randn(22050)  # 1 second of fake audio
sample_rate = 22050

# extract 4 MFCC features
mfccs = librosa.feature.mfcc(y=fake_audio, sr=sample_rate, n_mfcc=4)

# average across time to get 4 single numbers
features = np.mean(mfccs, axis=1)

print("Features extracted:", features)
print("Shape:", features.shape)