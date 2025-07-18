import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
INPUT_PATH = r"C:\Users\DELL\Downloads\final_noise_audio.wav"       # <-- change this
OUTPUT_PATH = r"E:\filtered_output.wav"
TARGET_FREQS = np.array([49.99306508,50.00539383,219.59962397,229.19138837,279.93650696,                                                       
287.42005579,314.95014563,314.96247438,334.13367443,441.33212101,
529.43334001,529.44566876,529.4579975,733.63436022,856.87250536,
1106.50495461,1112.08987656,1112.1022053,1112.11453405,1140.63092358,
1182.99249488,1208.91984774,1238.274592,1238.28692074,1335.72100048,
1373.57025074,1452.38792399,1452.40025274,1471.5221378,1471.53446655])

# Frequency tolerance (in Hz) — to catch neighboring bins
TOLERANCE = 5.0

# ----------------------------
# LOAD AUDIO
# ----------------------------
y, sr = librosa.load(INPUT_PATH, sr=None)

# ----------------------------
# STFT
# ----------------------------
n_fft = 4096
hop_length = n_fft // 4
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
mag, phase = np.abs(D), np.angle(D)

# ----------------------------
# FREQUENCY MASKING
# ----------------------------
# Get frequency bin centers
freqs = np.linspace(0, sr / 2, D.shape[0])

# Initialize full mask
mask = np.ones_like(mag)

# Zero out each target frequency ± tolerance
for target_freq in TARGET_FREQS:
    bin_indices = np.where(np.abs(freqs - target_freq) <= TOLERANCE)[0]
    mask[bin_indices, :] = 0  # suppress these frequencies

# Apply mask
D_filtered = mask * mag * np.exp(1j * phase)

# ----------------------------
# RECONSTRUCT & SAVE
# ----------------------------
y_filtered = librosa.istft(D_filtered, hop_length=hop_length)
sf.write(OUTPUT_PATH, y_filtered, sr)
print(f"✅ Filtered audio saved to: {OUTPUT_PATH}")

# ----------------------------
# OPTIONAL: VISUALIZE
# ----------------------------
plt.figure(figsize=(12, 4))
plt.title("Spectrogram After Frequency Removal")
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_filtered), ref=np.max),
                         sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
