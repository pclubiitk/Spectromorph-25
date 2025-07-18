import os
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# ----------------------------
# CONFIGURATION
# ----------------------------
CLEAN_DIR = r"D:\vs_code_python\Sperctromorph_GANs\data\clean_testset_wav"
ASSIGNMENT_NOISY_FILE = r"C:\Users\DELL\Downloads\final_noise_audio.wav"
OUTPUT_DIR = r"E:\noisy_sound"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SNR_DB = 0.1     # Lower means more challenging noise
NUM_DOMINANT_FREQS = 30 # Use top N frequency peaks

# ----------------------------
# STEP 1 — Extract Dominant Frequencies from Assignment Noise
# ----------------------------
def extract_dominant_frequencies(audio_path, top_n=6):
    y, sr = librosa.load(audio_path, sr=None)
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), d=1 / sr)

    magnitude = np.abs(Y[:len(Y) // 2])
    freq_pos = freqs[:len(Y) // 2]

    # Get top-N frequency peaks (ignore DC component)
    indices = np.argsort(magnitude[1:])[-top_n:] + 1
    dom_freqs = np.sort(freq_pos[indices])
    print("🎯 Dominant frequencies (Hz):", dom_freqs)
    return dom_freqs, sr

# ----------------------------
# STEP 2 — Tone Noise Generator
# ----------------------------
def generate_targeted_tone_noise(length, sr, freqs, amp=0.05):
    t = np.linspace(0, length / sr, length)
    tones = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    tones /= len(freqs)
    return tones * amp

# ----------------------------
# STEP 3 — Mix with Clean at SNR
# ----------------------------
def mix_noise(clean, noise, snr_db):
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(clean_power / (10**(snr_db / 10) * noise_power))
    return clean + noise * scaling_factor

# ----------------------------
# MAIN LOOP
# ----------------------------
def process_all_clean_files():
    freqs, sr = extract_dominant_frequencies(ASSIGNMENT_NOISY_FILE, NUM_DOMINANT_FREQS)

    for filename in os.listdir(CLEAN_DIR):
        if not filename.endswith(".wav"):
            continue

        clean_path = os.path.join(CLEAN_DIR, filename)
        audio, sr = sf.read(clean_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        audio = audio / np.max(np.abs(audio))  # Normalize

        length = len(audio)
        tone_noise = generate_targeted_tone_noise(length, sr, freqs)
        noisy = mix_noise(audio, tone_noise, TARGET_SNR_DB)
        noisy = np.clip(noisy, -1.0, 1.0)

        out_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(out_path, noisy, sr)
        print(f"✅ Noisy saved: {filename}")

# ----------------------------
# RUN IT
# ----------------------------
if __name__ == "__main__":
    extract_dominant_frequencies(ASSIGNMENT_NOISY_FILE, NUM_DOMINANT_FREQS)
