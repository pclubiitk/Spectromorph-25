import os
import numpy as np
import soundfile as sf

# CONFIG
CLEAN_DIR = r"D:\vs_code_python\Sperctromorph_GANs\data\clean_testset_wav"    # Input directory
NOISY_DIR = r"E:\noisy_sound"     # Output directory
AMPLITUDE_SCALE = 0.30       # Noise volume (try 0.1 to 0.3)

def add_white_noise_to_file(input_path, output_path):
    data, samplerate = sf.read(input_path)

    # Handle mono and stereo
    if data.ndim == 1:
        noise = np.random.normal(0, 1, len(data))
    else:
        noise = np.random.normal(0, 1, data.shape)

    # Normalize noise
    noise = noise / np.max(np.abs(noise))

    # Add scaled noise to signal
    noisy_data = data + AMPLITUDE_SCALE * noise

    # Clip to valid range
    noisy_data = np.clip(noisy_data, -1.0, 1.0)

    # Save
    sf.write(output_path, noisy_data, samplerate)

# MAIN
os.makedirs(NOISY_DIR, exist_ok=True)

for file in os.listdir(CLEAN_DIR):
    if file.endswith(".wav"):
        in_path = os.path.join(CLEAN_DIR, file)
        out_path = os.path.join(NOISY_DIR, file)
        add_white_noise_to_file(in_path, out_path)
        print(f"✅ Noisy file saved: {file}")
