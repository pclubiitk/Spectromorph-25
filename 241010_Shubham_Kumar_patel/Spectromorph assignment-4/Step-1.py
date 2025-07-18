import numpy as np
import matplotlib.pyplot as plt
import os
import librosa

Audio_path = r"E:\final_cleaned.wav"

audio, sr = librosa.load(Audio_path)

audio_ft = np.fft.fft(audio)

#print(audio.shape,audio_ft.shape)

print(audio_ft[10])

magnitude_audio_spectrum = np.abs(audio_ft)

print(magnitude_audio_spectrum[10])

def plot_magnitude_spectrum(audio, title, sr, f_ratio=1):
    ft = np.fft.fft(audio)
    magnitude_spectrum = np.abs(ft)

    #plot magnitude spectrum
    plt.figure(figsize=(10,5))

    frequency = np.linspace(0, sr, len(magnitude_spectrum))
    num_frequency_bins = int(len(frequency) * f_ratio)

    plt.plot(frequency[:num_frequency_bins],magnitude_spectrum[:num_frequency_bins])
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)

    plt.show()

plot_magnitude_spectrum(audio, title="Audio Magnitude Spectrum", sr=sr,f_ratio=0.5)