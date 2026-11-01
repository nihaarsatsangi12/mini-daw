"""
Schroeder reverb demo: apply my Schroeder reverberator to a WAV file and
plot the dry vs. wet waveform and spectrogram.

Usage:  python schroeder_demo.py input.wav
Writes: schroeder_output.wav, waveform_comparison.png, spectrogram_comparison.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

from dsp import schroeder_reverb


def plot_spectrogram(ax, signal, sr, title):
    f, t, sxx = spectrogram(signal, sr, nperseg=1024, noverlap=512)
    mesh = ax.pcolormesh(t, f, 10 * np.log10(sxx + 1e-9), shading="gouraud")
    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    plt.colorbar(mesh, ax=ax, label="Power (dB)")


def main(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono, as in Schroeder's design

    wet = schroeder_reverb(audio, sr)
    sf.write("schroeder_output.wav", wet, sr)

    dry = audio / (np.max(np.abs(audio)) + 1e-9)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6))
    axes[0].plot(dry, color="blue")
    axes[0].set_title("Waveform - Original (Dry)")
    axes[1].plot(wet, color="purple")
    axes[1].set_title("Waveform - After Schroeder Reverb (Wet)")
    for ax in axes:
        ax.set_xlabel("Samples")
    fig.tight_layout()
    fig.savefig("waveform_comparison.png", dpi=120)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    plot_spectrogram(axes[0], dry, sr, "Spectrogram - Original (Dry)")
    plot_spectrogram(axes[1], wet, sr, "Spectrogram - Schroeder Reverb (Wet)")
    fig.tight_layout()
    fig.savefig("spectrogram_comparison.png", dpi=120)

    print("Saved schroeder_output.wav, waveform_comparison.png and spectrogram_comparison.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python schroeder_demo.py input.wav")
    main(sys.argv[1])
