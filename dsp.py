"""
Custom DSP effects for the Mini DAW.

Everything here works on float32 NumPy arrays shaped (samples, channels).
Pedalboard expects (channels, samples), so `run_board` handles the conversion.
"""
import numpy as np
from scipy.signal import lfilter, butter, sosfiltfilt
from pedalboard import Pedalboard, PitchShift, Mix


def run_board(board, audio, sr):
    """Apply a Pedalboard chain to audio shaped (samples, channels)."""
    return board(audio.T, sr).T


def normalize(audio):
    """Scale down only if the signal would clip."""
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32)


# ---------------------------------------------------------------- Schroeder reverb
# My own implementation of Schroeder's 1962 reverberator:
# 4 parallel feedback comb filters -> sum -> 2 all-pass filters in series.

def comb_filter(x, delay, g):
    """Feedback comb filter: y[n] = x[n] + g * y[n - delay]."""
    a = np.zeros(delay + 1)
    a[0], a[delay] = 1.0, -g
    return lfilter([1.0], a, x)


def allpass_filter(x, delay, g):
    """Schroeder all-pass filter: y[n] = -g*x[n] + x[n - delay] + g*y[n - delay]."""
    b = np.zeros(delay + 1)
    b[0], b[delay] = -g, 1.0
    a = np.zeros(delay + 1)
    a[0], a[delay] = 1.0, -g
    return lfilter(b, a, x)


def schroeder_reverb(x, sr):
    """Return the fully wet reverb of a 1-D signal, normalised to a peak of 1."""
    comb_delays = [int(sr * 0.050), int(sr * 0.056), int(sr * 0.061), int(sr * 0.068)]
    comb_gains = [0.773, 0.802, 0.753, 0.733]

    out = sum(comb_filter(x, d, g) for d, g in zip(comb_delays, comb_gains))
    out = allpass_filter(out, int(sr * 0.005), 0.7)
    out = allpass_filter(out, int(sr * 0.0017), 0.7)
    return out / (np.max(np.abs(out)) + 1e-9)


def schroeder_effect(audio, sr, wet=0.3):
    """Blend the Schroeder reverb with the dry signal, channel by channel."""
    peak = np.max(np.abs(audio)) + 1e-9
    out = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        reverb = schroeder_reverb(audio[:, ch].astype(np.float64), sr) * peak
        out[:, ch] = (1 - wet) * audio[:, ch] + wet * reverb
    return normalize(out)


# ---------------------------------------------------------------- Multi-pitch harmonizer

def multi_pitch_octaves(audio, sr):
    """Layer the original with copies shifted +12, -12 and +24 semitones."""
    board = Pedalboard([
        Mix([
            Pedalboard([]),                  # original (0 semitones)
            Pedalboard([PitchShift(12.0)]),  # one octave up
            Pedalboard([PitchShift(-12.0)]), # one octave down
            Pedalboard([PitchShift(24.0)]),  # two octaves up
        ])
    ])
    return normalize(run_board(board, audio, sr))


# ---------------------------------------------------------------- De-esser

def simple_deesser(audio, sr, low_hz=5000.0, high_hz=10000.0,
                   threshold_db=-32.0, ratio=6.0, reduction_amount=0.7,
                   smoothing_ms=5.0):
    """
    Dynamic de-esser: isolate the sibilance band (about 5-10 kHz), measure its
    level, and turn that band down only while it is louder than the threshold.
    """
    high_hz = min(high_hz, 0.45 * sr)
    # Zero-phase band-pass, so the band lines up exactly with the original and
    # subtracting part of it really removes that part of the sound.
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    band = sosfiltfilt(sos, audio, axis=0).astype(np.float32)

    # Smoothed level of the band (moving average of the absolute value)
    win = max(1, int(sr * smoothing_ms / 1000))
    envelope = np.convolve(np.mean(np.abs(band), axis=1), np.ones(win) / win, mode="same")
    level_db = 20 * np.log10(envelope + 1e-8)

    # Compressor gain curve: above the threshold, level rises only 1/ratio as fast
    gain_db = np.zeros_like(level_db)
    over = level_db > threshold_db
    gain_db[over] = (threshold_db + (level_db[over] - threshold_db) / ratio) - level_db[over]
    gain = (10 ** (gain_db / 20.0))[:, None]

    # Remove part of the sibilance band: audio - amount * band * (1 - gain)
    out = audio + reduction_amount * (band * gain - band)
    return normalize(out)


# ---------------------------------------------------------------- 8D-style auto-panning

def three_d_audio(audio, sr, rotation_speed=0.5, depth=1.0):
    """
    Move the sound smoothly between the left and right speakers (auto-panning),
    using equal-power panning so the loudness stays constant.
    rotation_speed: sweeps per second (Hz). depth: 0 = centred, 1 = full left/right.
    """
    mono = audio.mean(axis=1)
    t = np.arange(len(mono)) / sr
    position = 0.5 + 0.5 * depth * np.sin(2 * np.pi * rotation_speed * t)  # 0 = left, 1 = right
    left = mono * np.cos(position * np.pi / 2)
    right = mono * np.sin(position * np.pi / 2)
    return normalize(np.stack([left, right], axis=1))
