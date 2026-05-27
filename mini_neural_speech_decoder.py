from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ["silence", "vowel-like", "consonant-like"]
EVENT_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def build_info(n_channels: int = 32, sfreq: float = 128.0) -> mne.Info:
    ch_names = [f"MEG{index:03d}" for index in range(1, n_channels + 1)]
    ch_types = ["grad"] * n_channels
    return mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)


def _class_waveform(
    class_index: int,
    times: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    phase_1 = rng.uniform(0.0, 2.0 * np.pi)
    phase_2 = rng.uniform(0.0, 2.0 * np.pi)

    if class_index == 0:
        signal = 0.15 * np.sin(2.0 * np.pi * 10.0 * times + phase_1)
        signal += 0.05 * np.sin(2.0 * np.pi * 3.0 * times + phase_2)
        return signal

    if class_index == 1:
        envelope = np.exp(-0.5 * ((times - 0.5) / 0.18) ** 2)
        signal = 0.9 * envelope * np.sin(2.0 * np.pi * 8.0 * times + phase_1)
        signal += 0.35 * np.sin(2.0 * np.pi * 14.0 * times + phase_2)
        return signal

    burst_positions = np.array([0.18, 0.42, 0.73]) + rng.normal(0.0, 0.02, 3)
    signal = np.zeros_like(times)
    for center in burst_positions:
        envelope = np.exp(-0.5 * ((times - center) / 0.05) ** 2)
        signal += 0.7 * envelope * np.sin(2.0 * np.pi * 22.0 * times + phase_1)
    signal += 0.25 * np.sin(2.0 * np.pi * 30.0 * times + phase_2)
    return signal


def generate_synthetic_epochs(
    n_epochs: int = 300,
    n_channels: int = 32,
    sfreq: float = 128.0,
    epoch_duration: float = 1.0,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray, mne.Info]:
    rng = np.random.default_rng(random_state)
    n_times = int(sfreq * epoch_duration)
    times = np.arange(n_times) / sfreq
    info = build_info(n_channels=n_channels, sfreq=sfreq)

    samples_per_class = n_epochs // len(CLASS_NAMES)
    labels = np.repeat(np.arange(len(CLASS_NAMES)), samples_per_class)
    if labels.size < n_epochs:
        remainder = n_epochs - labels.size
        labels = np.concatenate([labels, np.arange(remainder)])
    rng.shuffle(labels)

    channel_pattern = np.sin(np.linspace(0.0, 2.0 * np.pi, n_channels, endpoint=False))
    data = np.zeros((n_epochs, n_channels, n_times), dtype=float)

    for epoch_index, class_index in enumerate(labels):
        shared_signal = _class_waveform(class_index, times, rng)
        for channel_index in range(n_channels):
            scale = 0.75 + 0.35 * rng.random()
            spatial_bias = 1.0 + 0.18 * channel_pattern[channel_index]
            noise_level = 0.16 if class_index == 0 else 0.22
            noise = noise_level * rng.normal(size=n_times)
            channel_phase = rng.uniform(0.0, 2.0 * np.pi)
            rhythm = 0.08 * np.sin(2.0 * np.pi * 6.0 * times + channel_phase)
            data[epoch_index, channel_index] = scale * spatial_bias * shared_signal + rhythm + noise

    return data, labels.astype(int), info


def create_mne_objects(
    epoch_data: np.ndarray,
    labels: np.ndarray,
    info: mne.Info,
) -> tuple[mne.io.RawArray, mne.EpochsArray]:
    n_epochs, _, n_times = epoch_data.shape
    raw_data = epoch_data.transpose(1, 0, 2).reshape(len(info["ch_names"]), n_epochs * n_times)
    raw = mne.io.RawArray(raw_data, info, verbose="ERROR")
