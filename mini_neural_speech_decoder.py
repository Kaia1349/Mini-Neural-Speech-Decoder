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

    events = np.column_stack(
        [
            np.arange(n_epochs, dtype=int) * n_times,
            np.zeros(n_epochs, dtype=int),
            labels,
        ]
    )
    epochs = mne.EpochsArray(epoch_data, info, events=events, event_id=EVENT_ID, tmin=0.0, verbose="ERROR")
    return raw, epochs


def filter_epochs(epochs: mne.EpochsArray) -> mne.EpochsArray:
    return epochs.copy().filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")


def _bandpower_matrix(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / sfreq)
    spectrum = np.abs(np.fft.rfft(data, axis=-1)) ** 2
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return spectrum[..., band_mask].mean(axis=-1)


def extract_features(epochs: mne.EpochsArray) -> np.ndarray:
    data = epochs.get_data(copy=True)
    sfreq = float(epochs.info["sfreq"])

    mean_feature = data.mean(axis=-1)
    std_feature = data.std(axis=-1)
    peak_feature = np.abs(data).max(axis=-1)

    bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)]
    band_features = [_bandpower_matrix(data, sfreq, band) for band in bands]

    feature_blocks = [mean_feature, std_feature, peak_feature, *band_features]
    return np.concatenate(feature_blocks, axis=1)


def train_classifier(features: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": model,
        "x_test": x_test,
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


def plot_example_signal(raw: mne.io.RawArray, output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    time_vector = raw.times[:256]
    for channel_index in range(5):
        channel_trace = raw.get_data(picks=[channel_index])[0, :256]
        ax.plot(time_vector, channel_trace + channel_index * 2.0, label=raw.ch_names[channel_index])

    ax.set_title("Example Synthetic Neural Signals")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (offset per channel)")
    ax.legend(loc="upper right", ncol=1, fontsize=8)
    ax.grid(alpha=0.25)

    figure_path = output_dir / "example_neural_signal.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def plot_psd(epochs: mne.EpochsArray, output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    psd = epochs.compute_psd(fmin=1.0, fmax=40.0, verbose="ERROR")
    spectrum = psd.get_data().mean(axis=(0, 1))
    freqs = psd.freqs

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, spectrum, color="#0d6b6b", linewidth=2.0)
    ax.set_title("Average Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.grid(alpha=0.25)

    figure_path = output_dir / "power_spectral_density.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def plot_confusion_matrix_figure(
    cm: np.ndarray,
    output_dir: Path | str,
    labels: list[str] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels or CLASS_NAMES

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)

    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            ax.text(col_index, row_index, str(cm[row_index, col_index]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    figure_path = output_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def run_pipeline(output_dir: Path | str = "results") -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_data, labels, info = generate_synthetic_epochs()
    raw, epochs = create_mne_objects(epoch_data, labels, info)
    filtered_epochs = filter_epochs(epochs)
    features = extract_features(filtered_epochs)
    metrics = train_classifier(features, labels)

    figures = {
        "example_signal": plot_example_signal(raw, output_dir),
        "psd": plot_psd(filtered_epochs, output_dir),
        "confusion_matrix": plot_confusion_matrix_figure(metrics["confusion_matrix"], output_dir),
    }

    return {
        "raw": raw,
        "epochs": epochs,
        "filtered_epochs": filtered_epochs,
        "features": features,
        "labels": labels,
        "metrics": metrics,
        "figures": figures,
    }
