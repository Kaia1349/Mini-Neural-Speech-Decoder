from __future__ import annotations

import json
from pathlib import Path


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


notebook = {
    "cells": [
        markdown_cell(
            """
            # Mini Neural Speech Decoder

            This notebook demonstrates a lightweight neural-speech decoding workflow using **synthetic EEG/MEG-like signals**. The goal is not to reproduce a full brain-computer interface experiment, but to create a compact research-style pipeline that mirrors common steps used in speech decoding studies.

            We simulate three neural-response classes:
            - **silence**
            - **vowel-like speech**
            - **consonant-like speech**
            """
        ),
        markdown_cell(
            """
            ## Research Motivation

            Neural speech decoding research tries to map brain activity to speech-related states, such as heard phonemes, imagined speech, or articulatory intent. Real studies often use EEG, MEG, ECoG, or intracranial recordings. Here, we build a toy version of that workflow with synthetic data so the full process remains easy to run and inspect.

            The notebook covers:
            1. Signal simulation for 32 channels and 300 epochs
            2. MNE-Python `RawArray` and `EpochsArray` creation
            3. Band-pass filtering from 1 to 40 Hz
            4. Simple handcrafted feature extraction
            5. Multi-class logistic regression decoding
            6. Performance reporting and figure export
            """
        ),
        code_cell(
            """
            from pathlib import Path

            import matplotlib.pyplot as plt
            from IPython.display import Image, display

            from mini_neural_speech_decoder import CLASS_NAMES, run_pipeline

            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            """
        ),
        markdown_cell(
            """
            ## Run The Pipeline

            The helper module generates structured synthetic signals, converts them into MNE containers, filters the epochs, extracts a compact feature set, and trains a logistic regression classifier.
            """
        ),
        code_cell(
            """
            pipeline = run_pipeline(results_dir)
            metrics = pipeline["metrics"]

            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print()
            print("Classification report:")
            print(metrics["classification_report"])
            print("Confusion matrix:")
            print(metrics["confusion_matrix"])
            """
        ),
        markdown_cell(
            """
            ## Feature Design

            Each epoch is summarized using simple per-channel statistics:
            - mean
            - standard deviation
            - peak amplitude
            - average band power in four coarse bands: 1-4 Hz, 4-8 Hz, 8-12 Hz, and 12-30 Hz

            These features are intentionally simple. They help show the structure of a decoding pipeline without introducing heavier deep learning or sequence modeling.
            """
        ),
        code_cell(
            """
            print("Feature matrix shape:", pipeline["features"].shape)
            print("Epoch data shape:", pipeline["epochs"].get_data(copy=True).shape)
            print("Class labels:", CLASS_NAMES)
            """
        ),
        markdown_cell(
            """
            ## Saved Figures

            The following code displays the figures saved into the `results/` folder:
            - example neural signal plot
            - power spectral density plot
            - confusion matrix
            """
        ),
        code_cell(
            """
            for label, path in pipeline["figures"].items():
                print(label, "->", path)
                display(Image(filename=str(path)))
            """
        ),
        markdown_cell(
            """
            ## Interpretation

            Because the classes were simulated with distinct temporal and spectral patterns, even a simple linear classifier can often separate them well. In real neural speech decoding, however, signal-to-noise ratios are lower, subject variability is high, and decoding performance depends heavily on preprocessing, alignment, and model choice.
            """
        ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("Mini_Neural_Speech_Decoder.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
