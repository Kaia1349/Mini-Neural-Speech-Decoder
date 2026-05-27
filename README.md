# Mini Neural Speech Decoder

Mini Neural Speech Decoder is a small research-style Python project that simulates a neural-speech decoding pipeline inspired by EEG/MEG speech decoding studies. It uses synthetic multichannel signals, MNE-Python data structures, simple feature engineering, and a lightweight classifier to demonstrate the overall workflow in a form that is easy to run on a laptop.

## Project Motivation

Neural speech decoding sits at the intersection of neuroscience, signal processing, and machine learning. Researchers study whether neural activity can reveal speech perception, imagined speech, articulatory intent, or phoneme-level distinctions. This project provides a compact educational version of that pipeline using synthetic data, making it useful for learning and experimentation without requiring access to real recordings.

## Relation To Neural Speech Decoding

Real neural speech decoding studies often work with EEG, MEG, ECoG, or intracranial recordings and try to predict speech units or behavioral states from noisy neural measurements. This project mirrors several core steps from that setting:

- multichannel neural signal generation
- segmentation into epochs
- band-pass filtering
- handcrafted feature extraction
- supervised classification
- quantitative evaluation with accuracy and confusion analysis

The classes in this toy setup are:

- `silence`
- `vowel-like speech`
- `consonant-like speech`

These are simulated categories, not biologically validated neural signatures.

## Technical Pipeline

1. Generate 300 synthetic epochs with 32 channels.
2. Create MNE `RawArray` and `EpochsArray` objects.
3. Apply a 1 to 40 Hz band-pass filter.
4. Extract simple per-channel features:
   mean, standard deviation, peak amplitude, and coarse band power.
5. Train a scikit-learn `LogisticRegression` classifier.
6. Report accuracy, classification report, and confusion matrix.
7. Save figures into the `results/` directory.

## Tools Used

- Python
- NumPy
- MNE-Python
- scikit-learn
- matplotlib
- Jupyter Notebook

## Project Files

- [Mini_Neural_Speech_Decoder.ipynb](/Users/qiyanhuang/Documents/Codex/2026-05-24/create-a-small-python-research-project/Mini_Neural_Speech_Decoder.ipynb)
- [mini_neural_speech_decoder.py](/Users/qiyanhuang/Documents/Codex/2026-05-24/create-a-small-python-research-project/mini_neural_speech_decoder.py)
- [results](/Users/qiyanhuang/Documents/Codex/2026-05-24/create-a-small-python-research-project/results)

## Running The Project

Install the required packages if needed:

```bash
python3 -m pip install numpy matplotlib scikit-learn mne notebook
```

Then open the notebook:

```bash
jupyter notebook Mini_Neural_Speech_Decoder.ipynb
```

The notebook will generate the figures automatically in the `results/` folder when executed.

## Limitations

- The neural data are synthetic and intentionally simplified.
- The class structure is easier than a real speech decoding task.
- The model uses handcrafted summary features instead of temporal sequence modeling.
- There is no subject-to-subject variability, artifact rejection, or realistic sensor geometry.
- The reported scores should not be interpreted as evidence of real neural decoding performance.

## Future Work

- add time-frequency features or wavelet features
- compare logistic regression with SVMs, random forests, or shallow neural networks
- simulate subject variability and domain shifts
- include cross-validation and hyperparameter tuning
- extend the task from coarse classes to phoneme-like or word-like categories
