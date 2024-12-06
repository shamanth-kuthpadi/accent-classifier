import numpy as np
import pandas as pd
import subprocess
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
import glob
import os
import re
from sklearn.preprocessing import StandardScaler


# USE AS: sample_y, sample_sr = load_audio_file('ProcessedData/taru_reveal.wav')
def load_audio_file(file_path, sr = 48000, fixed_duration = 2.5):   
    
    # Load the audio file with the specified sampling rate
    y, sr = librosa.load(file_path, sr=sr)
    
    # Calculate the target number of samples for the fixed duration
    target_length = int(fixed_duration * sr)
    
    # Adjust the length of the time series - looking into how the length is being fixed
    y = librosa.util.fix_length(y, size = target_length)

    #print(f"Audio Time-Series: {y.shape}")
    #print(f"Sampling Rate: {sr}")
    #print(f"Fixed Duration (samples): {target_length}")

    return y, sr

# USE AS: sample_mfccs = extract_mfcc(sample_y, sample_sr)
def extract_mfcc(y, sr, n_mfcc=13):
    
    """
    Extract Mel-frequency cepstral coefficients (MFCCs) from an audio time-series.

    Parameters:
        y (numpy.ndarray): Audio time-series.
        sr (int): Sampling rate of the audio.
        n_mfcc (int): Number of MFCCs to return. Default is 13.

    Returns:
        numpy.ndarray: MFCCs array of shape (n_mfcc, time_frames).
    """

    # Computing the MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # print(f"MFCCs Shape: {mfccs.shape}")
    return mfccs

# USE AS: sample_delta_mfccs = extract_delta_mfcc(sample_mfccs)
def extract_delta_mfcc(mfccs):
    """
    Extract delta MFCCs (the first-order derivatives of the MFCCs).
    
    Parameters:
        mfccs (numpy.ndarray): The MFCCs array of shape (n_mfcc, time_frames).
    
    Returns:
        numpy.ndarray: The delta MFCCs array.
    """
    # Compute the delta (first derivative) of the MFCCs
    delta_mfccs = librosa.feature.delta(mfccs)
    # print(f"Delta MFCCs Shape: {delta_mfccs.shape}")
    return delta_mfccs

# USE AS: sample_spectral_centroid = extract_spectral_centroid(sample_y, sample_sr)
def extract_spectral_centroid(y, sr):
    """
    Extracts Spectral Centroid from an audio signal.
    
    Parameters:
        y (numpy.ndarray): Audio time-series.
        sr (int): Sampling rate of the audio signal.
    
    Returns:
        numpy.ndarray: Spectral centroid of the signal.
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # print(f"Spectral Centroid Shape: {spectral_centroid.shape}")
    return spectral_centroid

# USE AS: sample_tonnetz = extract_tonnetz(sample_y, sample_sr)
def extract_tonnetz(y, sr):
    """
    Extracts Tonnetz (Harmonic Features) from an audio signal.
    
    Parameters:
        y (numpy.ndarray): Audio time-series.
        sr (int): Sampling rate of the audio signal.
    
    Returns:
        numpy.ndarray: Tonnetz features.
    """
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    # print(f"Tonnetz Shape: {tonnetz.shape}")
    return tonnetz

# USE AS: 
def extract_features(y, sr):
    """
    Extracts various audio features from an audio signal.
    
    Parameters:
        y (numpy.ndarray): Audio time-series.
        sr (int): Sampling rate of the audio signal.
    
    Returns:
        tuple: (2D numpy array of extracted features, list of feature names)
    """
    # Extracting MFCCs
    mfccs = extract_mfcc(y, sr)
    mfcc_names = [f'mfcc_{i+1}' for i in range(mfccs.shape[0])]
    
    # Extracting Delta MFCCs
    delta_mfccs = extract_delta_mfcc(mfccs)
    delta_mfcc_names = [f'delta_mfcc_{i+1}' for i in range(delta_mfccs.shape[0])]
    
    # Extracting Spectral Centroid
    spectral_centroid = extract_spectral_centroid(y, sr).reshape(1, -1)
    spectral_centroid_name = ['spectral_centroid']
    
    # Extracting Tonnetz Features
    tonnetz = extract_tonnetz(y, sr)
    tonnetz_names = [f'tonnetz_{i+1}' for i in range(tonnetz.shape[0])]
    
    # Stack all features together
    features = np.vstack([mfccs, delta_mfccs, spectral_centroid, tonnetz])
    feature_names = mfcc_names + delta_mfcc_names + spectral_centroid_name + tonnetz_names

    # Normalize the features
    # NOTE: Scope to normalize each feature type separately using specific techniques for each type.
    features = StandardScaler().fit_transform(features)

    return features.T, feature_names  # Transpose to have samples as rows


def generate_dataset(name, directory):
    dataframes = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        label = re.match(r"([^_]+)_", filename)
        label = label.group(1)

        y, sr = librosa.load(f, sr=None)
        features, feature_names = extract_features(y, sr)
        df = pd.DataFrame(features, columns=feature_names)
        df['label'] = label
        dataframes.append(df)
    
    dataset = pd.concat(dataframes, ignore_index=True)
    dataset.to_csv(name + ".csv", index=False)

    return dataset
