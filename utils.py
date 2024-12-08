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
    y, sr = librosa.load(file_path, sr=sr)
    target_length = int(fixed_duration * sr)
    y = librosa.util.fix_length(y, size = target_length)

    return y, sr

# USE AS: sample_mfccs = extract_mfcc(sample_y, sample_sr)
def extract_mfcc(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return mfccs

# USE AS: sample_delta_mfccs = extract_delta_mfcc(sample_mfccs)
def extract_delta_mfcc(mfccs):
    delta_mfccs = librosa.feature.delta(mfccs)

    return delta_mfccs

# USE AS: sample_spectral_centroid = extract_spectral_centroid(sample_y, sample_sr)
def extract_spectral_centroid(y, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    return spectral_centroid

# USE AS: sample_tonnetz = extract_tonnetz(sample_y, sample_sr)
def extract_tonnetz(y, sr):
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    return tonnetz

# USE AS: 
def extract_features(y, sr):
    mfccs = extract_mfcc(y, sr)
    mfcc_names = [f'mfcc_{i+1}' for i in range(mfccs.shape[0])]
    
    delta_mfccs = extract_delta_mfcc(mfccs)
    delta_mfcc_names = [f'delta_mfcc_{i+1}' for i in range(delta_mfccs.shape[0])]
    
    spectral_centroid = extract_spectral_centroid(y, sr).reshape(1, -1)
    spectral_centroid_name = ['spectral_centroid']
    
    tonnetz = extract_tonnetz(y, sr)
    tonnetz_names = [f'tonnetz_{i+1}' for i in range(tonnetz.shape[0])]
    
    features = np.vstack([mfccs, delta_mfccs, spectral_centroid, tonnetz])
    feature_names = mfcc_names + delta_mfcc_names + spectral_centroid_name + tonnetz_names

    features = StandardScaler().fit_transform(features)

    return features.T, feature_names


def generate_dataset(directory, MAX_LEN):
    features_list = []
    labels = []

    for filename in sorted(os.listdir(directory)):
        f = os.path.join(directory, filename)
        match = re.match(r"([^_]+)_", filename)

        label = match.group(1)
        y, sr = librosa.load(f, sr=None)
        features, _ = extract_features(y, sr)
        if features.shape[0] < MAX_LEN:
            pad_width = MAX_LEN - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='edge') # edge padding seemed to work very well for the CNN model
        else:
            features = features[:MAX_LEN]
        
        features_list.append(features)
        labels.append(label)

    X = np.array(features_list)
    y = np.array(labels)

    return X, y
