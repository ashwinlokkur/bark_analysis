import os
import numpy
import scipy as sp

CATEGORIES = ["Want", "Angry"]

def feature_extraction(bark):
    #Mel-frequency cepstral coefficients
    data, sampling_rate = librosa.load(bark)
    librosa.display.waveplot(data, sr=sampling_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T,axis=0)
    return mfccs 

def main():
    training_set = []

main()