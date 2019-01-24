import os
import pandas as pd
import librosa
import glob
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import scipy as sp
# data, sampling_rate = librosa.load('bark_2.m4a')

# plt.figure(figsize=(12, 4))
# librosa.display.waveplot(data, sr=sampling_rate)
# plt.show()

# mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T,axis=0)

# print(mfccs)

# data2, sampling_rate2 = librosa.load('bark_1.m4a')

# plt.figure(figsize=(12, 4))
# librosa.display.waveplot(data2, sr=sampling_rate2)
# plt.show()

# mfccs2 = np.mean(librosa.feature.mfcc(y=data2, sr=sampling_rate2, n_mfcc=40).T,axis=0)


# print(mfccs2)
def compare_waves(wav1, wav2):
    result = 1 - sp.spatial.distance.cosine(wav1, wav2)
    return result

def main():
    barks = ['Angry/bark_1_single_a.m4a','Angry/bark_1_single_b.m4a','Want/bark_2_single_a.m4a','Home.m4a']
    bark_features = []
    for b in barks:
        data, sampling_rate = librosa.load(b)
        print(sampling_rate)
        plt.figure(figsize=(12, 4))
        plt.xlim(0,0.2)
        librosa.display.waveplot(data, sr=sampling_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T,axis=0)
        bark_features.append(mfccs)
        # plt.show()



    #result1 = 1 - sp.spatial.distance.cosine(bark_features[0], bark_features[1])
    
    for i in range(0, len(bark_features)):
        for j in range(i+1, len(bark_features)):
            print("RESULT OF WAVES "+str(i)+" AND "+str(j)+" IS "+str(compare_waves(bark_features[i], bark_features[j])))

main()