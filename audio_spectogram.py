import matplotlib.pyplot as plt
import librosa 
from librosa import display
import numpy as np
y, sr = librosa.load('Want/bark_2_single_a.m4a')
y_harm, y_perc = librosa.effects.hpss(y)
plt.subplot(3, 1, 3)
librosa.display.waveplot(y_harm, sr=sr, alpha=0.25)
librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5)
plt.title('Harmonic + Percussive')
plt.tight_layout()
plt.show()

# plt.figure(figsize=(12, 8))
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# plt.subplot(4, 2, 1)
# librosa.display.waveplot(y, sr=sr)
# plt.title('Stereo')
# plt.show()