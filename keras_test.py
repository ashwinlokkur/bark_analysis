import tensorflow as tf
import tensorflow.keras as keras
import librosa
import os
import numpy as np
DATADIR = "ScoobyBark"
CATEGORIES = ["Angry", "Want"]

training_data = []

def create_training_data():
    global training_data
    for c in CATEGORIES:
        path = os.path.join(DATADIR, c)
        catg_num = CATEGORIES.index(c)
        for bark in os.listdir(path):
            try:
                data, sampling_rate = librosa.load(os.path.join(path, bark))
                mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T,axis=0)
                print(type(mfccs), mfccs.shape)
                training_data.append([mfccs, catg_num])
            except Exception:
                pass

def main():
    create_training_data()
    X=[] #features x_train
    y=[] #labels y_train
    for features, labels in training_data:
        X.append(features)
        y.append(labels)
    X = np.array(X)
    y = np.array(y)
    
    X = tf.keras.utils.normalize(X, axis=0)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

    model.fit(X, y, epochs = 3)

    model.save('model1_bark.hdf5')

    d_test, sr_test = librosa.load('Home.m4a')
    mfc_test = np.mean(librosa.feature.mfcc(y=d_test, sr=sr_test, n_mfcc=40).T,axis=0)
    mfc_test = np.array(mfc_test)
    mfc_test = tf.keras.utils.normalize(mfc_test, axis=0)

    p = model.predict([mfc_test])
    print(np.argmax(p))
main()