import tensorflow as tf
import keras
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping
import os

# matplotlib inline
import matplotlib.pyplot as plt
import librosa
import librosa.display
#import IPython.display
import numpy as np
from numpy import argmax
import pandas as pd
import random

import warnings
warnings.filterwarnings('ignore')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Read Data
data = pd.read_csv('./music_analysis.csv')
print(data.head(10))

print(data.shape)

def folder_name(i):
    if i < 18000:
        return '0'
    elif i >= 18000 and i < 41000:
        return '1'
    elif i >= 41000 and i < 57000:
        return '2'
    elif i >= 57000 and i < 72000:
        return '3'
    elif i >= 72000 and i < 89000:
        return '4'
    elif i >= 89000 and i < 108000:
        return '5'
    elif i >= 108000 and i < 117000:
        return '6'
    elif i >= 117000 and i < 127000:
        return '7'
    elif i >= 127000 and i < 138000:
        return '8'
    else:
        return '9'

def genre_number(i):
    if i == 'Hip-Hop':
        return 0
    elif i == 'Pop':
        return 1
    elif i == 'Folk':
        return 2
    elif i == 'Rock':
        return 3
    elif i == 'Experimental':
        return 4
    elif i == 'International':
        return 5
    elif i == 'Electronic':
        return 6
    else:
        return '7' #instrumental

data['genre_number'] = data['genre'].apply(genre_number)
data['folder_number'] = data['file_name'].apply(folder_name)

data['file_name'] = data['file_name'].apply(lambda x: '{0:0>6}'.format(x))
i=0
for j in data['folder_number']:
    data['folder_number'][i] = j.zfill(3)
    i = i+1
data['path'] = data['folder_number'].astype('str') + '/' + data['file_name'].astype('str') + ".mp3"
data['path'].head(5)

# Example of Hip-Hop music
y, sr = librosa.load('./fma_small/000/000002.mp3', duration=10)
ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
ps.shape

ps

# IPython.display.Audio(data=y, rate=sr)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# Example of Pop music
y, sr = librosa.load('./datasets/fma_small/000/000010.mp3', duration=10)
ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
ps.shape

# IPython.display.Audio(data=y, rate=sr)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# Example of Rock music
y, sr = librosa.load('./datasets/fma_small/000/000255.mp3', duration=10)
ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
ps.shape

# IPython.display.Audio(data=y, rate=sr)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

rate = 0.8

for row in data.itertuples():
    y, sr = librosa.load(os.getcwd()+'/datasets/fma_small/' + row.path)
    y_changed = librosa.effects.time_stretch(y, rate=rate)
    str_tmp = os.getcwd()+'/datasets/fma_small_augmented/' + row.path
    path, file = os.path.split(str_tmp)
    if not os.path.exists(path):
        os.makedirs(path)
    librosa.output.write_wav(os.getcwd()+'/datasets/fma_small_augmented/' + row.path, y_changed, sr)

rate = 0.9

for row in data.itertuples():
    y, sr = librosa.load('./datasets/fma_small/' + row.path)
    y_changed = librosa.effects.time_stretch(y, rate=rate)
    str_tmp = './datasets/fma_small_augmented2/' + row.path
    path, file = os.path.split(str_tmp)
    if not os.path.exists(path):
        os.makedirs(path)
    librosa.output.write_wav('./datasets/fma_small_augmented2/' + row.path, y_changed, sr)

n_steps = 2

for row in data.itertuples():
    y, sr = librosa.load('./datasets/fma_small/' + row.path)
    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    str_tmp = './datasets/fma_small_augmented1/' + row.path
    path, file = os.path.split(str_tmp)
    if not os.path.exists(path):
        os.makedirs(path)
    librosa.output.write_wav('./datasets/fma_small_augmented1/' + row.path, y_changed, sr)

n_steps = -2

for row in data.itertuples():
    y, sr = librosa.load('./datasets/fma_small/' + row.path)
    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    str_tmp = './datasets/fma_small_augmented3/' + row.path
    path, file = os.path.split(str_tmp)
    if not os.path.exists(path):
        os.makedirs(path)
    librosa.output.write_wav('./datasets/fma_small_augmented3/' + row.path, y_changed, sr)

n_steps = -2
'''
for row in data.itertuples():
    y, sr = librosa.load('./datasets/fma_small/' + row.path)
    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    librosa.output.write_wav('./datasets/fma_small_augmented3/' + row.path, y_changed, sr)    
'''


#  IPython.display.Audio(data=y, rate=sr)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# Example of Rock music (time-stretch 0.8)
y, sr = librosa.load('./datasets/fma_small_augmented/000/000002.mp3', duration=10)
ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
ps.shape

# IPython.display.Audio(data=y, rate=sr)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# Example of Rock music (pitch-shift 2)
y, sr = librosa.load('./datasets/fma_small_augmented1/000/000002.mp3', duration=10)
ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
ps.shape

# IPython.display.Audio(data=y, rate=sr)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

D1 = []  # Dataset1

for row in data.itertuples():
    y, sr = librosa.load('./datasets/fma_small/' + row.path, duration=10)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    if ps.shape != (128, 431): continue
    D1.append((ps, row.genre_number))

D2 = []  # Dataset2 (time stretch 0.8)

for row in data.itertuples():
    y, sr = librosa.load('./datasets/fma_small_augmented/' + row.path, duration=10)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    if ps.shape != (128, 431): continue
    D2.append((ps, row.genre_number))

D3 = []  # Dataset3 (pitch shift 2)

for row in data.itertuples():
    y, sr = librosa.load('./datasets/fma_small_augmented1/' + row.path, duration=10)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    if ps.shape != (128, 431): continue
    D3.append((ps, row.genre_number))

D4 = []  # Dataset4 (time stretch 1.1)

for row in data.itertuples():
    y, sr = librosa.load('./datasets/fma_small_augmented2/' + row.path, duration=10)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    if ps.shape != (128, 431): continue
    D4.append((ps, row.genre_number))
D5 = []  # Dataset5 (pitch shift -2)

for row in data.itertuples():
    y, sr = librosa.load('./datasets/fma_small_augmented3/' + row.path, duration=10)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    if ps.shape != (128, 431): continue
    D5.append((ps, row.genre_number))

D = D1 + D2 + D3 + D4 + D5

print("Nu|mber of samples: ", len(D))

dataset = D
random.shuffle(dataset)

#train dev test split 8:1:1
train = dataset[:32000]
dev = dataset[32000:36000]
test = dataset[36000:]

X_train, Y_train = zip(*train)
X_dev, Y_dev = zip(*dev)
X_test, Y_test = zip(*test)

# Reshape for CNN input
X_train = np.array([x.reshape((128, 431, 1)) for x in X_train])
X_dev = np.array([x.reshape((128, 431, 1)) for x in X_dev])
X_test = np.array([x.reshape((128, 431, 1)) for x in X_test])

# One-Hot encoding for classes
Y_train = np.array(keras.utils.to_categorical(Y_train, 8))
Y_dev = np.array(keras.utils.to_categorical(Y_dev, 8))
Y_test = np.array(keras.utils.to_categorical(Y_test, 8))

model = Sequential()
input_shape = (128, 431, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="same"))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(8))
model.add(Activation('softmax'))

model.summary()

epochs = 200
batch_size = 8
learning_rate = 0.01
decay_rate = learning_rate / epochs
momentum = 0.9
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])

epochs = 100
batch_size = 64
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])


with K.tf.device('/gpu:0'):
    tb_hist = keras.callbacks.TensorBoard(log_dir='./.keras/graph', histogram_freq=0, write_graph=True, write_images=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
    hist = model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_dev, Y_dev), callbacks=[early_stopping, tb_hist])

plt.figure(figsize=(12, 8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'])
plt.show()

score = model.evaluate(x=X_test, y=Y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#terminal에 텐서보드 실행
# tensorboard --logdir="C:/Users/Yang Saewon/.keras/graph"

# 모델 저장
model.save('music_genre_classification.h5')

# 모델 불러오기
model = load_model('music_genre_classification.h5')

# Read Test Data
data = pd.read_csv('music_test.csv')

def genre_number(i):
    if i == 'Hip-Hop':
        return 0
    elif i == 'Pop':
        return 1
    elif i == 'Folk':
        return 2
    elif i == 'Rock':
        return 3
    elif i == 'Experimental':
        return 4
    elif i == 'International':
        return 5
    elif i == 'Electronic':
        return 6
    else:
        return '7'


data['genre_number'] = data['genre'].apply(genre_number)
data['file_name'] = data['file_name'].apply(lambda x: '{0:0>6}'.format(x))
data['path'] = data['file_name'].astype('str') + ".mp3"
data['path'].head(5)

D = []  # test dataset

for row in data.itertuples():
    y, sr = librosa.load('./datasets/fma_small/000/' + row.path, duration=10)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    if ps.shape != (128, 431): continue
    D.append((ps, row.genre_number))

print("Nu|mber of samples: ", len(D))

test = D
X_test, Y_test = zip(*test)
X_test = np.array([x.reshape((128, 431, 1)) for x in X_test])
Y_test = np.array(keras.utils.to_categorical(Y_test, 8))

yhat = model.predict_classes(X_test)

for i in range(len(D)):
    print('file_name: ' + data['file_name'][i] + ' True: ' + str(argmax(Y_test[i])) + ', Predict: ' + str(yhat[i]))
