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
import math

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
# print(device_lib.list_local_devices())




# PASOS: carga dataset de audio y de voz, coge 10 segundos de cada uno, calcula el RMS, los normaliza y los une --- despues ya se elige el porcentaje de audio



# AUDIO

y, sr = librosa.load('./datasets/audio/021/021058.mp3', duration=10)

librosa.feature.rms(y=y)

S1, phase = librosa.magphase(librosa.stft(y))
rms1 = librosa.feature.rms(S=S1)
plt.figure()
plt.subplot(2, 1, 1)
# plt.plot(y) La señal 'y' toma valores entre -1 y 1, sin embargo creando el espectograma y representandolo de la siguiente forma se ve que la señal toma valores entre 0 y 10 aprox.
plt.semilogy(rms1.T, label='RMS Energy')
plt.xticks([])
plt.xlim([0, rms1.shape[-1]])
plt.legend(loc='best')
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S1, ref=np.max), y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()

media1 = np.mean(rms1)
print('Media del valor RMS del audio: ', media1)

# normalizar
# despues??



# VOZ

z, sr = librosa.load('./datasets/PATH DE UN SEGMENTO DE VOZ', duration=10)
librosa.feature.rms(y=z)
S2, phase = librosa.magphase(librosa.stft(y))
rms2 = librosa.feature.rms(S=S2)
plt.figure()
plt.subplot(2, 1, 1)
# plt.plot(y) La señal 'y' toma valores entre -1 y 1, sin embargo creando el espectograma y representandolo de la siguiente forma se ve que la señal toma valores entre 0 y 10 aprox.
plt.semilogy(rms2.T, label='RMS Energy')
plt.xticks([])
plt.xlim([0, rms2.shape[-1]])
plt.legend(loc='best')
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S2, ref=np.max), y_axis='log', x_axis='time')
plt.title('log Power spectrogram')
plt.tight_layout()

media2 = np.mean(rms2)
print('Media del valor RMS de la voz: ', media2)

# normalizar
# despues??

# Concatenar señal de audio y voz