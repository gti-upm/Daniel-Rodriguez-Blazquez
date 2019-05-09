from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from os import scandir, getcwd, makedirs
import os
from os.path import abspath
from time import time
import pprint
import numpy as np
from random import shuffle
import math
from keras.callbacks import TensorBoard
from pylab import savefig

''' Funcion que devuelve carpetas con directorios '''
def ls_dir(ruta = getcwd()):

    return [abspath(arch.path) for arch in scandir(ruta) if not arch.is_file()]


''' Funcion que devuelve carpetas con archivos '''
def ls_file(ruta = getcwd()):

    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]


''' Pregunta si se quiere generar un dataset '''
def ask_generate_dataset():
    answer = input("Empezar con el entrenamiento? (y/n): ")
    if answer == 'y':
        ret = True
    elif answer == 'n':
        ret = False
    else:
        ret = False
        print(input("Error. Solo se admite 'y' o 'n'"))
    return ret


''' Funcion que crea un diccionario con ciertas caracteristicas de la señal '''
def create_dictionary(song_path, duration, signal, sr):

    new_song = {
        "path": song_path,
        "duration": duration,
        "signal": signal,
        "sample_rate": sr,
        "RMS": np.mean(librosa.feature.rms(y=signal))
    }

    return new_song


''' Funcion que guarda los espectogramas en carpetas según su porcentaje y que devuelve la base de datos c '''
def save_melgrams(dataset_path):

    stats = {"dataset_size":len(dataset_path)}
    for k, song_path in enumerate(dataset_path):
        song_signal, song_sr = librosa.load(song_path, duration=None, sr=44100)
        song_duration = librosa.get_duration(y=song_signal, sr=song_sr)
        song_dictionary = create_dictionary(song_path, song_duration, song_signal, song_sr)

        mel_spectogram = extract_features(song_dictionary)
        if k == 0:
            stats["height"] = mel_spectogram.shape[0]
            stats["width"] = mel_spectogram.shape[1]
        folder_number = song_path.split("/")[-2]
        song_name = song_path.split("/")[-1].replace(".wav", "")

        path = os.path.join(getcwd(), 'datasets/train_spec/{}'.format(folder_number))
        print(f"Comprobando si existe {path}")
        if not os.path.exists(path):
            print("No existe, se crea")
            makedirs(path)
        print(f"Abrimos el archivo {path + '/mel_spectrogram_.txt'}")
        file_path = path + "/mel_spectrogram_{}".format(song_name)
        np.save(file_path, mel_spectogram)
        # with open(path + "/mel_spectrogram_{}.txt".format(song_name), "w+") as file:
        #   mel_spectogram = [str(value) for value in mel_spectogram]
        #   file.write(",".join(mel_spectogram))
        '''
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(spectrogram_dataset[0], ref=np.max), y_axis='mel', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        # savefig('mel-espectograma_{}.png'.format(k)) # Guarda la foto con el nombre de path
        '''

        # Poner etiqueta a cada espectrograma (k)
        # Guardar espectrograma


    return stats


''' Funcion que divide la base de datos en training 80%, test 20%, validation 20% '''
def split_dataset(dataset):

    print(len(dataset))
    aux1 = math.floor(0.8 * len(dataset))
    print(aux1)

    train = dataset[:(aux1-1)]
    test = dataset[aux1:]

    aux2 = math.floor(0.8 * len(train))
    print(aux2)

    validation = train[aux2:]
    train = train[:(aux2-1)]

    return train, test, validation


''' Funcion que genera el mel-espectrograma de la cancion que se le pasa como parametro '''
def compute_mel_gram(src, sr, power, duration):

    n_fft = 512
    n_mel = 96
    hop_len = 256
    n_sample = src.shape[0]
    n_sample_fit = int(duration*sr)

    if n_sample < n_sample_fit:  # if too short
        src = np.concatenate([src, np.zeros((int(duration*sr) - n_sample,))])
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    mel = librosa.feature.melspectrogram(
            y=src, sr=sr, hop_length=hop_len,
            n_fft=n_fft, n_mels=n_mel, power=power)
    ret = librosa.power_to_db(mel)

    return ret


''' Genera el dataset con los paths de todas las canciones ''' #-----------Descomentar----
def compute_dataset():

    # mix_folder_paths = ls_dir(ruta=getcwd() + '/datasets/train/')                     # Lista sin orden con los paths de todos los directorios de AUDIO
    mix_song_paths = ls_file(ruta=getcwd() + '/datasets/train/40/')

    '''all_mix_song_paths = []                                                           # Lista sin orden con los paths de todas las canciones (AUDIO)
    for mix_folder_path in mix_folder_paths:
        mix_song_paths = ls_file(mix_folder_path)
        all_mix_song_paths.extend(mix_song_paths)
   
    shuffle(all_mix_song_paths)

    #  Coger canciones de los paths
    return all_mix_song_paths
     '''

    return mix_song_paths

''' Funcion que devuelve el mel-espectrograma de una cancion '''
def extract_features(song):

    y = song['signal']
    sr = song['sample_rate']
    duration = song['duration']
    mel_spectrogram = compute_mel_gram(y, sr, 2.0, duration)
    savefig('')

    return mel_spectrogram



if __name__ == "__main__":

    cond = ask_generate_dataset()
    if cond:
        dataset_path = compute_dataset()
        stats = save_melgrams(dataset_path)
        print('Mel-espectrogramas creados: ', stats["dataset_size"])
        print('Tamaño de los mel-espectrogramas: alto: {} y ancho {}'.format(stats["height"], stats["width"]))

    spectogram_dataset = None
    [train, test, validation] = split_dataset(spectogram_dataset)

    tamaño_imagenes = [spectogram_dataset[0].shape[0], spectogram_dataset[0].shape[1]]

    model = InceptionV3(weights=None, include_top=False, input_shape=tamaño_imagenes)

    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')

    batch_size = 64
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    model.fit(spectogram_dataset, epochs=20, verbose=1, callbacks=[tensorboard])

    # scores = model.evaluate(spectrogram_dataset, target_data)
    # print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
    print(model.predict(spectogram_dataset).round())


    '''
    y, sr = librosa.load('./datasets/train/50/mix_1.wav', duration=5)
    duration = librosa.get_duration(y=y, sr=sr)
    ps = compute_mel_gram(y, sr, 2.0, duration)
    print(ps.shape)
    print('------------------------')

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(ps, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    img = image.load_img(ps, target_size=(96, 431))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model = Sequential
    model.add(Dense(16, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # model = InceptionV3(weights=None)

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['binary_accuracy'])

    model.fit(training_data, target_data, epochs=2)

    scores = model.evaluate(training_data, target_data)
    print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
    print(model.predict(training_data).round())

    preds = model.predict(x)
    print('Prediction:', decode_predictions(preds, top=1)[0][0])
    '''
