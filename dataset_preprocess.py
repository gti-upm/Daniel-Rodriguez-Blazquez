import librosa
from os import getcwd, makedirs
import os
import sys
import gflags
from time import time
import numpy as np
from random import shuffle
import math
from utils import list_dataset, create_dictionary, ask_generate_dataset, extract_features
from common_flags import FLAGS


''' Funcion que guarda los espectogramas en carpetas según su porcentaje y que devuelve la base de datos c '''
def save_melgrams(dataset_path):
    print('Tamaño del dataset:', len(dataset_path))

    for song_path in dataset_path:
        song_signal, song_sr = librosa.load(song_path, duration=None, sr=44100)
        song_duration = librosa.get_duration(y=song_signal, sr=song_sr)
        song_dictionary = create_dictionary(song_path, song_duration, song_signal, song_sr)

        mel_spectogram = extract_features(song_dictionary)

        folder_number = song_path.split("/")[-2]
        song_name = song_path.split("/")[-1].replace(".wav", "")

        path = os.path.join(getcwd(), 'datasets/train_spec/{}'.format(folder_number))
        if not os.path.exists(path):
            makedirs(path)

        file_path = path + "/mel_spectrogram_{}".format(song_name)

        np.save(file_path, mel_spectogram)

'''
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
# savefig('mel-espectograma_{}.png'.format(k)) # Guarda la foto con el nombre de path
'''


''' Genera el dataset con los paths de todas las canciones '''
def compute_dataset():
    song_paths = list_dataset('./datasets/train/')
    shuffle(song_paths)
    return song_paths


''' Función que crea una lista con la inversa del valor en dB de 1 a 10 '''
def target(vector):
    factor = []
    for db in vector:
        new = 1/math.pow(10, db/20)
        factor.append(new)
    return factor


''' Resta el db_level a la señal de música (audio) y el resultado se le suma a la señal de voz '''
def mix_signals(audio_dict, speech_dict, factor, db_level):

    audio = audio_dict['signal']
    audio *= factor

    speech = speech_dict['signal']
    signal = speech + audio

    mean = np.mean(signal)
    signal_db = 20 * math.log10(mean)

    new_mixed_song = {
        'db_level': db_level,
        'signal': signal,
        'sr': speech_dict['sr'],
        'signal_db': signal_db
    }
    return new_mixed_song


''' Función que normaliza los valores de las señales a -14 LUFS '''
def normalize(dictionary):
    signal = dictionary['signal']
    mean = np.mean(signal)

    # Si la media de la señal es negativa o cero, asignamos el valor 10e-6
    if mean <= 0:
        new_mean = 0.000001
        diff = new_mean - mean
        signal += diff
        mean = np.mean(signal)

    signal_db = 20 * math.log10(mean)

    # target_nu contiene el valor de referencia de -14 dB en unidades naturales: 0.1995262315
    diff_nu = abs(mean - FLAGS.target_nu)

    if FLAGS.target_level >= signal_db:
        signal += diff_nu
    else:
        signal -= diff_nu
    signal_db = 20 * math.log10(mean)

    signal_dictionary = {
        'sr': dictionary['sr'],
        'duration': dictionary['duration'],
        'signal': signal,
        'signal_db': signal_db
    }
    return signal_dictionary


''' Genera un dataset '''
def generate_dataset():
    all_audio_song_paths = list_dataset('./datasets/audio/')
    all_speech_song_paths = list_dataset('./datasets/speech/')

    # Vector de 1 a 10 para los dB
    vector_aux = list(range(1, 11, 1))
    targets = target(vector_aux)

    # Aleatoriza el orden de los segmentos de voz
    shuffle(all_speech_song_paths)

    # Comprueba si hay canciones de voz y de audio
    if (len(all_speech_song_paths) or len(all_audio_song_paths)) == 0:
        return

    aux = 0
    for k, audio_song_path in enumerate(all_audio_song_paths):

        # Carga la cancion y la duracion total
        audio_signal, audio_sr = librosa.load(audio_song_path,
                                              duration=None,
                                              sr=44100)

        audio_duration = librosa.get_duration(y=audio_signal,
                                              sr=audio_sr)

        # Comprueba si la duración del audio es menor que 5 y si lo es pasa a la siguiente cancion de audio
        if audio_duration < 5:
            # Pasa a la siguiente canción con música
            continue

        # Carga 5s de la señal de audio
        audio_signal, audio_sr = librosa.load(audio_song_path,
                                              duration=5,
                                              sr=44100)

        audio_duration = librosa.get_duration(y=audio_signal,
                                              sr=audio_sr)

        # Crea un dictionario con ciertas características de la señal
        audio_dictionary = create_dictionary(audio_song_path, audio_duration, audio_signal, audio_sr)

        # Coge una canción de la base de datos de voz y la quita de la lista (pop)
        speech_song_path = all_speech_song_paths.pop(0)

        # Carga la cancion y la duracion total
        speech_signal, speech_sr = librosa.load(speech_song_path,
                                                duration=None,
                                                sr=44100)

        speech_duration = librosa.get_duration(y=speech_signal,
                                               sr=speech_sr)

        exit_for = False
        # Si la voz dura menos de 5s, pasa a la siguiente
        while speech_duration < 5:
            # Comprueba si hay canciones de voz
            if not all_speech_song_paths:
                exit_for = True
                break
            speech_song_path = all_speech_song_paths.pop(0)
            speech_signal, speech_sr = librosa.load(speech_song_path,
                                                    duration=None,
                                                    sr=44100)

            speech_duration = librosa.get_duration(y=speech_signal,
                                                   sr=speech_sr)
        if exit_for:
            break

        # Carga 5s de la señal de voz
        speech_signal, speech_sr = librosa.load(speech_song_path,
                                                duration=5,
                                                sr=44100)

        speech_duration = librosa.get_duration(y=speech_signal,
                                               sr=speech_sr)

        # Crea un dictionario con ciertas características de la señal
        speech_dictionary = create_dictionary(speech_song_path, speech_duration, speech_signal, speech_sr)

        for db_level in vector_aux:

            audio_dict = normalize(audio_dictionary)

            speech_dict = normalize(speech_dictionary)

            new_mixed_song = mix_signals(audio_dict, speech_dict, targets[db_level-1], db_level)

            '''
            extract_features(audio_dict)
            extract_features(speech_dict)
            extract_features(new_mixed_song)
            '''

            path = os.path.join(getcwd(), 'datasets/train/', str(new_mixed_song['db_level']))
            if not os.path.exists(path):
                makedirs(path)
            librosa.output.write_wav(path + '/mix_' + str(k) + '.wav', new_mixed_song['signal'],
                                     new_mixed_song['sr'])

        aux += 1
        if aux % 100 == 0:
            total = len(vector_aux) * len(all_audio_song_paths)
            print('{} audios procesados de {}'.format(aux, total))
            if aux == len(all_audio_song_paths):
                songs_counter = len(vector_aux) * aux
                print('Proceso terminado. ¡Dataset de {} canciones creado con éxito!'.format(songs_counter))
                break


if __name__ == "__main__":

    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)

    start = time()
    cond = ask_generate_dataset()
    if cond:
        generate_dataset()
        dataset_path = compute_dataset()
        save_melgrams(dataset_path)

    # import pdb
    # show = pprint.PrettyPrinter(indent=4).pprint                                          # Función que imprime 'bonito'
    # pdb.set_trace()

    end = time()

    print('tiempo de ejecucion: {}'.format(end - start))  # Saber el tiempo de ejecución
    input('Pulsa enter para salir')

'''    
# with ThreadPoolExecutor(max_workers = 10) as executor:                    # Ejecuta la función generate_mixes con varios hilos en paralelo
#     tasks = executor.map(generate_mixes, song_and_speech_pairs)
#     done = 0
#     for task in tasks:
#         task.result()
#         done += 1
#         if done % 1000 == 0:
#             print("{} audios procesados de {}".format(done, zip_len))   
'''