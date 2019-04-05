# from concurrent.futures import ThreadPoolExecutor
import librosa
from os import scandir, getcwd, makedirs
import os
from os.path import abspath
from time import time
import pprint
import numpy as np
from random import shuffle

''' Funcion que devuelve carpetas con directorios '''
def ls_dir(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if not arch.is_file()]


''' Funcion que devuelve carpetas con archivos '''
def ls_file(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]


''' Aplica el porcentaje que se le pase como parámetro a la señal de audio y el resultado se le suma a la señal de voz '''
def percentage(audio, speech, speech_sr, percentage):
    signal = speech + audio * (percentage / 100)
    new_measured_song = {
        "percentage": str(percentage),
        "signal": signal,
        "sample_rate": speech_sr
    }

    return new_measured_song


''' Pregunta si se quiere generar un dataset '''
def ask_generate_dataset():
    answer = input("Generar base de datos? (y/n): ")
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


''' Genera un dataset '''
def generate_dataset():
    audio_folder_paths = ls_dir(ruta=getcwd() + '/datasets/audio/')     # Lista sin orden con los paths de todos los directorios de AUDIO
    all_audio_song_paths = []                                           # Lista sin orden con los paths de todas las canciones (AUDIO)
    for audio_folder_path in audio_folder_paths:
        audio_song_paths = ls_file(audio_folder_path)
        all_audio_song_paths.extend(audio_song_paths)                   # Añade nuevas canciones a la lista de canciones

    speech_folder_paths = ls_dir(ruta=getcwd() + '/datasets/speech/')   # Lista sin orden con los paths de todos los directorios de VOZ
    all_speech_song_paths = []                                          # Lista sin orden con los paths de todas las canciones (VOZ)
    for speech_folder_path in speech_folder_paths:
        speech_song_paths = ls_file(speech_folder_path)
        all_speech_song_paths.extend(speech_song_paths)                 # Añade nuevas canciones a la lista de canciones

    # dataset_songs = []
    vector_aux = list(range(0, 101, 5))
    shuffle(all_speech_song_paths)                                      # Aleatoriza el orden de los segmentos de voz
    counter_aux = 0
    for k, audio_song_path in enumerate(all_audio_song_paths):

        if len(all_speech_song_paths) == 0:                                   # Comprueba si hay canciones de voz
            break

        audio_signal, audio_sr = librosa.load(audio_song_path, duration=5, sr=44100)
        audio_duration = librosa.get_duration(y=audio_signal, sr=audio_sr)

        if audio_duration < 5:
            continue

        audio_dictionary = create_dictionary(audio_song_path, audio_duration, audio_signal, audio_sr)

        speech_song_path = all_speech_song_paths.pop(0)
        speech_signal, speech_sr = librosa.load(speech_song_path, duration=5, sr=44100)
        speech_duration = librosa.get_duration(y=speech_signal, sr=speech_sr)

        exit_for = False
        while speech_duration < 5:                                      # Si la voz dura menos de 5s, pasa a la siguiente
            if not all_speech_song_paths:                               # Comprueba si hay canciones de voz
                exit_for = True
                break
            speech_song_path = all_speech_song_paths.pop(0)
            speech_signal, speech_sr = librosa.load(speech_song_path, duration=5, sr=44100)
            speech_duration = librosa.get_duration(y=speech_signal, sr=speech_sr)
        if exit_for:
            break

        speech_dictionary = create_dictionary(speech_song_path, speech_duration, speech_signal, speech_sr)

        for i, percentage_level in enumerate(vector_aux):
            audio_dictionary['RMS'] = speech_dictionary['RMS']
            # -----------------
            new_measured_song = percentage(audio_signal, speech_signal, speech_sr, percentage_level)  # -----------------
            path = os.path.join(getcwd(), 'datasets/intro/', new_measured_song['percentage'])
            # path, file = path.split(str_tmp)
            if not os.path.exists(path):
                makedirs(path)
            # dataset_songs.append(new_measured_song)                              # Añade las mezclas al dataset que utilizaremos mas tarde
            librosa.output.write_wav(path + '/mix_'+str(k)+'.wav', new_measured_song['signal'], new_measured_song['sample_rate'])

    counter_aux += 1
    if counter_aux == len(all_audio_song_paths):
        songs_counter = len(vector_aux)*counter_aux
        print('Proceso terminado. ¡Dataset de {} canciones creado con éxito!'.format(songs_counter))

if __name__ == "__main__":
    start = time()
    cond = ask_generate_dataset()
    if cond:
        generate_dataset()

    # import pdb
    # show = pprint.PrettyPrinter(indent=4).pprint                                          # Función que imprime 'bonito'
    # pdb.set_trace()

    end = time()

    print("tiempo de ejecucion: {}".format(end-start))                                      # Saber el tiempo de ejecución

    input("Pulsa enter para salir")

'''
def haz_cosas(image):
    return 4+4
if __name__ == "__main__":
    images = list()
    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = list()
        for image in images:
            tasks.append(executor.submit(image, haz_cosas))
        contador = 0
        for task in tasks:
            task.result()
            print("Llevo {} imagenes".format(contador))
            contador += 1
        executor.map(images, haz_cosas)
    for image in images:
        haz_cosas(image)


new_audio_song = {
        "path": audio_song_path,
        "duration": audio_duration,
        "signal": audio_signal,
        # "mel-spectogram": audio_mel_spectogram,
        "sample_rate": audio_sr,
        "RMS": np.mean(librosa.feature.rms(y=audio_signal)),
        # "normalized_value":
    }

new_speech_song = {
            "path": speech_song_path,  # nueva_cancion["path"]
            "duration": speech_duration,
            "signal": speech_signal,
            # "mel-spectogram": speech_mel_spectogram,
            "sample_rate": speech_sr,
            "RMS": np.mean(librosa.feature.rms(y=speech_signal)),
            # "normalized_value":
        }

audio_folder_paths = sorted(audio_folder_paths)  # Ordena por numero las carpetas
audio_song_paths = sorted(audio_song_paths)  # Ordena por numero las canciones

speech_folder_paths = sorted(speech_folder_paths)  # Ordena por numero las carpetas
speech_song_paths = sorted(speech_song_paths)  # Ordena por numero las canciones

# Mel-spectrogram parameters
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

audio_mel_spectogram = compute_mel_gram(audio_signal, audio_sr, 2.0, audio_duration)
speech_mel_spectogram = compute_mel_gram(speech_signal, speech_sr, 2.0, speech_duration)
'''