# from concurrent.futures import ThreadPoolExecutor
import librosa
from os import scandir, getcwd
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

''' Mel-spectrogram parameters '''
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

''' Aplica el porcentaje que se le pase como parámetro a la señal de audio y el resultado se le suma a la señal de voz '''
def percentage(audio, speech, percentage):
    return speech + audio*(percentage/100)

''' Pregunta si se quiere generar un dataset '''
def ask_generate_dataset():
    answer = input("Generar base de datos? (s/n): ")
    while answer != "s" or answer != "n":
        answer = input("Error. Solo se admite 's' o 'n'")

    answer = True if answer == 's' else False
    return answer

''' Genera un dataset '''
def generate_dataset():
    # Lista con los paths de todos los directorios de AUDIO
    audio_folder_paths = ls_dir(ruta=getcwd() + '/datasets/audio/')
    # audio_folder_paths = sorted(audio_folder_paths)  # Ordena por numero las carpetas

    # Lista con los paths de todas las canciones de los directorios que haya (AUDIO)
    all_audio_song_paths = []
    for audio_folder_path in audio_folder_paths:
        audio_song_paths = ls_file(audio_folder_path)
        # audio_song_paths = sorted(audio_song_paths)  # Ordena por numero las canciones
        all_audio_song_paths.extend(audio_song_paths)  # Añade nuevas canciones a la lista de canciones

    # Lista con los paths de todos los directorios de VOZ
    speech_folder_paths = ls_dir(ruta=getcwd() + '/datasets/speech/')
    # speech_folder_paths = sorted(speech_folder_paths)  # Ordena por numero las carpetas

    # Lista con los paths de todas las canciones de los directorios que haya (VOZ)
    all_speech_song_paths = []
    for speech_folder_path in speech_folder_paths:
        speech_song_paths = ls_file(speech_folder_path)
        # speech_song_paths = sorted(speech_song_paths)  # Ordena por numero las canciones
        all_speech_song_paths.extend(speech_song_paths)  # Añade nuevas canciones a la lista de canciones

    # Comprobar que dataset_songs no esta vacio para que no se sobreescriban los datos

    dataset_songs = []
    vector_aux = range(0, 101)[::5]
    all_speech_song_paths = shuffle(all_speech_song_paths)
    contador_aux = 0
    for audio_song_path in all_audio_song_paths:

        if not all_speech_song_paths:
            break

        audio_signal, audio_sr = librosa.load(audio_song_path, duration=5, sr=44100)
        audio_duration = librosa.get_duration(y=audio_signal, sr=audio_sr)
        # audio_mel_spectogram = compute_mel_gram(audio_signal, audio_sr, 2.0, audio_duration)

        if audio_duration < 5:
            continue

        new_audio_song = {
            "path": audio_song_path,
            "duration": audio_duration,
            "signal": audio_signal,
            # "mel-spectogram": audio_mel_spectogram,
            "sample_rate": audio_sr,
            "RMS": np.mean(librosa.feature.rms(y=audio_signal)),
            # "normalized_value":
        }

        speech_song_path = all_speech_song_paths.pop(0)
        speech_signal, speech_sr = librosa.load(speech_song_path, duration=5, sr=44100)
        speech_duration = librosa.get_duration(y=speech_signal, sr=speech_sr)
        exit_for = False
        while speech_duration < 5:
            if not all_speech_song_paths:
                exit_for = True
                break
            speech_song_path = all_speech_song_paths.pop(0)
            speech_signal, speech_sr = librosa.load(speech_song_path, duration=5, sr=44100)
            speech_duration = librosa.get_duration(y=speech_signal, sr=speech_sr)
        # speech_mel_spectogram = compute_mel_gram(speech_signal, speech_sr, 2.0, speech_duration)
        if exit_for:
            break

        new_speech_song = {
            "path": speech_song_path,  # nueva_cancion["path"]
            "duration": speech_duration,
            "signal": speech_signal,
            # "mel-spectogram": speech_mel_spectogram,
            "sample_rate": speech_sr,
            "RMS": np.mean(librosa.feature.rms(y=speech_signal)),
            # "normalized_value":
        }

        for i in vector_aux:
            new_song = percentage(audio_signal, speech_signal, vector_aux[i])
            dataset_songs.append(new_song)  # Añade segmentos de 10 seg de audio al dataset que utilizaremos mas tarde

        contador_aux += 1
        if contador_aux == len(all_audio_song_paths):
            print('Proceso terminado. ¡Dataset de {} canciones creado con éxito!'.format(contador_aux))



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
'''

if __name__ == "__main__":
    start = time()

    if ask_generate_dataset():
        generate_dataset()

    # import pdb
    # show = pprint.PrettyPrinter(indent=4).pprint                                          # Función que imprime 'bonito'
    # pdb.set_trace()

    end = time()

    print("tiempo de ejecucion: {}".format(end-start))                                      # Saber el tiempo de ejecución

    input("Pulsa enter para salir")