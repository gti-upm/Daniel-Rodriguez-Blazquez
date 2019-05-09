# from concurrent.futures import ThreadPoolExecutor
import librosa
from os import scandir, getcwd, makedirs
import os
from os.path import abspath
from time import time
# import pprint
import numpy as np
from random import shuffle
import math



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


''' Funcion que devuelve carpetas con directorios '''
def ls_dir(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if not arch.is_file()]


''' Funcion que devuelve carpetas con archivos '''
def ls_file(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]


''' Aplica el porcentaje que se le pase como parámetro a la señal de audio y el resultado se le suma a la señal de voz '''
def mix_signals(diff, audio, speech, speech_sr, percentage):
    audio = audio + diff
    signal = speech + audio * (percentage / 100)
    new_mixed_song = {
        "percentage": str(percentage),
        "signal": signal,
        "sample_rate": speech_sr
    }
    return new_mixed_song


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


''' Función que ajusta el valor RMS de la música al de la voz '''
def adjust_rms(audio_rms, speech_rms):
    diff = math.sqrt(abs(pow(speech_rms, 2) - pow(audio_rms, 2)))
    audio_rms = audio_rms + diff if audio_rms < speech_rms else audio_rms - diff
    return audio_rms                                                                    # ---------------------------


''' Genera un dataset '''
def generate_dataset():
    audio_folder_paths = ls_dir(ruta=getcwd() + '/datasets/audio/')                     # Lista sin orden con los paths de todos los directorios de AUDIO
    all_audio_song_paths = []                                                           # Lista sin orden con los paths de todas las canciones (AUDIO)
    for audio_folder_path in audio_folder_paths:
        audio_song_paths = ls_file(audio_folder_path)
        all_audio_song_paths.extend(audio_song_paths)                                   # Añade nuevas canciones a la lista de canciones

    speech_folder_paths = ls_dir(ruta=getcwd() + '/datasets/speech/english/')           # Lista sin orden con los paths de todos los directorios de VOZ
    all_speech_song_paths = []                                                          # Lista sin orden con los paths de todas las canciones (VOZ)
    for speech_folder_path in speech_folder_paths:
        speech_song_paths = ls_file(speech_folder_path)
        all_speech_song_paths.extend(speech_song_paths)                                 # Añade nuevas canciones a la lista de canciones

    vector_aux = list(range(0, 101, 5))
    shuffle(all_speech_song_paths)                                                      # Aleatoriza el orden de los segmentos de voz

    if (len(all_speech_song_paths) or len(all_audio_song_paths)) == 0:                    # Comprueba si hay canciones de voz y de audio
        return

    aux = 0
    for k, audio_song_path in enumerate(all_audio_song_paths):

        audio_signal, audio_sr = librosa.load(audio_song_path, duration=None, sr=44100) # Carga en audio_signal la duracion total de la cancion ------------------
        audio_duration = librosa.get_duration(y=audio_signal, sr=audio_sr)

        if audio_duration < 5:                                                          # Comprueba si la duración del audio es menor que 5 y si lo es pasa a la siguiente cancion de audio
            continue

        audio_signal, audio_sr = librosa.load(audio_song_path, duration=5, sr=44100)    # Carga 5s de audio despues de comprobar que la duracion es mayor que 5
        audio_duration = librosa.get_duration(y=audio_signal, sr=audio_sr)
        audio_dictionary = create_dictionary(audio_song_path, audio_duration, audio_signal, audio_sr)

        speech_song_path = all_speech_song_paths.pop(0)
        speech_signal, speech_sr = librosa.load(speech_song_path, duration=None, sr=44100)
        speech_duration = librosa.get_duration(y=speech_signal, sr=speech_sr)

        exit_for = False
        while speech_duration < 5:                                                      # Si la voz dura menos de 5s, pasa a la siguiente
            if not all_speech_song_paths:                                               # Comprueba si hay canciones de voz
                exit_for = True
                break
            speech_song_path = all_speech_song_paths.pop(0)
            speech_signal, speech_sr = librosa.load(speech_song_path, duration=None, sr=44100)
            speech_duration = librosa.get_duration(y=speech_signal, sr=speech_sr)
        if exit_for:
            break

        speech_signal, speech_sr = librosa.load(speech_song_path, duration=5, sr=44100)
        speech_duration = librosa.get_duration(y=speech_signal, sr=speech_sr)
        speech_dictionary = create_dictionary(speech_song_path, speech_duration, speech_signal, speech_sr)

        for i, percentage_level in enumerate(vector_aux):

            diff = adjust_rms(audio_dictionary['RMS'], speech_dictionary['RMS'])

            new_mixed_song = mix_signals(diff, audio_signal, speech_signal, speech_sr, percentage_level)

            path = os.path.join(getcwd(), 'datasets/train/', new_mixed_song['percentage'])
            if not os.path.exists(path):
                makedirs(path)
            if not os.path.exists(path + '/mix_' + str(k) + '.wav'):
                librosa.output.write_wav(path + '/mix_' + str(k) + '.wav', new_mixed_song['signal'], new_mixed_song['sample_rate'])
            librosa.output.write_wav(path + '/mix_' + str(k) + '.wav', new_mixed_song['signal'], new_mixed_song['sample_rate'])

        aux += 1
        if aux % 100 == 0:
            total = len(vector_aux)*len(all_audio_song_paths)
            print("{} audios procesados de {}".format(aux, total))
            if aux == len(all_audio_song_paths):
                songs_counter = len(vector_aux)*aux
                print('Proceso terminado. ¡Dataset de {} canciones creado con éxito!'.format(songs_counter))
                break


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


Función que genera las canciones mezcladas
def generate_mixes(audio_path, speech_path):
    vector_aux = list(range(0, 101, 5))
    audio_signal, audio_sr = librosa.load(audio_path, duration=None, sr=44100)      #----------------------
    audio_duration = librosa.get_duration(y=audio_signal, sr=audio_sr)

    if audio_duration < 5:
        return

    audio_dictionary = create_dictionary(audio_path, audio_duration, audio_signal, audio_sr)

    speech_path = speech_path
    speech_signal, speech_sr = librosa.load(speech_path, duration=None, sr=44100)   #----------------------
    speech_duration = librosa.get_duration(y=speech_signal, sr=speech_sr)

    if speech_duration < 5:
        return

    speech_dictionary = create_dictionary(speech_path, speech_duration, speech_signal, speech_sr)

    for percentage in vector_aux:
        audio_adjusted = adjust_rms(audio_dictionary['RMS'], speech_dictionary['RMS'])
        new_mixed_song = mix_signals(audio_adjusted, speech_signal, speech_sr, percentage)  # -----------------
        path = os.path.join(getcwd(), 'datasets/train/', new_mixed_song['percentage'])
        if not os.path.exists(path):
            makedirs(path)
        file_name = audio_path.split("/")[-1].split(".")[0]
        file_path = "{}/mix_{}.wav".format(path, file_name)
        librosa.output.write_wav(file_path, new_mixed_song['signal'], new_mixed_song['sample_rate'])
    return None 
    
#   song_and_speech_pairs = zip(all_audio_song_paths, all_speech_song_paths)    # Forma parejas con los paths de voz y audio------------------------------------
#   print('Number of audio songs: ' + str(len(all_audio_song_paths)))
#   print('Number of voice songs: ' + str(len(all_speech_song_paths)))
#   zip_len = min(len(all_audio_song_paths), len(all_speech_song_paths))
#   print('Number of combined segments: ' + str(zip_len))
#   aux = 0
    
# with ThreadPoolExecutor(max_workers = 10) as executor:                    # Ejecuta la función generate_mixes con varios hilos en paralelo
#     tasks = executor.map(generate_mixes, song_and_speech_pairs)
#     done = 0
#     for task in tasks:
#         task.result()
#         done += 1
#         if done % 1000 == 0:
#             print("{} audios procesados de {}".format(done, zip_len))   
    
#   for pair in song_and_speech_pairs:
#        generate_mixes(pair)
#       aux += 1
#       if aux % 1000 == 0:
#           print("{} audios procesados de {}".format(aux, zip_len))
    
'''