# from concurrent.futures import ThreadPoolExecutor
import librosa
from os import scandir, getcwd
from os.path import abspath

def ls_dir(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if not arch.is_file()]
def ls_file(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]


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

contenido = ls_dir(ruta=getcwd()+'/datasets/fma_small/')
contenido = sorted(contenido)

todas_las_canciones = []
for carpeta in contenido:
    canciones = ls_file(carpeta)
    canciones = sorted(canciones)
    todas_las_canciones.extend(canciones)
    # librosa.get_duration(filename=librosa.util.example_audio_file())
canciones_con_datos = []
from time import time
start = time()
for song_path in todas_las_canciones:
    if librosa.get_duration(y=y, sr=44100):
        y, sr = librosa.load(song_path, duration=10, sr=44100)

        nueva_cancion = {
            "path": song_path,
            "duracion": librosa.get_duration(y=y, sr=sr)
            # "RMS": librosa.feature.rms(S=librosa.magphase(librosa.stft(y))[1])
         }
        # nueva_cancion["path"]
        canciones_con_datos.append(nueva_cancion)

    else:
        continue

end = time()
print("tiempo de ejecucion: {}".format(end-start))

input("Pulsa enter para salir")