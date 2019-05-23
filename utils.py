import numpy as np
import json
import os
from os.path import abspath
from os import scandir, getcwd
import librosa
import matplotlib.pyplot as plt
import itertools
from keras.utils.generic_utils import Progbar
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from random import shuffle


def compute_predictions_and_gt(model, generator, steps, verbose=1):
    """
    Generate predictions and associated ground truth for the input samples
    from a data generator. The generator should return the same kind of data as
    accepted by `predict_on_batch`.

    Function adapted from keras `predict_generator`.

    # Arguments
        model: Model instance containing the trained model.
        generator: Generator yielding batches of input samples.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_q_size: Maximum size for the generator queue.
        pickle_safe: If `True`, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        verbose: verbosity mode, 0 or 1.

    # Returns
        Numpy array(s) of predictions and associated ground truth.

    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    steps_done = 0
    all_outs = []
    all_steerings = []

    if verbose == 1:
        progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(generator)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, gt_steer = generator_output
            elif len(generator_output) == 3:
                x, gt_steer, _ = generator_output
            else:
                raise ValueError('output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))
        else:
            raise ValueError('Output not valid for current evaluation')

        outs = model.predict_on_batch(x)

        if not isinstance(outs, list):
            outs = [outs]
        if not isinstance(gt_steer, list):
            gt_steer = [gt_steer]

        if not all_outs:
            for out in outs:
                all_outs.append([])

        if not all_steerings:
            for steer in gt_steer:
                all_steerings.append([])

        for i, out in enumerate(outs):
            all_outs[i].append(out)

        for i, steer in enumerate(gt_steer):
            all_steerings[i].append(steer)

        steps_done += 1
        if verbose == 1:
            progbar.update(steps_done)

    if steps_done == 1:
        return [out for out in all_outs], [steer for steer in all_steerings]
    else:
        return np.squeeze(np.array([np.concatenate(out) for out in all_outs])), \
               np.squeeze(np.array([np.concatenate(steer) for steer in all_steerings]))


def compute_predictions(model, generator, steps, verbose=0):
    """
    Generate predictions and associated ground truth for the input samples
    from a data generator. The generator should return the same kind of data as
    accepted by `predict_on_batch`.

    Function adapted from keras `predict_generator`.

    # Arguments
        model: Model instance containing the trained model.
        generator: Generator yielding batches of input samples.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_q_size: Maximum size for the generator queue.
        pickle_safe: If `True`, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        verbose: verbosity mode, 0 or 1.

    # Returns
        Numpy array(s) of predictions

    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    steps_done = 0
    all_outs = []

    if verbose == 1:
        progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(generator)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, _ = generator_output
            elif len(generator_output) == 3:
                x, _, _ = generator_output
            else:
                raise ValueError('output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))
        else:
            x = generator_output
            # raise ValueError('Output not valid for current evaluation')

        outs = model.predict_on_batch(x)

        if not isinstance(outs, list):
            outs = [outs]

        if not all_outs:
            # Len of this list is related to the number of
            # outputs per model(1 in our case)
            for out in outs:
                all_outs.append([])

        for i, out in enumerate(outs):
            all_outs[i].append(out)

        steps_done += 1
        if verbose == 1:
            progbar.update(steps_done)

    if steps_done == 1:
        return [out for out in all_outs]
    else:
        return np.squeeze(np.array([np.concatenate(out) for out in all_outs]))


def model_to_json(model, json_model_path):
    """
    Serialize model into json.
    """
    model_json = model.to_json()

    with open(json_model_path, "w") as f:
        f.write(model_json)


def json_to_model(json_model_path):
    """
    Serialize json into model.
    """
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    return model


def list_to_file(data, f_name):
    with open(f_name, 'w') as f:
        for item in data:
            f.write("%s\n" % item)


def file_to_list(f_name):
    ret_data = []
    with open(f_name, 'r') as f:
        data = f.readlines()
    for item in data:
        ret_data.append(item.split('\n')[0])
    for item in ret_data:
        if item == "" or None:
            ret_data.remove(item)
    return ret_data


def write_to_file(dictionary, f_name):
    """
    Writes everything is in a dictionary in json model.
    """
    with open(f_name, "w") as f:
        json.dump(dictionary, f)
        print("Written file {}".format(f_name))


def plot_loss(path_to_log):
    """
    Read log file and plot losses.

    # Arguments
        path_to_log: Path to log file.
    """
    # Read log file
    log_file = os.path.join(path_to_log, "log.txt")
    try:
        log = np.genfromtxt(log_file, delimiter='\t', dtype=None, names=True)
    except ImportError:
        raise IOError("Log file not found")

    train_loss = log['train_loss']
    val_loss = log['val_loss']
    time_steps = list(range(train_loss.shape[0]))

    # Plot losses
    plt.plot(time_steps, train_loss, 'r--', time_steps, val_loss, 'b--')
    plt.legend(["Training loss", "Validation loss"])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(path_to_log, "log.png"))


def plot_confusion_matrix(phase, path_to_results, real_labels, pred_labels, classes,
                          normalize=True):
    """
    Plot and save confusion matrix computed from predicted and real labels.

    # Arguments
        path_to_results: Location where saving confusion matrix.
        real_labels: List of real labels.
        pred_prob: List of predicted probabilities.
        normalize: Boolean, whether to apply normalization.
    """

    # Generate confusion matrix
    cm = confusion_matrix(real_labels, pred_labels)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, float('%.3f' % (cm[i, j])),
                 horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if phase == 'test':
        plt.savefig(os.path.join(path_to_results, "confusion.png"))
    elif phase == 'outer_test':
        plt.savefig(os.path.join(path_to_results, "outer_confusion.png"))


''' Funcion que devuelve carpetas con directorios '''
def ls_dir(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if not arch.is_file()]


''' Funcion que devuelve carpetas con archivos '''
def ls_file(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]


''' Funcion que lista carpetas y ficheros '''
def list_dataset(path):
    all_song_paths = []
    folder_paths = ls_dir(path)                     # Lista sin orden con los paths de todos los directorios
    for folder_path in folder_paths:
        song_paths = ls_file(folder_path)
        all_song_paths.extend(song_paths)           # Añade nuevos paths de canciones a la lista de paths de canciones
    # shuffle(all_song_paths)
    return all_song_paths


''' Pregunta si se quiere generar un dataset '''
def ask_create_model():
    answer = input("¿Crear modelo desde cero? (y/n): ")
    if answer == 'y':
        ret = True
    elif answer == 'n':
        ret = False
    else:
        ret = False
        print(input("Error. Solo se admite 'y' o 'n'"))
    return ret


''' Pregunta si se quiere generar un dataset '''
def ask_generate_dataset():
    answer = input("¿Crear base de datos? (y/n): ")
    if answer == 'y':
        ret = True
    elif answer == 'n':
        ret = False
    else:
        ret = False
        print(input("Error. Solo se admite 'y' o 'n'"))
    return ret


''' Pregunta si se quiere generar un dataset '''
def ask_generate_csv():
    answer = input("¿Crear csv? (y/n): ")
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
        'path': song_path,
        'duration': duration,
        'signal': signal,
        'sr': sr
    }
    return new_song


''' Funcion que devuelve el mel-espectrograma de una cancion '''
def extract_features(dict):
    y = dict['signal']
    sr = dict['sr']
    duration = dict['duration']
    mel_spectrogram = compute_mel_gram(duration, sr, 2.0, y)

    '''
    plt.figure(figsize=(10, 4))
    # x = librosa.power_to_db(mel_spectrogram, ref=np.max)
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), sr=44100, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    '''

    return mel_spectrogram


''' Funcion que genera el mel-espectrograma de la cancion que se le pasa como parametro '''
def compute_mel_gram(separation, sr, power, segment):
    n_fft = 512
    n_mel = 96
    hop_len = 256
    n_sample = segment.shape[0]
    n_sample_fit = int(separation*sr)

    if n_sample < n_sample_fit:  # if too short
        src = np.concatenate([segment, np.zeros((int(separation*sr) - n_sample,))])
    elif n_sample > n_sample_fit:  # if too long
        src = segment[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    else:
        src = segment
    mel = librosa.feature.melspectrogram(
            y=src, sr=sr, hop_length=hop_len,
            n_fft=n_fft, n_mels=n_mel, power=power)

    ret = librosa.power_to_db(mel)
    return ret
