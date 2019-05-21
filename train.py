from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import backend as k
import librosa
import librosa.display
from os import getcwd, makedirs
import os
import numpy as np
import sys
import gflags
from keras.callbacks import TensorBoard
from . import __name__
import data_utils
import logz
import log_utils
from common_flags import FLAGS
import utils
from utils import ls_dir, ls_file, list_dataset, create_dictionary, ask_generate_spectrograms, extract_features
from keras.callbacks import ReduceLROnPlateau
from time import time, strftime, localtime

TRAIN_PHASE = 1


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


''' Genera el dataset con los paths de todas las canciones ''' #-----------Descomentar----
def compute_dataset():
    # song_paths = list_dataset('/datasets/train/')
    # shuffle(song_paths)
    song_paths = ls_file(ruta=getcwd() + '/datasets/train/5/')

    return song_paths


def train_model(train_generator, val_generator, model, initial_epoch):
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Save model with the lowest validation loss
    weights_path = os.path.join(FLAGS.experiment_rootdir, 'weights_{epoch:03d}.h5')
    write_best_model = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                       save_best_only=True, save_weights_only=True)

    # Save training and validation losses.
    logz.configure_output_dir(FLAGS.experiment_rootdir)
    save_model_and_loss = log_utils.MyCallback(filepath=FLAGS.experiment_rootdir)

    # Train model
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    str_time = strftime("%Y%b%d_%Hh%Mm%Ss", localtime(time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(str_time), histogram_freq=0)
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    callbacks = [write_best_model, save_model_and_loss, lr_reducer, tensorboard]

    model.fit_generator(train_generator, val_generator,
                        epochs=FLAGS.epochs,
                        verbose=FLAGS.verbose,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch)


def _main():
    '''
    # Set random seed
    if FLAGS.random_seed:
        seed = np.random.randint(0, 2 * 31 - 1)
    else:
        seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)
    '''

    # Set training phase
    k.set_learning_phase(TRAIN_PHASE)

    # Create the experiment rootdir if not already there:
    if not os.path.exists(FLAGS.experiment_rootdir):
        os.makedirs(FLAGS.experiment_rootdir)

    # Split the data into training, validation and test sets
    if FLAGS.initial_epoch == 0:
        data_utils.cross_val_create(FLAGS.data_path)

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height
    print('Tamaño de los mel-espectrogramas: Alto: {}, Ancho: {}'.format(img_height, img_width))

    # Generate training data with real-time augmentation
    train_data_gen = data_utils.DataGenerator(rescale=1. / 255)

    # Iterator object containing training data to be generated batch by batch
    train_generator = train_data_gen.flow_from_directory('train',
                                                         shuffle=True,
                                                         target_size=(img_height, img_width),
                                                         batch_size=FLAGS.batch_size)

    # Generate validation data with real-time augmentation
    val_data_gen = data_utils.DataGenerator(rescale=1. / 255)

    # Iterator object containing validation data to be generated batch by batch
    val_generator = val_data_gen.flow_from_directory('val',
                                                     shuffle=False,
                                                     target_size=(img_height, img_width),
                                                     batch_size=FLAGS.batch_size)

    # Check if the number of classes in data corresponds to the one specified
    # assert train_generator.num_classes == FLAGS.num_classes, \
        # " Not matching output dimensions in training data."

    # Check if the number of classes in data corresponds to the one specified
    # assert val_generator.num_classes == FLAGS.num_classes, \
        # " Not matching output dimensions in validation data."

    # Weights to restore
    weights_path = FLAGS.initial_weights

    # Epoch from which training starts
    initial_epoch = FLAGS.initial_epoch
    if not FLAGS.restore_model:
        # In this case weights are initialized randomly
        weights_path = None
    else:
        # In this case weights are initialized as specified in pre-trained model
        initial_epoch = FLAGS.initial_epoch

    # Define model
    model = InceptionV3(weights=None, include_top=False, input_shape=[img_height, img_width], classes=1)

    model.summary()

    # scores = model.evaluate(spectrogram_dataset, target_data)
    # print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
    # print(model.predict(spectogram_dataset).round())

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    utils.model_to_json(model, json_model_path)

    # Train model
    train_model(train_generator, val_generator, model, initial_epoch)

    # Plot training and validation losses
    utils.plot_loss(FLAGS.experiment_rootdir)


if __name__ == "__main__":

    cond = ask_generate_spectrograms()
    if cond:
        dataset_path = compute_dataset()
        save_melgrams(dataset_path)

    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    _main()


'''
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
