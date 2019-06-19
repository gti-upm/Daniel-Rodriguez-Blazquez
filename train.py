from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as k
import os
import numpy as np
import sys
import gflags
from . import __name__
import data_utils
import logz
import log_utils
from common_flags import FLAGS
import utils
from utils import json_to_model, model_to_json
from time import time, strftime, localtime

TRAIN_PHASE = 1


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 10, 15, 20, 25 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = FLAGS.initial_lr
    if epoch > 150:
        lr *= 0.5e-3
    elif epoch > 100:
        lr *= 1e-2
    elif epoch > 50:
        lr *= 1e-1
    elif epoch > 0:
        lr *= 1
    print('Learning rate: ', lr)
    return lr

def train_model(train_generator, val_generator, model, initial_epoch):

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])

    # Save model with the lowest validation loss
    weights_path = os.path.join(FLAGS.experiment_rootdir, 'weights_{epoch:03d}.h5')
    write_best_model = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                       save_best_only=True, save_weights_only=True)

    # Save training and validation losses.
    logz.configure_output_dir(FLAGS.experiment_rootdir)
    save_model_and_loss = log_utils.MyCallback(filepath=FLAGS.experiment_rootdir)

    # Train model
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=FLAGS.verbose)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   verbose=FLAGS.verbose,
                                   min_lr=0.5e-6)
    # earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=FLAGS.verbose)

    str_time = strftime("%Y%b%d_%Hh%Mm%Ss", localtime(time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(str_time), histogram_freq=0)

    callbacks = [write_best_model, save_model_and_loss, lr_reducer, lr_scheduler, tensorboard]

    model.fit_generator(train_generator, validation_data=val_generator,
                        epochs=FLAGS.epochs,
                        verbose=FLAGS.verbose,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch,
                        use_multiprocessing=True)


def _main():

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
    train_data_gen = data_utils.DataGenerator()

    # Iterator object containing training data to be generated batch by batch
    train_generator = train_data_gen.flow_from_directory('train',
                                                         shuffle=True,
                                                         target_size=(img_height, img_width),
                                                         classes=FLAGS.num_classes,
                                                         batch_size=FLAGS.batch_size)

    # Generate validation data with real-time augmentation
    val_data_gen = data_utils.DataGenerator()

    # Iterator object containing validation data to be generated batch by batch
    val_generator = val_data_gen.flow_from_directory('val',
                                                     shuffle=False,
                                                     target_size=(img_height, img_width),
                                                     classes=FLAGS.num_classes,
                                                     batch_size=FLAGS.batch_size)

    # Check if the number of classes in data corresponds to the one specified
    assert train_generator.num_classes == FLAGS.num_classes, \
        " Not matching output dimensions in training data."

    # Check if the number of classes in data corresponds to the one specified
    assert val_generator.num_classes == FLAGS.num_classes, \
        " Not matching output dimensions in validation data."

    # Weights to restore
    weights_path = FLAGS.initial_weights

    # Epoch from which training starts
    initial_epoch = FLAGS.initial_epoch

    if FLAGS.restore_model:
        # In this case weights are initialized as specified in pre-trained model
        # initial_epoch = FLAGS.initial_epoch
        try:
            # Carga estructura de la red
            json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
            model = json_to_model(json_model_path)

            # Carga los pesos
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except ImportError:
            print("Impossible to find weight path. Returning untrained model")
    else:
        # In this case weights are initialized randomly
        weights_path = None

        # Define model
        bot_model = InceptionV3(weights=None, include_top=False,
                                input_shape=[img_height, img_width, 1],
                                classes=train_generator.num_classes)
        bot_model.summary()
        input = Input(shape=[img_height, img_width, 1])
        top = bot_model(input)

        # intermediate = Dropout()(top)
        # top = Flatten()(intermediate)
        top = Flatten()(top)
        top = Dense(FLAGS.num_classes, activation='softmax', name='predictions')(top)
        model = Model(inputs=input, outputs=top)

    model.summary()

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model_to_json(model, json_model_path)

    # Train model
    train_model(train_generator, val_generator, model, initial_epoch)

    # Plot training and validation losses
    utils.plot_loss(FLAGS.experiment_rootdir)


if __name__ == "__main__":

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
    
    # Set random seed
    if FLAGS.random_seed:
        seed = np.random.randint(0, 2 * 31 - 1)
    else:
        seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    # Define model
    cond = ask_create_model()
    if cond:
        bot_model = InceptionV3(weights=None, include_top=False,
                            input_shape=[img_height, img_width, 1],
                            classes=train_generator.num_classes)
        input = Input(shape=[img_height, img_width, 1])
        top = bot_model(input)
        top = Flatten()(top)
        top = Dense(FLAGS.num_classes, activation='softmax', name='predictions')(top)
        model = Model(inputs=input, outputs=top)
    else:
        if weights_path:
            try:
                json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
                model = json_to_model(json_model_path)

                model.load_weights(weights_path)
                print("Loaded model from {}".format(weights_path))
            except ImportError:
                print("Impossible to find weight path. Returning untrained model")
    
'''
