import os
import numpy as np
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from common_flags import FLAGS
import pandas as pd
from utils import list_dataset, ask_generate_csv
import math


class DataGenerator(ImageDataGenerator):
    """
    Generate mini-batches of images and labels with real-time augmentation.
    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.
    """

    def flow_from_directory(self, directory, target_size=(96, 862), color_mode='rgb',
                            classes=None, class_mode='categorical', batch_size=32, shuffle=True,
                            seed=None, save_to_dir=False, save_prefix='', save_format='png',
                            follow_links=False, subset=None, interpolation='nearest'):
        return DirectoryIterator(directory, self, target_size=target_size,
                                 batch_size=batch_size, shuffle=shuffle, seed=seed, follow_links=follow_links)


class DirectoryIterator(Iterator):
    """
    Class for managing data loading of images and labels

    # Arguments
       phase: training, validation or test stage
       num_classes: Output dimension (number of classes).
       image_data_generator: Image Generator.
       target_size: tuple of integers, dimensions to resize input images to.
       batch_size: The desired batch size
       shuffle: Whether to shuffle data or not
       seed: numpy seed to shuffle data
       follow_links: Bool, whether to follow symbolic links or not
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(96, 862),
                 batch_size=32, shuffle=True, seed=None, follow_links=False):

        self.image_data_generator = image_data_generator
        self.target_size = target_size
        self.follow_links = follow_links

        # File of database for the phase
        if directory == 'train':
            csv_file = os.path.join(FLAGS.experiment_rootdir, 'train.csv')
        elif directory == 'val':
            csv_file = os.path.join(FLAGS.experiment_rootdir, 'validation.csv')
        else:
            csv_file = os.path.join(FLAGS.experiment_rootdir, 'test.csv')

        self.file_names, self.ground_truth = load_spectrograms(csv_file)

        # Number of samples in data
        self.samples = len(self.file_names)
        # Check if data is empty
        if self.samples == 0:
            raise IOError("Did not find any data")

        # print('Found {} images belonging to {} classes.'.format(
        #     self.samples, FLAGS.num_classes))

        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)


    def next(self):
        """
        Public function to fetch next batch
        # Returns: The next batch of images and commands.
        """
        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)


    def _get_batches_of_transformed_samples(self, index_array):
        """
        Public function to fetch next batch.
        Image transformation is not under thread lock, so it can be done in
        parallel
        # Returns: The next batch of images and categorical labels.
        """

        # Initialize batches and indexes
        batch_x, batch_y = [], []

        # Build batch of image data
        for i, j in enumerate(index_array):
            x = np.load(self.file_names[j])
            # Data augmentation
            # x = self.image_data_generator.random_transform(x)
            # x = self.image_data_generator.standardize(x)
            batch_x.append(x)
            batch_y.append(self.ground_truth[j])

        # batch_x = np.expand_dims(np.asarray(batch_x), axis=3)

        return batch_x, np.asarray(batch_y)


''' Funcion que divide la base de datos en training 80%, test 20%, validation 20% '''

def cross_val_create(dataset_path):
    data_path = list_dataset(dataset_path)

    aux1 = math.floor(0.8*len(data_path))

    train = data_path[:(aux1 - 1)]
    test = data_path[aux1:]

    aux2 = math.floor(0.8*len(train))

    validation = train[aux2:]
    train = train[:(aux2 - 1)]

    cond = ask_generate_csv()
    if cond:
        make_csv(train, 'training.csv')
        make_csv(validation, 'validation.csv')
        make_csv(test, 'test.csv')

    [training_songs_list, training_gt] = load_spectrograms('training.csv')
    [validation_songs_list, validation_gt] = load_spectrograms('validation.csv')
    [test_songs_list, test_gt] = load_spectrograms('test.csv')

    '''
    # File names, moments and labels of all samples in data.
    file_names = utils.file_to_list(os.path.join(path, 'data.txt'))
    labels = utils.file_to_list(os.path.join(path, 'labels.txt'))
    order = list(range(len(file_names)))
    sh(order)
    order = np.asarray(order)
    index4 = int(round(len(order) / 4))
    index2 = int(round(len(order) / 2))

    # Create files of directories, labels and moments
    utils.list_to_file([file_names[i] for i in order[index2:]],
                       os.path.join(FLAGS.experiment_rootdir, 'train_files.txt'))
    utils.list_to_file([file_names[i] for i in order[index4:index2]],
                       os.path.join(FLAGS.experiment_rootdir, 'val_files.txt'))
    utils.list_to_file([file_names[i] for i in order[0:index4]],
                       os.path.join(FLAGS.experiment_rootdir, 'test_files.txt'))
    utils.list_to_file([labels[i] for i in order[index2:]],
                       os.path.join(FLAGS.experiment_rootdir, 'train_labels.txt'))
    utils.list_to_file([labels[i] for i in order[index4:index2]],
                       os.path.join(FLAGS.experiment_rootdir, 'val_labels.txt'))
    utils.list_to_file([labels[i] for i in order[0:index4]],
                       os.path.join(FLAGS.experiment_rootdir, 'test_labels.txt'))
    return
    '''

def load_spectrograms(csv_file):    # Old cross_val_load()
    csv = pd.read_csv(csv_file)
    file_name_paths = csv.values[:, 1]
    ground_truth = csv.values[:, 2]

    song_list = []
    label_list = []

    for file_name_path in file_name_paths:
        loaded_song = np.load(file_name_path)
        song_list.append(loaded_song)

    for label in ground_truth:
        loaded_label = int(label)
        label_list.append(loaded_label)

    return song_list, label_list

''' Funcion que crea el CSV a través de los paths '''
def make_csv(paths, phase):
    file_path = []
    data_label = []

    for path in paths:
        label = path.split("/")[-2]
        file_path.append(path)
        data_label.append(label)

    data = {
        'file_path': file_path,
        'label': data_label
    }

    df = pd.DataFrame(data, columns=['file_path', 'label'])
    df.to_csv('{}'.format(phase))