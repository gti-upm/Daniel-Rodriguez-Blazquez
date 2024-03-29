import gflags
from common_flags import FLAGS
import sys
import numpy as np
import os
from sklearn import metrics
from keras import backend as k
import utils
import data_utils
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, accuracy_score
'''
# import the pyplot and wavfile modules
import matplotlib.pyplot as plot
from scipy.io import wavfile
from os import getcwd, makedirs
'''

TEST_PHASE = 1
CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def compute_highest_classification_errors(pred_probs, real_labels, n_errors=20):
    """
    Compute the 'n_errors' highest errors predicted by the network

    # Arguments
       pred_probs: predicted probabilities by the network.
       real_labels: real labels (ground truth).
       n_errors: Number of samples with highest error to be returned.

    # Returns
       highest_errors: Indexes of the samples with highest errors.
    """
    assert np.all(pred_probs.shape == real_labels.shape)
    dist = abs(pred_probs - 1)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors


def evaluate_classification(pred_probs, pred_labels, real_labels):
    """
    Evaluate some classification metrics. Compute average accuracy and highest
    errors.
    # Arguments
       pred_probs: predicted probabilities by the network.
       pred_labels: predicted labels by the network.
       real_labels: real labels (ground truth).
    # Returns
       dictionary: dictionary containing the evaluated classification metrics
    """
    # Compute average accuracy
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)

    # Compute highest errors
    highest_errors = compute_highest_classification_errors(pred_probs, real_labels, n_errors=20)

    # Return accuracy and highest errors in a dictionary
    dictionary = {"ave_accuracy": ave_accuracy,
                  "highest_errors": highest_errors.tolist()}
    return dictionary


def _main():
    # Set testing mode (dropout/batch normalization)
    k.set_learning_phase(TEST_PHASE)

    # Split the data into training, validation and test sets
    if FLAGS.initial_epoch == 0:
        data_utils.cross_val_create(FLAGS.data_path)

    # Generate testing data
    test_data_gen = data_utils.DataGenerator()

    # Iterator object containing testing data to be generated batch by batch
    test_generator = test_data_gen.flow_from_directory('test',
                                                       shuffle=False,
                                                       target_size=(FLAGS.img_height, FLAGS.img_width),
                                                       batch_size=FLAGS.batch_size)

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.json_to_model(json_model_path)

    # Load weights
    weights_load_path = os.path.abspath('./experiment_6/weights_039.h5')
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except ImportError:
        print("Impossible to find weight path. Returning untrained model")

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Get predictions and ground truth
    n_samples = test_generator.samples
    nb_batches = int(np.ceil(n_samples / FLAGS.batch_size))
    probs_per_class, ground_truth = utils.compute_predictions_and_gt(model, test_generator, nb_batches, verbose=FLAGS.verbose)

    # Predicted probabilities
    pred_probs = np.max(probs_per_class, axis=-1)
    # Predicted labels
    pred_labels = np.argmax(probs_per_class, axis=-1)
    # Real labels (ground truth)
    real_labels = np.argmax(ground_truth, axis=-1)

    # Evaluate predictions: Average accuracy and highest errors
    print("-----------------------------------------------")
    print("Evaluation:")
    evaluation = evaluate_classification(pred_probs, pred_labels, real_labels)
    print("-----------------------------------------------")

    # Save evaluation
    utils.write_to_file(evaluation, os.path.join(FLAGS.experiment_rootdir, 'test_results.json'))

    # Save predicted and real steerings as a dictionary
    labels_dict = {'pred_labels': pred_labels.tolist(),
                   'real_labels': real_labels.tolist()}
    utils.write_to_file(labels_dict, os.path.join(FLAGS.experiment_rootdir,
                                                  'predicted_and_real_labels.json'))

    # Visualize confusion matrix
    utils.plot_confusion_matrix('test', FLAGS.experiment_rootdir, real_labels,
                                pred_labels, CLASSES, normalize=True)

    print('Accuracy:', accuracy_score(real_labels, pred_labels))
    print('F1 score:', f1_score(real_labels, pred_labels, average='micro'))
    print('Recall:', recall_score(real_labels, pred_labels, average='micro'))
    print('Precision:', precision_score(real_labels, pred_labels, average='micro'))
    print('\n clasification report:\n', classification_report(real_labels, pred_labels))
    print('\n confussion matrix:\n', confusion_matrix(real_labels, pred_labels))



if  __name__ == "__main__":

    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)

    '''
    # Read the wav file (mono)
    samplingFrequency, signalData = wavfile.read(os.path.join(getcwd(), 'datasets/train/5/') + 'mix_24.wav')
    # Plot the signal read from wav file
    plot.figure(figsize=(10, 4))
    plot.subplot()
    plot.specgram(signalData, Fs=samplingFrequency)
    plot.xlabel('Time (s)')
    plot.ylabel('Frequency (Hz)')
    plot.colorbar(format='%+2.0f dB')
    plot.title('Mel spectrogram')
    plot.tight_layout()
    plot.show()
    '''

    _main()
