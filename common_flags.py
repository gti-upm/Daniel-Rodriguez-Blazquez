import gflags

FLAGS = gflags.FLAGS

# Random seed
gflags.DEFINE_bool('random_seed', True, 'Random seed')

# Input
gflags.DEFINE_integer('target_level', -14, 'Target Level dB')
gflags.DEFINE_float('target_nu', 0.1995262315, 'Target Level nu')
gflags.DEFINE_integer('sr', 44100, 'Sample Rate')
gflags.DEFINE_integer('img_width', 862, 'Target Image Width')
gflags.DEFINE_integer('img_height', 96, 'Target Image Height')
gflags.DEFINE_string('img_mode', "rgb", 'Load mode for images, either rgb or grayscale')
gflags.DEFINE_string('h_grid', 50, 'Horizontal size of the grid classifiers')
gflags.DEFINE_string('v_grid', 25, 'Vertical size of the grid classifiers')

# Training parameters
gflags.DEFINE_integer('batch_size', 128, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 100, 'Number of epochs for training')
gflags.DEFINE_integer('verbose', 1, 'Type of verbose for training')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')
gflags.DEFINE_float('initial_lr', 1e-3, 'Initial learning rate for adam')
gflags.DEFINE_string('f_output', 'softmax', 'Output function')

# Testing parameters
gflags.DEFINE_float('IOU', 0.5, 'Threshold for the IoU')
gflags.DEFINE_float('NMS', 0.3, 'Threshold for the NMS')
gflags.DEFINE_float('marker_threshold', 0.4, 'Probabilities higher than this threshold are marked as vehicles')
gflags.DEFINE_float('area_threshold', 19000/(1.875**2), 'Max size of the detections')

# Files
# ---------- OJO -------------
gflags.DEFINE_string('audio_rootdir', "./datasets/audio/", 'Folder containing audio folders')
gflags.DEFINE_string('speech_rootdir', "./datasets/speech/", 'Folder containing speech folders')
gflags.DEFINE_string('experiment_rootdir', "./experiment_6/", 'Folder containing all the logs, model weights and results')
gflags.DEFINE_string('train_dir', "datasets/train_spec/", 'Folder containing training experiments')
gflags.DEFINE_string('val_dir', "datasets/val_spec/", 'Folder containing validation experiments')
gflags.DEFINE_string('test_dir', "datasets/test_spec/", 'Folder containing testing experiments')
gflags.DEFINE_string('exp_name', "exp_1", 'Name of the experiment to be processed')
gflags.DEFINE_string('data_path', "./datasets/train_spec", 'Folder containing the whole dataset')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained model for training')
gflags.DEFINE_string('weights_fname', "weights_039.h5", '(Relative) filename of model weights')
gflags.DEFINE_string('initial_weights', './experiment_6/weights_039.h5', '(Relative) filename of model initial training weights')
gflags.DEFINE_string('json_model_fname', "model_struct.json", 'Model struct json serialization, filename')
gflags.DEFINE_integer('num_classes', 10, 'Number of classes')
