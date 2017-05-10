from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


__path__ = os.path.abspath(os.path.dirname(__file__))
VRSUMM_DATA_PATH = os.path.normpath(os.path.join(__path__, "/data1/common_datasets/vrsumm"))


class ModelConfig(object):

    def __init__(self):
        self.model_name = "inception"
        self.experiment_name = "gazemodel"
        self.experiment_checkpoint = None
        self.from_other_checkpoint = None
        self.dataset_name = "VAS"
        if self.dataset_name == "VAS":
            self.input_file_pattern = os.path.join(VRSUMM_DATA_PATH, "TFRecord/VAS/train-????-of-0100")
        else:
            self.input_file_pattern = ""
        self.image_format = "jpeg"
        self.values_per_input_shard = 2000
        self.input_queue_capacity_factor = 2
        self.num_input_reader_threads = 2
        self.num_preprocess_threads = 4

        self.batch_size = 12

        self.video_height = 299
        self.video_width = 299
        self.video_steps = 16

        self.image_feature_name = "image/data"
        self.inception_checkpoint_file = "/data1/common_datasets/vrsumm/PretrainedModules/inception_v3.ckpt"
        self.c3d_checkpoint_file = "/data1/common_datasets/vrsumm/PretrainedModules/sports1m_finetuning_ucf101.model"
        self.c3d_mean_file = "/data1/common_datasets/vrsumm/PretrainedModules/c3d_mean.npy"

        self.initializer_scale = 0.08
        self.dropout_keep_prob = 0.7
        self.lstm_dropout_keep_prob = 0.5
        self.final_activation = 'sigmoid'
        self.train_CNN = False
class TrainingConfig(object):
    """ Wrapper class for training hyperparameters"""

    def __init__(self):
        """Sets the default training hyperparameters"""
        self.num_examples_per_epoch = 99999
        self.optimizer = "Adam"
        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0

        # Learning rate when fine tuning the Inception v3 parameters.
        self.train_inception_learning_rate = 0.0001

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5000
