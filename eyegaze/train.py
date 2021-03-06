"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from eyegaze import configuration
from eyegaze.model import GazeModel
FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "/data1/common_datasets/VAS/PretrainedModules/inceptionv3.model",
                       "Path to a pretrained CVS model.")
tf.flags.DEFINE_string("train_dir", "/data2/yj/experiment/",
                       "Directory for saving and loading model checkpoints.")
#tf.flags.DEFINE_boolean("train_CNN", False,
#                        "Whether to train CVS submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 10000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 10,
                        "Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    assert FLAGS.train_dir, "--train_dir is required"
    model_config = configuration.ModelConfig()
    #model_config.input_file_pattern = FLAGS.input_file_pattern
    #model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
    training_config = configuration.TrainingConfig()

    # Create training directory.
    train_dir = FLAGS.train_dir + model_config.experiment_name
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)
    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        if model_config.model_name == 'inception':
            model = GazeModel.GazeModel(
                model_config, mode="train", train_inception=model_config.train_CNN)
        elif model_config.model_name == 'c3d':
            model = GazeModel.GazeModel(
                model_config, mode="train", train_inception=model_config.train_CNN)
        else:
            tf.logging.info("Not Implemented %s Yet", model_config.model_name)
        model.build()

        # Set up the learning rate.
        learning_rate_decay_fn = None
        if model_config.train_CNN:
            learning_rate = tf.constant(training_config.train_inception_learning_rate)
        else:
            learning_rate = tf.constant(training_config.initial_learning_rate)
        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                    model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                            training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=training_config.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model.total_loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

    sess_configure = tf.ConfigProto()
    sess_configure.gpu_options.allow_growth = True
    # Run training.
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=100,
        init_fn=model.init_fn,
        saver=saver,
        session_config=sess_configure)


if __name__ == "__main__":
    tf.app.run()
