from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import tfplot
import tfplot.summary
from eyegaze.op import image_embedding
from eyegaze.op import inputs as input_ops
from eyegaze.op import video_processing
slim = tf.contrib.slim

from tensorflow.contrib.layers.python.layers import initializers
from eyegaze.op import loss
import h5py
import pudb

class GazeModel(object):
    """
    """
    def __init__(self, config, mode, train_inception=False):
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        self.reader = tf.TFRecordReader()
        #self.video_hdf5 = h5py.File(self.config.video_hdf_path, 'r')
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        # A float32 Tensor with shape [batch_size, video_steps, height, width, channels].
        self.videos = None
        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.image_embeddings = None
        self.pos_score = None

        self.total_loss = None
        self.target_maxmargin_losses = None

        # Functions for inception/c3d submodel from checkpoint
        self.inception_variables = []
        self.c3d_variables = []
        self.init_fn = None

        self.global_step = None


    def is_training(self):
        return self.mode == "train"

    def process_video(self, frm_paths, thread_id=0):
        return video_processing.process_video(frm_paths,
                                              is_training=self.is_training(),
                                              step_length=16,
                                              height=self.config.video_height,
                                              width=self.config.video_height,
                                              resize_height=self.config.video_height,
                                              resize_width=self.config.video_width,
                                              thread_id=thread_id,
                                              image_format=self.config.image_format)
    def process_image(self, frm_path, thread_id=0):
        return video_processing.process_image(frm_path,
                                              is_training=self.is_training(),
                                              height=280,
                                              width=280,
                                              resize_height=280,
                                              resize_width=280,
                                              thread_id=thread_id,
                                              image_format=self.config.image_format)

    def parse_subshot_example(self, serialized_example):
        """
        """
        # TODO feature_name to be written in config.
        dataset_prefix = self.config.dataset_name
        context, sequence = tf.parse_single_sequence_example(
            serialized_example,
            context_features={
                dataset_prefix+"/video_id": tf.FixedLenFeature([], tf.string),
                dataset_prefix+"/video_path": tf.FixedLenFeature([], tf.string),
                dataset_prefix+"/key": tf.FixedLenFeature([], tf.string),
            },
            sequence_features={
                dataset_prefix+"/pupil_size": tf.FixedLenSequenceFeature([], dtype=tf.float32)
                dataset_prefix+"/frm_paths": tf.FixedLenSequenceFeature([], dtype=tf.string)
            })

        result = {
            "key": context[dataset_prefix+"/key"],
            "vid_key": context[dataset_prefix+"/video_id"],
            "pupil_size": sequence[dataset_prefix+"/pupil_size"],
            "frm_paths": sequence[dataset_prefix+"/frm_paths"]
        }
        return result

    def build_inputs(self):
        """Input prefetching, preprocessing, and batching."""
        if self.mode == "inference" or self.mode == "eval":
            # 360 degree video or highlight segment in long video... TODO
            pass
            self.image_feed = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size,280,280,3], name="image_feed")
            #input_feed = tf.placeholder(dtype=tf.int64,
            #                            shape=[None],
            #                            name="input_feed")
            #images = tf.expand_dims(self.process_image(image_feed),0)
            images = self.image
            #input_seqs = tf.expand_dims(input_feed,1)

        else:
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                is_training=self.is_training(),
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                shuffle=True,
                num_reader_threads=self.config.num_input_reader_threads)
            assert self.config.batch_size % 2 == 0
            assert self.config.num_preprocess_threads % 2 == 0
            # Add negative Prefetch serialized SequenceExample protos.
            enqueue_list = []
            for thread_id in range(self.config.num_preprocess_threads):
                #video = tf.stack([video_pos,video_neg],axis=1)
                serialized_sequence_example = input_queue.dequeue()
                feature_pos = self.parse_subshot_example(serialized_sequence_example)
                video = self.process_video_pupil(feature_pos["frm_paths"], feature_pos["pupil_size"])
                enqueue_list.append([video, pupil_size])

            # Batch inputs.
            queue_capacity = (2 * self.config.num_preprocess_threads *
                              self.config.batch_size)

            videos, pupil_sizes = tf.train.batch_join(
                enqueue_list,
                batch_size=self.config.batch_size,
                capacity=queue_capacity,
                dynamic_pad=True,
                name="batch_and_pad")

        #self.videos_pos = videos_pos
        #self.videos_neg = videos_neg
        self.videos = videos
        self.pupil_sizes = pupil_sizes
    def build_image_embeddings(self):
        """Builds the image model subraph and generate image embeddings.
        Inputs:
            self.images
        Outputs:
            self.image_embeddings
        """
        #video_batch = tf.concat([self.videos_pos, self.videos_neg],axis=0)
        video_batch = tf.reshape(self.videos, [-1,299,299,3])
        inception_output = image_embedding.inception_v3(
            video_batch,
            trainable=self.train_inception,
            is_training=self.is_training())
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
        '''
        c3d_output, self.c3d_weights, self.c3d_biases = image_embedding.c3d(
            video_batch,
            trainable=self.train_inception,
            is_training=self.is_training())

        c3d_variables = self.c3d_weights.values() + self.c3d_biases.values()
        self.c3d_variables = {'var_name/'+var.name[:-2]:var for var in c3d_variables}
        #inception_s5 = inception_output
        c3d_conv5 = c3d_output['conv5']
        c3d_pool5 = tf.squeeze(tf.nn.max_pool3d(c3d_conv5, ksize=[1,2,1,1,1], strides=[1,2,1,1,1], padding='SAME', name="temp_pool"))
        '''
        with tf.variable_scope("image_imbedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=inception_output,
                num_outputs=self.config.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)
        self.image_embeddings = image_embeddings
        #self.logits = score
    def build_model(self):

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.config.num_lstm_units, state_is_tuple=True)
        if self.mode == "train":
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                input_keep_prob=self.config.lstm_dropout_keep_prob,
                output_keep_prob=self.config.lstm_dropout_keep_prob)
        zero_state = lstm_cell.zero_state(
            batch_size=self.config.batch_size, dtype=tf.float32)
        video_batch = tf.reshape(self.image_embeddings, [self.config.batch_size, -1, self.config.embedding_size])
        output_states, final_state = lstm_cell(video_batch, zero_state)
        pudb.set_trace()
        output_state_pack = tf.transpose(tf.stack(output_states), [1,0,2])
        with tf.variable_scope("pupil_score") as scope:
             pupil_score = tf.contrib.layers.fully_connected(
                inputs=output_states,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)

        l2_loss = (pupil_score - self.pupil_sizes) ** 2
        batch_loss = tf.reduce_sum(l2_loss)/self.config.batch_size
        tf.losses.add_loss(batch_loss)
        total_loss = batch_loss # tf.losses.get_total_loss()

        # Add loss summaries
        tf.summary.scalar("losses/batch_loss", batch_loss)
        tf.summary.scalar("losses/total_loss", total_loss)
        for var in tf.trainable_variables():
            tf.summary.histogram("parameters/" + var.op.name, var)

        self.total_loss = total_loss
    def build_eval_model(self):
        # Define eval scores
        self.logits = self.logits

    def setup_submodule_initializer(self):
        if self.mode != "inference":
            if self.config.from_other_checkpoint is not None:
                saver = tf.train.Saver()
                def restore_check(sess):
                    tf.logging.info("Restoring from other checkpoint repo %s",
                                    self.config.from_other_checkpoint)
                    saver.restore(sess, self.config.from_other_checkpoint)
                self.init_fn = restore_check
                return
            # Restore CNN variables only
            saver = tf.train.Saver(self.inception_variables)
            #saver_c3d = tf.train.Saver(self.c3d_variables)
            def restore_fn(sess):
                tf.logging.info("Restoring variables from inception checkpoint file %s",
                                self.config.inception_checkpoint_file)
                saver.restore(sess, self.config.inception_checkpoint_file)
                #tf.logging.info("Restoring variables from c3d checkpoint file %s",
                #                self.config.c3d_checkpoint_file)
                #saver_c3d.restore(sess, self.config.c3d_checkpoint_file)
            self.init_fn = restore_fn

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step


    def build(self):
        self.build_inputs()
        self.build_image_embeddings()
        if self.mode == 'eval':
            self.build_eval_model()
        else:
            self.build_model()
        self.setup_submodule_initializer()
        self.setup_global_step()
