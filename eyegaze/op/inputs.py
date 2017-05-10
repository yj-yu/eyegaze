from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np


def load_video_feature_from(hdf_f, vid_key):
    def read_h5_field(vid_key):
        video_feature = np.array(hdf_f[vid_key])
        video_feature = np.transpose(video_feature, [0, 2, 3, 1])
        assert list(video_feature.shape[1:]) == [7, 7, 2048]
        return video_feature

    video_op = tf.py_func(read_h5_field, [vid_key], tf.float32)
    video_op.set_shape([None, 7, 7, 2048])

    return video_op

'''
def load_video_from(vid_path):
    video_op = tf.py_func(read_h5_field, [vid_key], tf.float32)
    video_op.set_shape([None, 7, 7, 2048])

    return video_op
'''


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shuffle=True,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
    """
    """
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        tf.logging.fatal("Found no input files matching %s", file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), file_pattern)

    if is_training:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=shuffle, capacity=16, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        if shuffle:
            values_queue = tf.RandomShuffleQueue(
                capacity=capacity,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string],
                name="random_" + value_queue_name)
        else:
            values_queue = tf.FIFOQueue(
                capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

    else:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=False, capacity=1, name=shard_queue_name)
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(
            capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
        values_queue, enqueue_ops))
    tf.summary.scalar(
        "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
        tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

    return values_queue


