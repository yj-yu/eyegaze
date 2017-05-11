import os
import sys
import random
import threading

from collections import Counter
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
tf.flags.DEFINE_string("video_dir", "../../dataset/vid_frm",
                       "Video frame dirctory.")

tf.flags.DEFINE_string("train_positive_file", "../../dataset/VAS/DataFrame/pupil_Train.csv",
                       "Training dataframe positive csv file.")
tf.flags.DEFINE_string("val_dataframe_file", "../../dataset/VAS/DataFrame/pupil_val.csv",
                       "Validation dataframe csv file.")
tf.flags.DEFINE_string("test_dataframe_file", "../../dataset/VAS/DataFrame/pupil_test.csv",
                       "Test dataframe csv file.")

tf.flags.DEFINE_string("output_dir", "../../dataset/VAS/TFRecord/",
                       "Output TFRecord data directory.")

tf.flags.DEFINE_integer("train_shards", 100,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 10,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 10,
                        "Number of shards in testing TFRecord files.")

# WTL : Where to Look, HIGH : Highlight detection, GIF : GIF Highlight
tf.flags.DEFINE_string("type", "VAS",
                       "Summary work type: VAS")

tf.flags.DEFINE_integer("min_subshot_frm", 4,
                        "The minimum number of frames in subshot "
                        "The subshot has less than this number will be discarded.")


tf.flags.DEFINE_integer("num_threads", 5,
                        "Number of threads to preprocess the videos.")

tf.flags.DEFINE_integer("video_steps", 40,
                        "Number of video steps")

FLAGS = tf.flags.FLAGS

# Video means highlight or non-highlight subshot of 16 frames.
VideoMetadata = namedtuple("VideoMetadata",
                           ["video_id", "video_path", "information"])

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a byte Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _float_feature_list(values):
    """Wrapper for inserting an float32 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting an byte FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(video):
    """Builds a SequenceExample proto for an video-caption pair etc..

    Args:
        video: An VideoMetadata object.

    Returns:
        A SequenceExample proto.
    """

    if FLAGS.type in ["VAS"]:
        tp = FLAGS.type
        context = tf.train.Features(feature={
            tp+"/video_id": _bytes_feature(video.video_id),
            tp+"/video_path": _bytes_feature(video.video_path),
            tp+"/key": _bytes_feature(video.information["key"]),
        })
        #pudb.set_trace()
        frms = os.listdir(video.video_path)
        frms = filter(lambda x: (x[-3:] == "jpg" or x[-4:] == "jpeg") , frms)
        frms = sorted(frms)
        # Make frms has 16 frms
        chunk_len = 360
        frm_len = len(frms)
        if frm_len > chunk_len:
            # Randomly select or..
            frms = frms[:chunk_len]
        else:
            temp_len = len(frms)
            if len(frms) == 0:
                return None
            temp = [frms[i%temp_len] for i in range(chunk_len)]
            frms = temp

        frm_paths = [os.path.join(video.video_path, frm) for frm in frms]
        mat_dat = h5py.File(video.information['rawdata']).get(video.information["key"])
        pupil_data = []
        for sub_name, subject in mat_dat.iteritems():
            try:
                pupil_data.append(subject['pupilsize'].value)
            except:
                continue
        pupil_size = np.mean(pupil_data,axis=0).tolist()[0]
        # 1 is high, 0 is non-high
        feature_lists = tf.train.FeatureLists(feature_list={
            tp+"/frm_paths": _bytes_feature_list(frm_paths),
            tp+"/pupil_size": _float_feature_list(pupil_size),
        })
        sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)

    return sequence_example


def _process_videos(thread_index, ranges, name, videos, num_shards):
    """Processes and saves a subset of video metadata as TFRecord files in one thread.

    Each thread produces N shards where N = num_shards / num_threads.
    For instance, if num_shards = 128, and num_threads = 2, then the first
    thread would produce shards [0, 64).

    Args:
        thread_index: Integer thread identifier within [0, len(ranges)].
        ranges: A list of pairs of integers specifying the ranges of the datset to
            process in parallel.
        name: Unique identifier specifying the dataset.
        videos: List of VideoMetadata.
        num_shards: Integer number of shards for the output files.
    """
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_videos_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-0002-of-0010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.4d-of-%.4d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        videos_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for v in videos_in_shard:
            video = videos[v]

            sequence_example = _to_sequence_example(video)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 100:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_videos_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d %s working data to %s." %
              (datetime.now(), thread_index, shard_counter, FLAGS.type, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d %s working data to %d shards." %
          (datetime.now(), thread_index, counter, FLAGS.type, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, videos, num_shards):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
        name: Unique identifier specifying the dataset.
        videos: List of VideoMetadata.
        num_shards: Integer number of shards for the output files.
    """
    if FLAGS.type in ["VAS"]:
        random.seed(12345)
        random.shuffle(videos)

        num_threads = min(num_shards, FLAGS.num_threads)
        spacing = np.linspace(0, len(videos), num_threads + 1).astype(np.int)
        ranges = []
        threads = []
        for i in xrange(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])

        coord = tf.train.Coordinator()

        print("Launching %d threads for spacings: %s" % (num_threads, ranges))
        for thread_index in xrange(len(ranges)):
            args = (thread_index, ranges, name, videos, num_shards)
            t = threading.Thread(target=_process_videos, args=args)
            t.start()
            threads.append(t)

        coord.join(threads)
        print("%s: Finished processing all %d video metadata in the data set '%s'." %
              (datetime.now(), len(videos), name))


def _load_and_process_metadata(df_file):
    """Loads video metadata from a dataframe file and processes the meta informations.

    Args:
        df_file: csv file containing meta informations.

    Returns:
        A list of VideoMetadata.
    """
    if FLAGS.type in ["VAS"]:
        df = pd.read_csv(df_file, sep='\t')
        df = df.set_index('key')

        id_to_video = []
        id_to_label = {}
        for key in df.index:
            id_to_video.append((key, df.loc[key, 'path']))
            id_to_label.update({key: df.loc[key, 'rawdata']})

        assert len(id_to_video) == len(id_to_label)

        video_metadata = []
        for key, video_path in id_to_video:
            #video_path = os.path.join(FLAGS.video_dir, video_name)
            #video_path = video_path
            information = {"rawdata": id_to_label[key], "key": key}
            video_metadata.append(VideoMetadata(key, video_path, information))

        print("Finished processing video metadata")

    return video_metadata


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards)
    assert _is_valid_num_shards(FLAGS.val_shards)
    assert _is_valid_num_shards(FLAGS.test_shards)

    FLAGS.output_dir = os.path.join(FLAGS.output_dir, FLAGS.type)

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    train_pos = _load_and_process_metadata(FLAGS.train_positive_file)
    val_dataset = _load_and_process_metadata(FLAGS.val_dataframe_file)
    #test_dataset = _load_and_process_metadata(FLAGS.test_dataframe_file)

    #train_captions = [video.information["caption"] for video in train_dataset]

    _process_dataset("train", train_pos, FLAGS.train_shards)

    _process_dataset("val", val_dataset, FLAGS.val_shards)
    #_process_dataset("test", test_dataset, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()
