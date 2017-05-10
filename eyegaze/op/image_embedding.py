# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image embedding ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
import pudb
slim = tf.contrib.slim


def inception_v3(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 f_endpoint='Mixed_7c',
                 pooled=False,
                 scope="InceptionV3"):
  """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
      batch_norm_params = {
          "is_training": is_inception_model_training,
          "trainable": trainable,
          # Decay for the moving averages.
          "decay": 0.9997,
          # Epsilon to prevent 0s in variance.
          "epsilon": 0.001,
          # Collection containing the moving mean and moving variance.
          "variables_collections": {
              "beta": None,
              "gamma": None,
              "moving_mean": ["moving_vars"],
              "moving_variance": ["moving_vars"],
          }
      }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v3_base(images,final_endpoint=f_endpoint, scope=scope)

        with tf.variable_scope("logits"):
          shape = net.get_shape()
          if pooled:
              net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
          net = slim.dropout(
              net,
              keep_prob=dropout_keep_prob,
              is_training=is_inception_model_training,
              scope="dropout")
          #net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net

def c3d(images,
        trainable=True,
        is_training=True,
        weight_decay=0.00004,
        stddev=0.1,
        dropout_keep_prob=0.8,
        use_batch_norm=True,
        batch_norm_params=None,
        add_summaries=True,
        scope="c3d"):
    def conv3d(name, l_input, w, b, trainable=True):
        return tf.nn.bias_add(tf.nn.conv3d(l_input,w,strides=[1,1,1,1,1], padding='SAME'),
                              b)
    def conv2d(name, l_input, w, b, trainable=True):
        return tf.nn.bias_add(tf.nn.conv2d(l_input,w,strides=[1,1,1,1], padding='SAME'),
                              b)
    def max_pool(name, l_input, k):
        return tf.nn.max_pool3d(l_input, ksize=[1,k,2,2,1], strides=[1,k,2,2,1], padding='SAME')
    """Builds an C3D for image embeddings.
    Args:
    images: A float32 Tensor of shape [batch, 16, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
        tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.
    Returns:
    end_points: A dictionary of activations from c3d layers.v
    """
    # Only consider the inception model to be in training mode if it's trainable.
    is_model_training = trainable and is_training
    # Batch norm not implemented yet : yj
    if use_batch_norm:
    # Default parameters for batch normalization.
        if not batch_norm_params:
            batch_norm_params = {
                "is_training": is_model_training,
                "trainable": trainable,
                # Decay for the moving averages.
                "decay": 0.9997,
                # Epsilon to prevent 0s in variance.
                "epsilon": 0.001,
                # Collection containing the moving mean and moving variance.
                "variables_collections": {
                    "beta": None,
                    "gamma": None,
                    "moving_mean": ["moving_vars"],
                    "moving_variance": ["moving_vars"],
                }
            }
        else:
            batch_norm_params = None

    if trainable:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        wd_train = 0.0005
    else:
        weights_regularizer = None
        wd_train = 0.0000
    def _variable_with_weight_decay(name, shape, wd, trainable=True):
        var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer(), trainable=trainable)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _variable_on_cpu(name, shape, initializer, trainable=True):
        with tf.device('/cpu:0'): #gpu?
            var = tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
        return var

    weights = {
        'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], wd_train, trainable=trainable),
        'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], wd_train, trainable=trainable),
        'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], wd_train, trainable=trainable),
        'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], wd_train, trainable=trainable),
        'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], wd_train, trainable=trainable),
        'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], wd_train, trainable=trainable),
        'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], wd_train, trainable=trainable),
        'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], wd_train, trainable=trainable),
        'wd1': _variable_with_weight_decay('wd1', [8192, 4096], wd_train, trainable=trainable),
        'wd2': _variable_with_weight_decay('wd2', [4096, 4096], wd_train, trainable=trainable),
#        'out': _variable_with_weight_decay('wout', [4096,c3d_model.NUM_CLASSES], wd_train, trainable=trainable),
#        'wcregion': _variable_with_weight_decay('wcregion', [3, 3, 1024, 1], 0.0005)
    }
    biases = {
        'bc1': _variable_with_weight_decay('bc1', [64], 0.000, trainable=trainable),
        'bc2': _variable_with_weight_decay('bc2', [128], 0.000, trainable=trainable),
        'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000, trainable=trainable),
        'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000, trainable=trainable),
        'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000, trainable=trainable),
        'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000, trainable=trainable),
        'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000, trainable=trainable),
        'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000, trainable=trainable),
        'bd1': _variable_with_weight_decay('bd1', [4096], 0.000, trainable=trainable),
        'bd2': _variable_with_weight_decay('bd2', [4096], 0.000, trainable=trainable),
#        'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000, trainable=trainable),
#        'bcregion': _variable_with_weight_decay('bcregion', [1], 0.000)
    }

    net = {}

    net['conv1'] = conv3d('conv1', images, weights['wc1'], biases['bc1'], trainable=trainable )
    net['conv1'] = tf.nn.relu(net['conv1'], 'relu1')
    net['pool1'] = max_pool('pool1', net['conv1'], k=1)

    # Convolution Layer
    net['conv2'] = conv3d('conv2', net['pool1'], weights['wc2'], biases['bc2'], trainable=trainable)
    net['conv2'] = tf.nn.relu(net['conv2'], 'relu2')
    net['pool2'] = max_pool('pool2', net['conv2'], k=2)

    # Convolution Layer
    net['conv3'] = conv3d('conv3a', net['pool2'], weights['wc3a'], biases['bc3a'], trainable=trainable)
    net['conv3'] = tf.nn.relu(net['conv3'], 'relu3a')
    net['conv3'] = conv3d('conv3b', net['conv3'], weights['wc3b'], biases['bc3b'], trainable=trainable)
    net['conv3'] = tf.nn.relu(net['conv3'], 'relu3b')
    net['pool3'] = max_pool('pool3', net['conv3'], k=2)

    # Convolution Layer
    net['conv4'] = conv3d('conv4a', net['pool3'], weights['wc4a'], biases['bc4a'], trainable=trainable)
    net['conv4'] = tf.nn.relu(net['conv4'], 'relu4a')
    net['conv4'] = conv3d('conv4b', net['conv4'], weights['wc4b'], biases['bc4b'], trainable=trainable)
    net['conv4'] = tf.nn.relu(net['conv4'], 'relu4b')
    net['pool4'] = max_pool('pool4', net['conv4'], k=2)

    # Convolution Layer
    net['conv5'] = conv3d('conv5a', net['pool4'], weights['wc5a'], biases['bc5a'], trainable=trainable)
    net['conv5'] = tf.nn.relu(net['conv5'], 'relu5a')
    net['conv5'] = conv3d('conv5b', net['conv5'], weights['wc5b'], biases['bc5b'], trainable=trainable)
    net['conv5'] = tf.nn.relu(net['conv5'], 'relu5b')
    net['pool5'] = max_pool('pool5', net['conv5'], k=2)
    net['pool5'] = max_pool('pool5', net['pool5'], k=2)


    # Add summaries.
    #if add_summaries:
    #for v in end_points.values():
    #    tf.contrib.layers.summaries.summarize_activation(v)

    return net, weights, biases


def c3d_full(images,
        trainable=True,
        is_training=True,
        weight_decay=0.00004,
        stddev=0.1,
        dropout_keep_prob=0.8,
        use_batch_norm=True,
        batch_norm_params=None,
        add_summaries=True,
        scope="c3d"):
    def conv3d(name, l_input, w, b):
        return tf.nn.bias_add(tf.nn.conv3d(l_input,w,strides=[1,1,1,1,1], padding='SAME'),
                              b)
    def conv2d(name, l_input, w, b):
        return tf.nn.bias_add(tf.nn.conv2d(l_input,w,strides=[1,1,1,1], padding='SAME'),
                              b)
    def max_pool(name, l_input, k):
        return tf.nn.max_pool3d(l_input, ksize=[1,k,2,2,1], strides=[1,k,2,2,1], padding='SAME')
    """Builds an C3D for image embeddings.
    Args:
    images: A float32 Tensor of shape [batch, 16, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
        tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.
    Returns:
    end_points: A dictionary of activations from c3d layers.v
    """
    # Only consider the inception model to be in training mode if it's trainable.
    is_model_training = trainable and is_training
    # Batch norm not implemented yet : yj
    if use_batch_norm:
    # Default parameters for batch normalization.
        if not batch_norm_params:
            batch_norm_params = {
                "is_training": is_model_training,
                "trainable": trainable,
                # Decay for the moving averages.
                "decay": 0.9997,
                # Epsilon to prevent 0s in variance.
                "epsilon": 0.001,
                # Collection containing the moving mean and moving variance.
                "variables_collections": {
                    "beta": None,
                    "gamma": None,
                    "moving_mean": ["moving_vars"],
                    "moving_variance": ["moving_vars"],
                }
            }
        else:
            batch_norm_params = None

    if trainable:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        weights_regularizer = None

    def _variable_with_weight_decay(name, shape, wd):
        var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'): #gpu?
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    weights = {
        'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
        'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
        'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
        'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
        'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
        'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
        'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
        'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
        'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
#        'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
#        'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005),
#        'wcregion': _variable_with_weight_decay('wcregion', [3, 3, 1024, 1], 0.0005)
    }
    biases = {
        'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
        'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
        'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
        'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
        'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
        'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
        'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
        'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
        'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
#        'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
#        'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
#        'bcregion': _variable_with_weight_decay('bcregion', [1], 0.000)
    }

    net = {}

    net['conv1'] = conv3d('conv1', images, weights['wc1'], biases['bc1'])
    net['conv1'] = tf.nn.relu(net['conv1'], 'relu1')
    net['pool1'] = max_pool('pool1', net['conv1'], k=1)

    # Convolution Layer
    net['conv2'] = conv3d('conv2', net['pool1'], weights['wc2'], biases['bc2'])
    net['conv2'] = tf.nn.relu(net['conv2'], 'relu2')
    net['pool2'] = max_pool('pool2', net['conv2'], k=2)

    # Convolution Layer
    net['conv3'] = conv3d('conv3a', net['pool2'], weights['wc3a'], biases['bc3a'])
    net['conv3'] = tf.nn.relu(net['conv3'], 'relu3a')
    net['conv3'] = conv3d('conv3b', net['conv3'], weights['wc3b'], biases['bc3b'])
    net['conv3'] = tf.nn.relu(net['conv3'], 'relu3b')
    net['pool3'] = max_pool('pool3', net['conv3'], k=2)

    # Convolution Layer
    net['conv4'] = conv3d('conv4a', net['pool3'], weights['wc4a'], biases['bc4a'])
    net['conv4'] = tf.nn.relu(net['conv4'], 'relu4a')
    net['conv4'] = conv3d('conv4b', net['conv4'], weights['wc4b'], biases['bc4b'])
    net['conv4'] = tf.nn.relu(net['conv4'], 'relu4b')
    net['pool4'] = max_pool('pool4', net['conv4'], k=2)

    # Convolution Layer
    net['conv5'] = conv3d('conv5a', net['pool4'], weights['wc5a'], biases['bc5a'])
    net['conv5'] = tf.nn.relu(net['conv5'], 'relu5a')
    net['conv5'] = conv3d('conv5b', net['conv5'], weights['wc5b'], biases['bc5b'])
    net['conv5'] = tf.nn.relu(net['conv5'], 'relu5b')
    net['pool5'] = max_pool('pool5', net['conv5'], k=2)
    net['pool5'] = tf.transpose(net['pool5'], perm=[0,1,4,2,3])

    net['dense1'] = tf.reshape(net['pool5'], [-1, weights['wd1'].get_shape().as_list()[0]])
    net['dense1'] = tf.nn.relu( tf.matmul(net['dense1'], weights['wd1']) + biases['bd1'] )

    # Add summaries.
    #if add_summaries:
    #for v in end_points.values():
    #    tf.contrib.layers.summaries.summarize_activation(v)

    return net, weights, biases


