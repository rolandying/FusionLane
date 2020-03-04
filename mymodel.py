"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers

import preprocessing
import xception

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4
_LR_ = 0.0001


def atrous_spatial_pyramid_pooling(inputs, batch_norm_decay, is_training, depth=256):
  """Atrous Spatial Pyramid Pooling.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.

  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("aspp"):

    atrous_rates = [6, 12, 18]


    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
          image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net


def model_generator(num_classes,
                    pre_trained_model,
                    batch_norm_decay,
                    data_format='channels_last'):
  """Generator for the models.

  Args:
    num_classes: The number of possible classes for image classification.
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    base_architecture: The architecture of base Resnet building block.
    pre_trained_model: The path to the directory that contains pre-trained models.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
      Only 'channels_last' is supported currently.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the DeepLab v3 model.
  """
  if data_format is None:
    # data_format = (
    #     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    pass

  if batch_norm_decay is None:
    batch_norm_decay = _BATCH_NORM_DECAY

  def model(inputs,region, is_training):
    """Constructs the ResNet model given the inputs."""
    if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
      region = tf.transpose(region, [0, 3, 1, 2])

    # tf.logging.info('net shape: {}'.format(inputs.shape))
    # encoder
    with tf.contrib.slim.arg_scope(xception.xception_arg_scope(batch_norm_decay=batch_norm_decay)):
      logits, end_points,low_level_features = xception.xception(inputs,
                                             region,
                                             num_classes=None,
                                             is_training=is_training)

    if is_training and pre_trained_model != None :
      #exclude = ['xception/logits', 'global_step']
      variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=None)
      tf.train.init_from_checkpoint(pre_trained_model,
                                    {v.name.split(':')[0]: v for v in variables_to_restore})

    inputs_size = tf.shape(inputs)[1:3]
    net = end_points['Logits']
    encoder_output = atrous_spatial_pyramid_pooling(net, batch_norm_decay, is_training)

    with tf.variable_scope("lstm"):
      with tf.contrib.slim.arg_scope(xception.xception_arg_scope(batch_norm_decay=batch_norm_decay)):
        with arg_scope([layers.batch_norm], is_training=is_training):
          k_size = encoder_output.get_shape().as_list()[2]
          #k_size = 21
          net = layers_lib.conv2d(encoder_output, 64, [1, 1], stride=1, scope="conv1_1x1")          
          #net = layers_lib.conv2d(net, 4, [1, 1], stride=1, scope="conv2_1x1")
          rnn_input = tf.reshape(net,[-1,4,k_size,k_size,64])
          cell_1 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[k_size, k_size, 64], output_channels=64, kernel_shape=[3, 3])
          #cell_2 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[k_size, k_size, 256], output_channels=256, kernel_shape=[3, 3])
          #cell_3 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[k_size, k_size, 256], output_channels=256, kernel_shape=[3, 3])
          multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell_1])
          dropout_rnn = tf.contrib.rnn.DropoutWrapper(multi_rnn_cell, input_keep_prob=0.4)
          #init_state = multi_rnn_cell.zero_state(12,dtype=tf.float32)
          rnn_outputs, state = tf.nn.dynamic_rnn(cell = dropout_rnn,
                                                 inputs=rnn_input,
                                                 time_major=False,
                                                 dtype=tf.float32)

    with tf.variable_scope("decoder"):
      with tf.contrib.slim.arg_scope(xception.xception_arg_scope(batch_norm_decay=batch_norm_decay)):
        with arg_scope([layers.batch_norm], is_training=is_training):
          with tf.variable_scope("low_level_features"):
            low_level_features = layers_lib.conv2d(low_level_features, 24,
                                                   [1, 1], stride=1, scope='conv_1x1')
            low_level_features_size = tf.shape(low_level_features)[1:3]

          with tf.variable_scope("upsampling_logits"):
            encoder_output_re = tf.reshape(rnn_outputs,[-1,k_size,k_size,64])
            net = layers_lib.conv2d(encoder_output_re, 64, [3, 3], stride=1, scope='conv_3x3_1')
            net = layers_lib.conv2d(net, 64, [3, 3], stride=1, scope='conv_3x3_2')
            net = tf.image.resize_bilinear(net, low_level_features_size, name='upsample_1')
            net = tf.concat([net, low_level_features], axis=3, name='concat')
            net = layers_lib.conv2d(net, 44, [3, 3], stride=1, scope='conv_3x3_3')
            net = layers_lib.conv2d(net, 44, [3, 3], stride=1, scope='conv_3x3_4')
            net = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')
            #print(inputs.get_shape().as_list())
            low_level_features_two = layers_lib.conv2d(inputs, 1, [1, 1], stride=1, scope='low_level_feature_conv_1x1')
            net = tf.concat([net,low_level_features_two], axis=3, name='concat_2')
            logits = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='outputs')

    return logits

  return model


def network_model_fn(features,labels, mode, params):
  
  if isinstance(features, dict):
    features = features['feature']

  region =  features[:,:,:,-1:]
  features =  features[:,:,:,0:-1]
  
  images = tf.cast(
      tf.map_fn(preprocessing.mean_image_addition, features),
      tf.uint8)
  

  network = model_generator(params['num_classes'],
                            params['pre_trained_model'],
                            params['batch_norm_decay'])

  logits = network(features,region, mode == tf.estimator.ModeKeys.TRAIN)

  pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

  pred_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                   [pred_classes, params['batch_size'], params['num_classes']],
                                   tf.uint8)
  print(pred_classes.get_shape().as_list())
  predictions = {
      'classes': pred_classes,
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
      'decoded_labels': pred_decoded_labels
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
    
    predictions_without_decoded_labels = predictions.copy()
    del predictions_without_decoded_labels['decoded_labels']
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'preds': tf.estimator.export.PredictOutput(
                predictions_without_decoded_labels)
        })

  gt_decoded_labels = tf.py_func(preprocessing.decode_labels,
                                 [labels, params['batch_size'], params['num_classes']], tf.uint8)

  labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.

  logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
  labels_flat = tf.reshape(labels, [-1, ])

  valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
  valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
  valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

  preds_flat = tf.reshape(pred_classes, [-1, ])
  valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
  confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=params['num_classes'])

  predictions['valid_preds'] = valid_preds
  predictions['valid_labels'] = valid_labels
  predictions['confusion_matrix'] = confusion_matrix

  label0_weight =1  # background
  label1_weight = 8 # solidline
  label2_weight = 8 # dashline
  label3_weight = 8 # arrow
  label4_weight = 8 # otherline
  label5_weight = 10 # stopline
  label6_weight = 2 # backpoints
  not_ignore_mask = tf.to_float(tf.equal(valid_labels, 0)) * label0_weight + \
                    tf.to_float(tf.equal(valid_labels, 1)) * label1_weight + \
                    tf.to_float(tf.equal(valid_labels, 2)) * label2_weight + \
                    tf.to_float(tf.equal(valid_labels, 3)) * label3_weight + \
                    tf.to_float(tf.equal(valid_labels, 4)) * label4_weight + \
                    tf.to_float(tf.equal(valid_labels, 5)) * label5_weight + \
                    tf.to_float(tf.equal(valid_labels, 6)) * label6_weight

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=valid_logits, labels=valid_labels,weights=not_ignore_mask)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  if not params['freeze_batch_norm']:
    train_var_list = [v for v in tf.trainable_variables()]
  else:
    train_var_list = [v for v in tf.trainable_variables()
                      if 'beta' not in v.name and 'gamma' not in v.name]

  # Add weight decay to the loss.
  with tf.variable_scope("total_loss"):
    loss = cross_entropy + params.get('weight_decay', _WEIGHT_DECAY) * tf.add_n(
        [tf.nn.l2_loss(v) for v in train_var_list])
    #loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf.summary.image('images',
                     tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
                     max_outputs=params['tensorboard_images_max_outputs'])  # Concatenate row-wise.

    global_step = tf.train.get_or_create_global_step()

    if params['learning_rate_policy'] == 'piecewise':
      # Scale the learning rate linearly with the batch size. When the batch size
      # is 128, the learning rate should be 0.1.
      initial_learning_rate = 0.1 * params['batch_size'] / 128
      batches_per_epoch = params['num_train'] / params['batch_size']
      # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
      boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
      values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
      learning_rate = tf.train.piecewise_constant(
          tf.cast(global_step, tf.int32), boundaries, values)
    elif params['learning_rate_policy'] == 'poly':
      learning_rate = tf.train.polynomial_decay(
          params['initial_learning_rate'],
          tf.cast(global_step, tf.int32) - params['initial_global_step'],
          params['max_iter'], params['end_learning_rate'], power=params['power'])
    else:
      raise ValueError('Learning rate policy must be "piecewise" or "poly"')

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)
    #train_op = optimizer.minimize(loss)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      valid_labels, valid_preds)
  mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, params['num_classes'])
  metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_px_accuracy')
  tf.summary.scalar('train_px_accuracy', accuracy[1])

  def compute_mean_iou(total_cm, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(params['num_classes']):
      tf.identity(iou[i], name='train_iou_class{}'.format(i))
      tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result

  train_mean_iou = compute_mean_iou(mean_iou[1])

  tf.identity(train_mean_iou, name='train_mean_iou')
  tf.summary.scalar('train_mean_iou', train_mean_iou)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)
