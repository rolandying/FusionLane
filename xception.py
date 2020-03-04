from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import range

import tensorflow as tf
slim = tf.contrib.slim

'''
==================================================================
Based on the Xception Paper (https://arxiv.org/pdf/1610.02357.pdf)
==================================================================
REGULARIZATION CONFIGURATION:
- weight_decay: 1e-5
- dropout: no dropout
- aux_loss: no aux loss
OPTIMIZATION CONFIGURATION (for Google JFT Dataset):
- optimizer: RMSProp
- momentum: 0.9
- initial_learning_rate: 0.001
- learning_rate_decay: 0.9 every 3/350 epochs (every 3M images; total 350M images per epoch)
'''
def xception(inputs,
             region,
             is_training,
             num_classes=6,
             scope='xception'):

    '''
    The Xception Model!
    
    Note:
    The padding is included by default in slim.conv2d to preserve spatial dimensions.
    INPUTS:
    - inputs(Tensor): a 4D Tensor input of shape [batch_size, height, width, num_channels]
    - num_classes(int): the number of classes to predict
    - is_training(bool): Whether or not to train
    OUTPUTS:
    - logits (Tensor): raw, unactivated outputs of the final layer
    - end_points(dict): dictionary containing the outputs for each layer, including the 'Predictions'
                        containing the probabilities of each output.
    '''
    with tf.variable_scope('Xception') as sc:
        end_points_collection = sc.name + '_end_points'
        
        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1),\
         slim.arg_scope([slim.separable_conv2d, slim.conv2d, slim.avg_pool2d], outputs_collections=[end_points_collection]),\
         slim.arg_scope([slim.batch_norm], is_training=is_training):

            #===========ENTRY FLOW==============

            net_a = slim.conv2d(region, 8, [3,3], padding='same', scope='a_layer1_conv')
            net_a = tf.nn.relu(net_a, name = 'a_layer1_relu')
            net_a = slim.batch_norm(net_a, scope = 'a_layer1_bn')

            net_a = slim.conv2d(net_a, 16, [3,3], padding='same', scope='a_layer2_conv')
            net_a = tf.nn.relu(net_a, name = 'a_layer2_relu')
            net_a = slim.batch_norm(net_a, scope = 'a_layer2_bn')

            net_a = slim.separable_conv2d(net_a, 32, [3,3], stride=2,  padding = 'same', scope='a_layer3_dws_conv')
            net_a = tf.nn.relu(net_a, name = 'a_layer3_relu')
            net_a = slim.batch_norm(net_a, scope = 'a_layer3_bn')

            net = slim.conv2d(inputs, 24, [3,3], padding='same', scope='block0_conv1')
            net = tf.nn.relu(net, name='block0_relu1')
            net = slim.batch_norm(net, scope='block0_bn1')

            net = slim.conv2d(net, 48, [3,3], padding='same', scope='block0_conv2')
            net = tf.nn.relu(net, name='block0_relu2')
            net = slim.batch_norm(net, scope='block0_bn2')

            net = slim.separable_conv2d(net, 96, [3,3], stride=2,  padding = 'same', scope='block0_dws_conv1')
            net = tf.nn.relu(net, name='block0_relu3')
            net = slim.batch_norm(net, scope='block0_bn3')
            net = tf.nn.relu(net, name='block0_relu3')

            net = tf.concat([net,net_a],3,name='fuse1')

            residual = slim.conv2d(net, 192, [1,1], stride=2, scope='block0_res_conv')
            residual = slim.batch_norm(residual, scope='block0_res_bn')

            #Block 1

            net_a = slim.separable_conv2d(net_a, 64, [3,3], padding = 'same', scope='a_layer4_dws_conv')
            net_a = tf.nn.relu(net_a, name = 'a_layer4_relu')
            net_a = slim.batch_norm(net_a, scope = 'a_layer4_bn')

            net_a = slim.separable_conv2d(net_a, 64, [3,3], stride=2,  padding = 'same', scope='a_layer5_dws_conv')
            net_a = tf.nn.relu(net_a, name = 'a_layer5_relu')
            net_a = slim.batch_norm(net_a, scope = 'a_layer5_bn')
            
            net = slim.separable_conv2d(net, 192, [3,3], padding = 'same', scope='block1_dws_conv1')
            net = tf.nn.relu(net, name='block1_relu1')
            net = slim.batch_norm(net, scope='block1_bn1')
            
            net = slim.separable_conv2d(net, 192, [3,3], padding = 'same', scope='block1_dws_conv2')
            net = tf.nn.relu(net, name='block1_relu2')
            net = slim.batch_norm(net, scope='block1_bn2')

            net = slim.separable_conv2d(net, 192, [3,3], stride=2,  padding = 'same', scope='block1_dws_conv3')
            net = tf.nn.relu(net, name='block1_relu3')
            net = slim.batch_norm(net, scope='block1_bn3')
            
            net = tf.add(net, residual, name='block1_add')

            low_level_features = net
            
            net = tf.concat([net,net_a],3,name='fuse2')

            residual = slim.conv2d(net, 384, [1,1], stride=2, scope='block1_res_conv')
            residual = slim.batch_norm(residual, scope='block1_res_bn')

            #Block 2

            net_a = slim.separable_conv2d(net_a, 128, [3,3], padding = 'same', scope='a_layer6_dws_conv')
            net_a = tf.nn.relu(net_a, name = 'a_layer6_relu')
            net_a = slim.batch_norm(net_a, scope = 'a_layer6_bn')

            net_a = slim.separable_conv2d(net_a, 128, [3,3], stride=2,  padding = 'same', scope='a_layer7_dws_conv')
            net_a = tf.nn.relu(net_a, name = 'a_layer7_relu')
            net_a = slim.batch_norm(net_a, scope = 'a_layer7_bn')

            net = slim.separable_conv2d(net, 384, [3,3], padding = 'same', scope='block2_dws_conv1')
            net = tf.nn.relu(net, name='block2_relu1')
            net = slim.batch_norm(net, scope='block2_bn1')

            net = slim.separable_conv2d(net, 384, [3,3], padding = 'same', scope='block2_dws_conv2')
            net = tf.nn.relu(net, name='block2_relu2')
            net = slim.batch_norm(net, scope='block2_bn2')        

            net = slim.separable_conv2d(net, 384, [3,3], stride=2,  padding = 'same', scope='block2_dws_conv3')
            net = tf.nn.relu(net, name='block2_relu3')
            net = slim.batch_norm(net, scope='block2_bn3')        
            
            net = tf.add(net, residual, name='block2_add')

            net = tf.concat([net,net_a],3,name='fuse3')

            #===========MIDDLE FLOW===============
            for i in range(8):
                block_prefix = 'block%s_' % (str(i + 3))

                residual = net
                net = slim.separable_conv2d(net, 512, [3,3], padding = 'same', scope=block_prefix+'dws_conv1')
                net = tf.nn.relu(net, name=block_prefix+'relu1')
                net = slim.batch_norm(net, scope=block_prefix+'bn1')
                net = slim.separable_conv2d(net, 512, [3,3], padding = 'same', scope=block_prefix+'dws_conv2')
                net = tf.nn.relu(net, name=block_prefix+'relu2')
                net = slim.batch_norm(net, scope=block_prefix+'bn2')
                net = slim.separable_conv2d(net, 512, [3,3], padding = 'same', scope=block_prefix+'dws_conv3')
                net = tf.nn.relu(net, name=block_prefix+'relu3')
                net = slim.batch_norm(net, scope=block_prefix+'bn3')
                net = tf.add(net, residual, name=block_prefix+'add')


            #========EXIT FLOW============

            net = slim.separable_conv2d(net, 512, [3,3], padding = 'same', scope='block11_dws_conv1')
            net = tf.nn.relu(net, name='block11_relu1')
            net = slim.batch_norm(net, scope='block11_bn1')

            net = slim.separable_conv2d(net, 512, [3,3], padding = 'same', scope='block11_dws_conv2')
            net = tf.nn.relu(net, name='block11_relu2')
            net = slim.batch_norm(net, scope='block11_bn2')

            net = slim.separable_conv2d(net, 1024, [3,3], stride=2, padding = 'same', scope='block11_dws_conv3')
            net = tf.nn.relu(net, name='block11_relu3')
            logits = slim.batch_norm(net, scope='block11_bn3')
            
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            end_points['Logits'] = logits

        return logits, end_points, low_level_features

def xception_arg_scope(weight_decay=0.00001,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001):
  '''
  The arg scope for xception model. The weight decay is 1e-5 as seen in the paper.
  INPUTS:
  - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
  - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
  - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.
  OUTPUTS:
  - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
  '''
  # Set weight_decay for weights in conv2d and separable_conv2d layers.
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=None,
                      activation_fn=None):
            
    # Set parameters for batch_norm. Note: Do not set activation function as it's preset to None already.
    with slim.arg_scope([slim.batch_norm],
                        decay=batch_norm_decay,
                        epsilon=batch_norm_epsilon) as scope:
      return scope