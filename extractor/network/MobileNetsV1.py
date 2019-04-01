"""
MIT License
Copyright (c) 2019 XinZhe  Wang
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Encoding = UTF-8
By XinZhe, Wang, 2019/3/31
Usage: Network---MobileNetsV1 
"""
from utils.generate_unique_id import generate_unique_id
import tensorflow as tf 
import numpy as np
import cv2

#init some parameters
stddev = 0.009
weight_decay=0.00004
weights_init = tf.truncated_normal_initializer(stddev=stddev)
regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
trainable = True
number_classes = 1000

tf.enable_eager_execution()

#MbileNetsV1
def MbileNetsV1(path_to_image, number_classes):
    img_dir = path_to_image
    img_data = np.array(cv2.imread(img_dir, 1)).astype('float32')
    img_data = cv2.resize(img_data, (224, 224))
    img_data = tf.reshape(img_data, [1, img_data.shape[0], img_data.shape[1], 3])
    print img_data
    with tf.variable_scope('mobilenetv1'):
         #conv0
         net = tf.layers.Conv2D(32, [3, 3] , strides=2, padding='SAME', kernel_initializer=weights_init, name='conv0')(img_data)
         net = tf.layers.BatchNormalization(axis=-1)(net)
         net = tf.nn.relu6(net, name='conv0/relu6')
         #conv1
         net = depthwise_separable_conv2d(net, [3,3], strides=1, channel_multiplier=1, padding='SAME')
         net = point_conv2d(net, 64, [1, 1], strides=1)
         #conv2
         net = depthwise_separable_conv2d(net, [3,3], strides=2, channel_multiplier=1, padding='SAME')
         net = point_conv2d(net, 128, [1, 1], strides=1)
         #conv3
         net = depthwise_separable_conv2d(net, [3,3], strides=1, channel_multiplier=1, padding='SAME')
         net = point_conv2d(net, 128, [1, 1], strides=1)
         #conv4
         net = depthwise_separable_conv2d(net, [3,3], strides=2, channel_multiplier=1, padding='SAME')
         net = point_conv2d(net, 256, [1, 1], strides=1)
         #conv5
         net = depthwise_separable_conv2d(net, [3,3], strides=1, channel_multiplier=1, padding='SAME')
         net = point_conv2d(net, 256, [1, 1], strides=1)
         #conv6
         net = depthwise_separable_conv2d(net, [3,3], strides=2, channel_multiplier=1, padding='SAME')
         net = point_conv2d(net, 512, [1, 1], strides=1)
         #conv7-conv11
         for _ in range(5):
             net = depthwise_separable_conv2d(net, [3,3], strides=1, channel_multiplier=1, padding='SAME')
             net = point_conv2d(net, 512, [1, 1], strides=1)
         #conv12
         net = depthwise_separable_conv2d(net, [3,3], strides=2, channel_multiplier=1, padding='SAME')
         net = point_conv2d(net, 1024, [1, 1], strides=1)
         #conv13  
         net = depthwise_separable_conv2d(net, [3,3], strides=1, channel_multiplier=1, padding='SAME')
         net = point_conv2d(net, 1024, [1, 1], strides=1)
         
         feature = net.numpy()
         feature = feature.astype('uint8')
         feature = feature[0][:, :, 0]
         result_dir = 'static/images/' + 'MbileNetsV1' + generate_unique_id()
         cv2.imwrite(result_dir,feature)
         
         #avg_pool  shape -->[1,1,1024]
         net = global_avg(net)
         #dropout  
         net = tf.layers.Dropout(rate=0.001)(net)
         #fc        shape -->[1,1,number_classes]
         net = tf.layers.Conv2D(number_classes, [1, 1] , strides=2, padding='SAME', name='fc')(net)
         #SpatialSqueeze
         logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
         return result_dir
       

#define depthwise_separable_conv2d
def depthwise_separable_conv2d(inputs, size, strides, channel_multiplier,  padding):
    in_channel=inputs.get_shape().as_list()[-1]
    w = tf.get_variable('w', [size[0], size[1], in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    net=tf.nn.depthwise_conv2d(input=inputs, filter=w, strides=[1, strides, strides, 1], padding=padding, name='depth_conv')
    net = tf.layers.BatchNormalization(axis=-1, name='depth_conv/bn')(net)
    net = tf.nn.relu6(net, name='depth_conv/relu')
    return net
 #define  point_conv2d
def point_conv2d(inputs,filters, size, strides):
    net = tf.layers.Conv2D(filters, [size[0], size[1]] , strides=strides, padding='SAME',kernel_initializer=weights_init, 
                       kernel_regularizer=regularizer, name='point_cnv')(inputs) 
    net = tf.layers.BatchNormalization(axis=-1, name='point_cnv/bn')(net)
    net = tf.nn.relu6(net, name='point_cnv/relu')
    return net     

def global_avg(x):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         

