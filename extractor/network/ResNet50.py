"""
MIT License

Copyright (c) 2019 Mingqi Yuan

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
By Mingqi, Yuan, 2019/3/31
Usage: Network---ResNet50
"""

from utils.generate_unique_id import generate_unique_id
import tensorflow as tf
import numpy as np
import cv2

tf.enable_eager_execution()

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal', activation=tf.nn.relu,
                      name=conv_name_base + '2a')(input_tensor)
    x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)


    x = tf.layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal', activation=tf.nn.relu,
                      name=conv_name_base + '2b')(x)
    x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    

    x = tf.layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal', activation=tf.nn.relu,
                      name=conv_name_base + '2c')(x)
    x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = tf.add(x, input_tensor)
    
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.layers.Conv2D(filters1, (1, 1), strides=strides, activation=tf.nn.relu,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    

    x = tf.layers.Conv2D(filters2, kernel_size, padding='same', activation=tf.nn.relu,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    

    x = tf.layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal', activation=tf.nn.relu,
                      name=conv_name_base + '2c')(x)
    x = tf.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = tf.layers.Conv2D(filters3, (1, 1), strides=strides, activation=tf.nn.relu,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = tf.layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = tf.add(x, shortcut)
    
    return x

def ResNet50(path_to_image):
    img_dir = path_to_image
    img_data = np.array(cv2.imread(img_dir, 1)).astype('float32')
    img_data = cv2.resize(img_data, (224, 224))
    img_data = tf.reshape(img_data, [1, img_data.shape[0], img_data.shape[1], 3])

    #x = keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_data)
    x = tf.layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(img_data)
    x = tf.layers.BatchNormalization(axis=3, name='bn_conv1')(x)

    #x = keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = tf.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = tf.layers.AveragePooling2D(name='avg_pool', pool_size=[7, 7], strides=[2, 2])(x)

    feature = x.numpy()
    feature = feature.astype('uint8')
    feature = np.reshape(feature, [32, 64])

    result_dir = 'static/images/' + 'ResNet50' + generate_unique_id()
    cv2.imwrite(result_dir,feature)

    return result_dir

