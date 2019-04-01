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
Usage: Network---AlexNet
"""

from utils.generate_unique_id import generate_unique_id
import tensorflow as tf
import numpy as np
import cv2

tf.enable_eager_execution()

def AlexNet(path_to_image):
    img_dir = path_to_image
    img_data = np.array(cv2.imread(img_dir, 1)).astype('float32')
    img_data = cv2.resize(img_data, (224, 224))
    img_data = tf.reshape(img_data, [1, img_data.shape[0], img_data.shape[1], 3])

    # Define the converlutional layer 1
    conv1 = tf.layers.Conv2D(filters=96, kernel_size=[11, 11], strides=[2, 2], activation=tf.nn.relu,
                                use_bias=True, padding='valid')(img_data)

    # Define the pooling layer 1
    pooling1 = tf.layers.AveragePooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(conv1)

    # Define the standardization layer 1
    stand1 = tf.layers.BatchNormalization(axis=3)(pooling1)

    # Define the converlutional layer 2
    conv2 = tf.layers.Conv2D(filters=256, kernel_size=[5, 5], strides=[2, 2], activation=tf.nn.relu,
                                use_bias=True, padding='valid')(stand1)

    # Defien the pooling layer 2
    pooling2 = tf.layers.AveragePooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(conv2)

    # Define the standardization layer 2
    stand2 = tf.layers.BatchNormalization(axis=3)(pooling2)

    # Define the converlutional layer 3
    conv3 = tf.layers.Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], activation=tf.nn.relu,
                                use_bias=True, padding='valid')(stand2)

    # Define the converlutional layer 4
    conv4 = tf.layers.Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], activation=tf.nn.relu,
                                use_bias=True, padding='valid')(conv3)

    # Define the converlutional layer 5
    conv5 = tf.layers.Conv2D(filters=382, kernel_size=[3, 3], strides=[2, 2], activation=tf.nn.relu,
                                use_bias=True, padding='valid')(conv4)

    feature = conv5.numpy()
    feature = feature.astype('uint8')
    feature = feature[0][:, :, 0]

    result_dir = 'static/images/extractor/' + 'AlexNet_' + generate_unique_id()
    cv2.imwrite(result_dir,feature)

    return result_dir
