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
Usage: Network---LeNet5 
"""

from utils.generate_unique_id import generate_unique_id
import tensorflow as tf
import numpy as np
import cv2

tf.enable_eager_execution()

def LeNet5(path_to_image):
    img_dir = path_to_image
    img_data = np.array(cv2.imread(img_dir, 1)).astype('float32')
    img_data = cv2.resize(img_data, (224, 224))
    img_data = tf.reshape(img_data, [1, img_data.shape[0], img_data.shape[1], 3])

    feature = tf.layers.Conv2D(filters=6, kernel_size=[5, 5], strides=(1, 1), padding='same',activation=tf.nn.relu)(img_data)
    feature = tf.layers.MaxPooling2D(pool_size= (2, 2), strides= (2, 2), padding='same')(feature)
    feature = tf.layers.Conv2D(filters=16, kernel_size=[5, 5], strides=(1, 1), padding= 'valid',activation=tf.nn.relu)(feature)
    feature = tf.layers.MaxPooling2D(pool_size= (2, 2), strides= (2, 2), padding='same')(feature)
    feature = tf.layers.Conv2D(filters= 120, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=tf.nn.relu)(feature)

    feature = feature.numpy()
    feature = feature.astype('uint8')
    feature = feature[0][:, :, 0]
    result_dir = 'static/images/extractor/' + 'LeNet5_' + generate_unique_id()
    cv2.imwrite(result_dir,feature)

    return result_dir
