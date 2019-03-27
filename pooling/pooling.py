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
By Mingqi, Yuan, 2019/3/27
Usage: Pooling
"""
from utils.generate_unique_id import generate_unique_id
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

tf.enable_eager_execution()

def avg_pooling(strides,
                padding,
                pool_size,
                path_to_image):
    data = {
        'path_avg_pooling': [],
        'path_max_pooling': [],
        'histogram_avg': [],
        'histogram_max': [],
    }

    img_dir = path_to_image
    img_data = np.array(cv2.imread(img_dir, 0)).astype('float32')
    img_data = tf.reshape(img_data, [1, img_data.shape[0], img_data.shape[1], 1])

    feature_map = tf.layers.average_pooling2d(inputs= img_data,
                                              pool_size= [pool_size, pool_size],
                                              strides= [strides, strides],
                                              padding= padding)
    feature_map = np.matrix(feature_map.numpy())
    feature_map = feature_map.astype('uint8')

    result_dir = 'static/images/' + 'avg_pooling' + generate_unique_id()
    cv2.imwrite(result_dir, feature_map)
    im = cv2.imread(result_dir, 0)

    data['path_avg_pooling'] = result_dir
    data['histogram_avg'] = plt.hist(im.ravel(), 256, [0, 256])[0]

    return data['path_avg_pooling'], data['histogram_avg']

def max_pooling(strides,
                padding,
                pool_size,
                path_to_image):
    data = {
        'path_avg_pooling': [],
        'path_max_pooling': [],
        'histogram_avg': [],
        'histogram_max': [],
    }

    img_dir = path_to_image
    img_data = np.array(cv2.imread(img_dir, 0)).astype('float32')
    img_data = tf.reshape(img_data, [1, img_data.shape[0], img_data.shape[1], 1])


    feature_map = tf.layers.max_pooling2d(inputs= img_data,
                                              pool_size= [pool_size, pool_size],
                                              strides= [strides, strides],
                                              padding= padding)

    feature_map = np.matrix(feature_map.numpy())

    feature_map = feature_map.astype('uint8')

    result_dir = 'static/images/' + generate_unique_id()
    cv2.imwrite(result_dir, feature_map)
    im = cv2.imread(result_dir, 0)

    data['path_max_pooling'] = result_dir
    data['histogram_max'] = plt.hist(im.ravel(), 256, [0, 256])[0]

    return data['path_max_pooling'], data['histogram_max']


