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
By Mingqi, Yuan, 2019/3/26
Usage: Standard convolution
"""

from utils.generate_unique_id import generate_unique_id
import tensorflow as tf
import numpy as np
import cv2

tf.enable_eager_execution()

def standard_conv(strides,
                  padding,
                  filter_size,
                  path_to_image):
    data = {
        'path_conv1': [],
        'path_conv2': [],
        'path_conv3': [],
        'path_conv4': [],
        'histogram_1C': [],
        'histogram_2C': [],
        'histogram_3C': []
    }

    img_dir = path_to_image
    img_data = np.array(cv2.imread(img_dir, 1)).astype('float32')
    img_data = tf.reshape(img_data, [1, img_data.shape[0], img_data.shape[1], 3])

    for i in range(4):
        img_data = tf.layers.conv2d(inputs= img_data,
                                    filters= 1,
                                    kernel_size= [filter_size, filter_size],
                                    strides= [strides, strides],
                                    padding= padding)

        img_data = np.matrix(img_data.numpy())
        img_data = img_data.astype('uint8')

        result_dir = 'static/images/' + str(i) + generate_unique_id()
        cv2.imwrite(result_dir, img_data)
        key = 'path_conv' + str(i+1)
        # save the path of each feature map
        data[key] = result_dir

        img_data = tf.reshape(img_data, [1, img_data.shape[0], img_data.shape[1], 1])
        img_data = tf.cast(img_data, dtype= tf.float32)
    return data

