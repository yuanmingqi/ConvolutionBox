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
# -*- coding:"utf-8" -*-
#!/usr/bin/env python 
"""
    @Time: 2019/4/1 17:18
    @Author: qiuzhongxi
    @File Name: GoogLeNet.py
    @Software: PyCharm
"""
from utils.generate_unique_id import generate_unique_id
import tensorflow as tf
import numpy as np
import cv2

decay = 0.9
regularizer = tf.contrib.layers.l2_regularizer(decay)
initializer = tf.contrib.layers.xavier_initializer()
num_classes = 1000

tf.enable_eager_execution()



def conv(inputs, deep, kener_size=(3,3), strides=2, name="conv", padding="SAME"):
    convlution=tf.layers.Conv2D(deep, kener_size, strides, padding=padding, name=name,
                         kernel_regularizer=regularizer,
                         kernel_initializer=initializer)(inputs)
    activation = tf.nn.relu(convlution)
    return activation

def inception(inputs, conv11_dep, conv33_reduce_dep, conv33_dep,
              conv55_reduce_dep, conv55_dep, pool_proj_dep, name="inception"):
    with tf.variable_scope(name):
        conv11 = conv(inputs, conv11_dep, [1,1], 1, padding="SAME", name="conv11")
        conv33_reduce = conv(inputs, conv33_reduce_dep, [1,1], 1, padding="SAME", name="conv11")
        conv33 = conv(conv33_reduce, conv33_dep, [3,3], 1, padding="SAME", name="conv33")
        conv55_reduce = conv(inputs, conv55_reduce_dep, [1,1], 1, padding="SAME", name="conv55_reduce")
        conv55 = conv(conv55_reduce, conv55_dep, [5,5], 1, padding="SAME", name="conv55")
        pool = tf.layers.MaxPooling2D([3, 3], 1, padding="SAME", name="pool")(inputs)
        pool_proj = conv(pool, pool_proj_dep, [1,1],1,padding="SAME", name="pool_proj")
        return tf.concat([conv11, conv33, conv55, pool_proj], axis=3)

def GoogLeNet(path_to_image):
    img_dir = path_to_image
    img_data = np.array(cv2.imread(img_dir, 1)).astype('float32')
    img_data = cv2.resize(img_data, (224, 224))
    img_data = tf.reshape(img_data, [1, img_data.shape[0], img_data.shape[1], 3])
    #conv0 -pool1
    net = conv(img_data, 64, [7,7], 2, name="conv0")
    net = tf.layers.MaxPooling2D([3, 3], 2, padding="SAME", name="poo0")(net)
    net = conv(net, 192, (3,3), 1, name="conv1")
    net = tf.layers.MaxPooling2D([3, 3],2, name="pool1")(net)
    #inception3a - inception3b
    net = inception(net, 64, 96, 128, 16, 32, 32, name="inception_3a")
    net = inception(net, 128, 128, 192, 32, 96, 64, name="inception_3b")
    net = tf.layers.MaxPooling2D([3, 3], 2, padding="SAME")(net)
    #inception4a - inception4e
    net = inception(net, 192, 96, 208, 16, 48, 64, name="inception_4a")
    net = inception(net, 160, 112, 224, 24, 64, 64, name="inception_4b")
    net = inception(net, 128, 128, 256, 24, 64, 64, name="inception_4c")
    net = inception(net, 112, 144, 288, 32, 64, 64, name="inception_4d")
    net = inception(net, 256, 160, 320, 32, 128, 128, name="inception_4e")
    net = tf.layers.MaxPooling2D([3, 3], 2, padding="SAME")(net)
    #inception5a - inception5b
    net = inception(net, 256, 160, 320, 32, 128, 128, name="inception_5a")
    net = inception(net, 384, 192, 384, 48, 128, 128, name="inception_5b")
    feature = net.numpy()
    feature = feature.astype('uint8')
    feature = feature[0][:, :, 0]
    result_dir = 'static/images/' + 'GoogLeNet' + generate_unique_id()
    #result_dir = 'test/' + 'GoogLeNet' + generate_unique_id()
    cv2.imwrite(result_dir, feature)
    #avg pool
    net = tf.layers.average_pooling2d(net, [7, 7], 1, padding="SAME")
    net = tf.layers.Dropout(0.6)(net)
    logit = tf.layers.Dense(num_classes, activation=tf.nn.softmax)(net)
    return  result_dir

if __name__ == "__main__":
    import os
    print(os.path.abspath(GoogLeNet("../../static/lena.jpg")))