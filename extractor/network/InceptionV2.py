# -*- coding:"utf-8" -*-
#!/usr/bin/env python 
"""
    @Time: 2019/4/1 19:51
    @Author: qiuzhongxi
    @File Name: InceptionV2.py
    @Software: PyCharm
"""
from utils.generate_unique_id import generate_unique_id
import tensorflow as tf
import numpy as np
import cv2

decay = 0.9
regularizer = tf.contrib.layers.l2_regularizer(decay)
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
num_classes = 1000

tf.enable_eager_execution()

def conv(inputs, deep, kener_size=(3,3), strides=2, stddev=0.1,name="conv", padding="SAME"):
    convlution=tf.layers.Conv2D(deep, kener_size, strides, padding=padding, name=name,
                         kernel_regularizer=regularizer,
                         kernel_initializer=trunc_normal(stddev))(inputs)
    activation = tf.nn.relu(convlution)
    return activation
def inception_a(inputs, conv11_dep=64, conv33_reduce_dep=64, conv33_dep=192, double_conv33_reduce_dep=64,
                double_cnv33_dep=64, pool_proj_dep=64, strides=1,
                 pool_strides=1, pool_type="max",stddev=0.1,
                is_conv11=True, name="inception"):
    with tf.variable_scope(name):
        if is_conv11:
            conv11 = conv(inputs, conv11_dep, [1,1], 1, stddev=stddev, name="conv11")
        conv33_reduce = conv(inputs, conv33_reduce_dep, [1,1], 1, stddev=stddev)
        conv33 = conv(conv33_reduce, conv33_dep, [3,3], strides, name="conv33")
        double_conv33_reduce = conv(inputs, double_conv33_reduce_dep, [1,1], 1, stddev=stddev)
        double_conv33_1 = conv(double_conv33_reduce, double_cnv33_dep, [3, 3], 1, name="double_conv33_1")
        double_conv33_2 = conv(double_conv33_1, double_cnv33_dep, [3,3], strides, name="double_conv33_2")
        pool = tf.layers.MaxPooling2D([3, 3], pool_strides, padding="SAME", name="max_pool")(inputs)
        if pool_type == "avg":
            pool = tf.layers.AveragePooling2D([3, 3], pool_strides, padding="SAME", name="avg_pool")(inputs)
        pool_proj = conv(pool, pool_proj_dep, [1,1], 1, stddev=stddev)
        if is_conv11:
            return tf.concat([conv11, conv33, double_conv33_2, pool_proj], 3)
        else:
            return tf.concat([conv33, double_conv33_2, pool_proj], 3)



def InceptionV2(path_to_image):
    img_dir = path_to_image
    img_data = np.array(cv2.imread(img_dir, 1)).astype('float32')
    img_data = cv2.resize(img_data, (224, 224))
    img_data = tf.reshape(img_data, [1, img_data.shape[0], img_data.shape[1], 3])
    net = conv(img_data, 64, [7, 7], 2, name="conv0", padding="SAME")
    net = tf.layers.MaxPooling2D([3,3], strides=2, name="pool0")(net)
    net = conv(net, 64, [1, 1], 1, name="conv1_reduce")
    net = conv(net, 192, [3, 3], 1, name="conv1")
    net = tf.layers.MaxPooling2D([3, 3], 2, name="pool1", padding="SAME")(net)
    net = inception_a(net, 64, 64, 64, 64, 96, 32, pool_type="avg", stddev=0.09, name="inception3a")
    net = inception_a(net, 64, 64, 96, 64, 96, 64, pool_type="avg", name="inception3b")
    net = inception_a(net, 0, 128, 160, 64, 96, 64, 2, 2, is_conv11=False, name="inception3c")
    net = inception_a(net, 224, 64, 96, 96, 128, 128, pool_type="avg", name="inception4a")
    net = inception_a(net, 192, 96, 128, 96, 128, 128, pool_type="avg", name="inception4b")
    net = inception_a(net, 160, 128, 160, 128, 160, 128, pool_type="avg", name="inception4c")
    net = inception_a(net, 96, 128, 192, 160, 192, 128, pool_type="avg", name="inception4d")
    net = inception_a(net, 0, 128, 192, 192, 256, 128, 2, 2, is_conv11=False,name="inception4e")
    net = inception_a(net, 352, 192, 320, 160, 224, 128, pool_type="avg", name="inception5a")
    net = inception_a(net, 352, 192, 320, 192, 224, 128, name="inception5b")
    #print(net.shape)
    feature = net.numpy()
    feature = feature.astype('uint8')
    feature = feature[0][:, :, 0]
    result_dir = 'static/images/' + 'GoogLeNet' + generate_unique_id()
    #result_dir = 'test/' + 'InceptionV2' + generate_unique_id()
    cv2.imwrite(result_dir, feature)
    net = tf.layers.AveragePooling2D([7,7], 1, name="avg_pool")(net)
    logit = tf.layers.Dense(num_classes,activation=tf.nn.softmax, name="softmax")(net)
    return result_dir
if __name__ == "__main__":
    import os
    print(os.path.abspath(InceptionV2("../../static/lena.jpg")))

