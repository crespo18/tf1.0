#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, weight):
    return tf.nn.conv2d(x, weight, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


mnist = input_data.read_data_sets('../mnist/mnist_gz', one_hot=True)
print('shape:\n', mnist.train.images.shape, mnist.train.labels.shape, mnist.validation.images.shape, mnist.validation.labels.shape, mnist.test.images.shape, mnist.test.labels.shape)

placeholder_x = tf.placeholder(tf.float32, [None, 784])
placeholder_y_ = tf.placeholder(tf.float32, [None, 10])
placeholder_keep_prob = tf.placeholder(tf.float32)

print('placeholder: \n', placeholder_x, placeholder_y_, placeholder_keep_prob)


x_image = tf.reshape(placeholder_x, [-1,28,28,1])
weight_conv1 = weight_variable([5,5,1,32])
bias_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + bias_conv1)
print('h_conv1: \n', h_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print('h_pool: \n', h_pool1)


weight_conv2 = weight_variable([5,5,32,64])
bias_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_conv2) + bias_conv2)
print('h_conv2: \n', h_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print('h_pool2: \n', h_pool2)


weight_fc = weight_variable([7*7*64, 1024])
bias_fc = bias_variable([1024])
flat_fc = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc = tf.nn.relu(tf.matmul(flat_fc, weight_fc) + bias_fc)
print('h_fc: \n', h_fc)

h_dropout = tf.nn.dropout(h_fc, placeholder_keep_prob)
print('h_dropout: \n', h_dropout)

weight_out = weight_variable([1024, 10])
bias_out = weight_variable([10])
y_out = tf.nn.softmax(tf.matmul(h_dropout, weight_out) + bias_out)
print('y_out: \n', y_out)


cross_entropy = -tf.reduce_sum(placeholder_y_ * tf.log(y_out))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_predict = tf.equal(tf.argmax(placeholder_y_, 1), tf.argmax(y_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for idx in range(1000):
        batch_x,batch_y = mnist.train.next_batch(50)
        if idx % 100 == 0:
            print('train  step: ', idx, " accuracy: ", sess.run(accuracy, feed_dict={placeholder_x:batch_x, placeholder_y_:batch_y, placeholder_keep_prob:1.0}))
        sess.run(train_step, feed_dict={placeholder_x:batch_x, placeholder_y_:batch_y, placeholder_keep_prob:0.5})

    print('test accuracy: ', sess.run(accuracy, feed_dict={placeholder_x:mnist.test.images, placeholder_y_:mnist.test.labels, placeholder_keep_prob:1.0}))
