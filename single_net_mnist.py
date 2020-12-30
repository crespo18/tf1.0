#!/usr/bin/env python
#-*- coding:utf-8 -*1

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



mnist = input_data.read_data_sets("../mnist/mnist_gz", one_hot=True)
print('train shape:\n', mnist.train.images.shape, mnist.train.labels.shape)
print('validation shape:\n', mnist.validation.images.shape, mnist.validation.labels.shape)
print('test shape:\n', mnist.test.images.shape, mnist.test.labels.shape)

#batch_x,batch_y = mnist.train.next_batch(100)
#print('train.batch shape: \n', batch_x.shape, batch_y.shape)
#print('train.batch data: \n', batch_x[0],'\n', batch_y[0])


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
print('placeholder x: ',x)
print('placeholder y_: ',y_)


w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
print('variable w: ', w)
print('variable b: ', b)


y = tf.nn.softmax(tf.matmul(x,w) + b)
print('predict y: ', y)


cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for idx in range(1000):
    batch_x,batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_x, y_:batch_y})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print('test accuracy: ', sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
