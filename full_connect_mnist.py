#!/usr/bin/env python
#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class FullConnectMnistModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def load_mnist_data(self, file_path):
        mnist = input_data.read_data_sets(file_path, one_hot=True)
        return mnist


    def get_batch(self, mnist, size):
        batch_x,batch_y = mnist.train.next_batch(size)
        return([batch_x,batch_y])


    def create_placeholder(self):
        placeholder_x = tf.placeholder(tf.float32, [None, 784])
        placeholder_y_ = tf.placeholder(tf.float32, [None, 10])
        return([placeholder_x, placeholder_y_])

    def construct_full_net(self, placeholder_x):
        variable_weight = tf.Variable(tf.zeros([784, 10]))
        variable_bias = tf.Variable(tf.zeros([10]))
        predict_y = tf.nn.softmax(tf.matmul(placeholder_x, variable_weight) + variable_bias)
        return(predict_y)

    def get_session(self):
        sess = tf.Session()
        return(sess)

    def initial_variable(self, sess):
        init = tf.global_variables_initializer()
        sess.run(init)


    def train_model(self, sess, mnist, placeholder_y_, predict_y, train_round, batch_size):
        cross_entropy = -tf.reduce_sum(placeholder_y_ * tf.log(predict_y))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        for idx in range(train_round):
            if (idx % 10 == 0):
                print('idx: ',idx)
            batch_x,batch_y = self.get_batch(mnist, batch_size)
            sess.run(train_step, feed_dict={placeholder_x:batch_x, placeholder_y_:batch_y})

    def evaluate_model(self, sess, mnist, placeholder_x, predict_y, placeholder_y_):
        correct_predict = tf.equal(tf.argmax(predict_y,1), tf.argmax(placeholder_y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
        print('mnist.test accuracy: ', sess.run(accuracy, feed_dict={placeholder_x:mnist.test.images, placeholder_y_:mnist.test.labels}))



if __name__ == '__main__':
    print('start')
    full_connect_model = FullConnectMnistModel('full connect')
    mnist = full_connect_model.load_mnist_data('../mnist/mnist_gz')
    print('train.shape:\n', mnist.train.images.shape, mnist.train.labels.shape)
    placeholder_x, placeholder_y_ = full_connect_model.create_placeholder()
    print('placeholder: ', placeholder_x, placeholder_y_)
    predict_y = full_connect_model.construct_full_net(placeholder_x)
    print('connect predcit_y:  ', predict_y)
    sess = full_connect_model.get_session()
    full_connect_model.initial_variable(sess)
    full_connect_model.train_model(sess, mnist, placeholder_y_, predict_y, 1000, 100)
    full_connect_model.evaluate_model(sess, mnist, placeholder_x, predict_y, placeholder_y_)
    
