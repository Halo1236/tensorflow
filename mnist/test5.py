#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2018/9/12 9:47
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 模型的输入和输出(占位)
# 28x28像素
x = tf.placeholder("float", shape=[None, 784])
# 0-9十种
y_ = tf.placeholder("float", shape=[None, 10])

# 模型的权重和偏移量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 梯度下降算法  step0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(10000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# 测试
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
