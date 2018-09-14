#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
created by Halo 2018/9/11 15:54
"""

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_random_normal = tf.random_normal(shape=[1000], mean=(5 * k), stddev=1)
mean_moving_normal = tf.truncated_normal(shape=[1000], mean=(5 * k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/random_output", mean_random_normal)
tf.summary.histogram("normal/truncated_output", mean_moving_normal)

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("/tmp/histogram_example2")

summaries = tf.summary.merge_all()

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
    k_val = step / float(N)
    summ = sess.run(summaries, feed_dict={k: k_val})
    writer.add_summary(summ, global_step=step)
