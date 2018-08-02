import tensorflow as tf
import numpy as np
from collections import deque

class actor(object) :
    def __init__(self, sess, input_size, action_size):
        self.input = tf.placeholder(tf.float32, shape=[1,3], name="input")
        self.sess = sess

        with tf.variable_scope("actor") :
            w1 = tf.Variable(tf.random_normal(shape=[input_size, 30]), dtype=tf.float32)
            l1 = tf.matmul(self.input, w1)
            l1 = tf.nn.relu(l1)

            w2 = tf.Variable(tf.random_normal(shape=[30, 10]), dtype=tf.float32)
            l2 = tf.matmul(l1, w2)
            l2 = tf.nn.relu(l2)

            w3 = tf.Variable(tf.random_normal(shape=[10, 5]), dtype=tf.float32)
            l3 = tf.matmul(l2, w3)
            l3 = tf.nn.relu(l3)

            w4 = tf.Variable(tf.random_normal(shape=[5, 2]), dtype=tf.float32)
            self.output = tf.matmul(l3, w4)


        pass

    def learn(self):
        pass

    def choose_action(self, state):
        state = state[np.newaxis, :]
        action = self.sess.run(self.output, feed_dict={self.input : state})

        pass

class critic(object) :

    def __init__(self, input_size):
        pass

    def learn(self):
        pass