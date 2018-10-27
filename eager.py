import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

img_size = 28
batch_size = 128
epochs = 8

class generator():
    def __init__(self, sess, lr):
        self.sess = sess
        self.lr = lr

    def create_generator(self):
        inputs = tf.placeholder(name='generator_input', shape=[None, 100, 1], dtype=tf.float32)
        w1 = tf.layers.dense(inputs, [None, 4*4*1024], activation=tf.nn.relu)
        w1_rs = tf.reshape(w1, [None, 4, 4, 1024])
        w2 = tf.nn.conv2d_transpose(w1_rs, output_shape=[None, 9, 9, 512], strides=[2, 2])


        """
        conv1 = tf.layers.conv2d(inputs=inputs, filters=16, kernel_size=[3, 3], padding='same',
                              activation=tf.nn.relu, kernel_initializer='he_normal')

        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], padding='same',
                                 activation=tf.nn.relu, kernel_initializer='he_normal')

        conv_one = tf.layers.conv2d(inputs=conv1, filters=1, kernel_size=[1, 1], strides=[1,1],
                                    activation=tf.nn.relu, kernel_initializer='he_normal')
        """

