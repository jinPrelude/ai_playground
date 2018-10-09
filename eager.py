import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
from tensorflow.python.keras.datasets import mnist

tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

class Block(tf.keras.Model) :
    def __init__(self, filter, kernel, stride):
        super(Block, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filter, kernel_size=(kernel,kernel), strides=(stride, stride),
                                           padding='same', kernel_initializer='he_normal')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.nn.relu

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CNN(tf.keras.Model) :
    def __init__(self, num_class):
        super(CNN, self).__init__()
        self.block = Block(16, 3, 2)
        self.dense = tf.keras.layers.Dense(num_class)

    def call(self, inputs, training=None, mask=None):
        x = self.block(inputs)
        output = self.dense(x)
        return output

device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'

with tf.device(device) :
    