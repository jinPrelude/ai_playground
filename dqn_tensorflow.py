import tensorflow as tf
import numpy as np
import gym
import argparse
import random
from collections import deque

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class agent() :
    def __init__(self, sess, input_dim, action_dim, learning_rate):
        self.sess = sess
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        with tf.variable_scope('network'):
            self.input, self.output = self.create_network()

        self.next_q = tf.placeholder(tf.float32, shape=[None, 1])
        self.reward = tf.placeholder(tf.float32, shape=[None, 1])

        self.loss = tf.reduce_mean(tf.square(self.reward + self.next_q - self.output))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def create_network(self):
        input = tf.placeholder(tf.float32, [None, self.input_dim])


        w1 = tf.get_variable('w1', shape=[self.input_dim, 200], dtype=tf.float32,
                             initializer=tf.random_normal_initializer())
        l1 = tf.matmul(input, w1)
        l1 = tf.nn.relu(l1)

        w2 = tf.get_variable('w2', shape=[200, 100], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        l2 = tf.matmul(l1, w2)
        l2 = tf.nn.relu(l2)

        w3 = tf.get_variable('w3', shape=[100, self.action_dim], dtype=tf.float32,
                           initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        l3 = tf.matmul(l2, w3)
        output = tf.nn.softmax(l3)


        return input, output

    def predict(self, input):

        output = self.sess.run(self.output, feed_dict={self.input:input})
        return output

    def train(self, reward, next_q, input):
        reward = np.reshape(reward, (-1, 1))
        next_q = np.reshape(next_q, (-1, 1))
        self.sess.run(self.optimizer, feed_dict={self.reward : reward,
                                                 self.next_q : next_q,
                                                 self.input : input})
# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def train(sess, env, args, actor) :

    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())

    # generate tensorboard
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    # actor.update_target_network()
    # critic.update_target_network()

    saver = tf.train.Saver()

    replay_buffer = ReplayBuffer(int(args['buffer_size']), 1)

    reward_mean = deque(maxlen=10)

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):
            env.render()
            # uo-process 노이즈 추가
            # add uo-process noise
            noise = np.random.normal(0.0, 1.0, 2)

            s = np.reshape(s, newshape=(1, -1))
            a = actor.predict(s) + noise
            a_final = np.argmax(a[0])
            s2, r, terminal, info = env.step(a_final)

            # replay buffer에 추가
            # Add to replay buffer
            replay_buffer.add(np.reshape(s, (actor.input_dim,)), np.reshape(a, (actor.action_dim,)), r,
                              terminal, np.reshape(s2, (actor.input_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                next_q = actor.predict(s_batch)
                next_q = np.argmax(next_q, axis=1)
                actor.train(r_batch, next_q, s_batch)
            s = s2

def main(args) :
    with tf.Session() as sess :

        env = gym.make(args['env'])

        input_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        actor = agent(sess, input_dim, action_dim, args['learning_rate'])

        train(sess, env, args, actor)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.003)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--buffer_size', default=10000)
    parser.add_argument('--minibatch_size', default=128)

    parser.add_argument('--env', default='CartPole-v0')
    parser.add_argument('--max_episodes', default=100)
    parser.add_argument('--max_episode_len', default=500)
    parser.add_argument('--summary_dir', default='./results/dqn')


    args = vars(parser.parse_args())

    main(args)