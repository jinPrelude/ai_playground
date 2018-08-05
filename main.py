"""
Original of this code is from "https://github.com/pemami4911/deep-rl".
Brought for study, and re-create to tensorflow-only version.
"""
import tensorflow as tf
import numpy as np
import gym
from collections import deque
import argparse
import pprint as pp
import tflearn


from replay_buffer import ReplayBuffer


class ActorNetwork(object):


    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network를 생성합니다.
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        # It stores the parameters the network has.
        #
        self.network_params = tf.trainable_variables()

        # Target Actor network를 생성합니다.
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        # It stores the parameters the target network has.
        # We should slice the tf.trainable_variables() because unlike
        # network_params target_actor_network is has made above.
        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]


        # Op for periodically updating target network with online network
        # weights
        # update_target_network_params = tau*t theta[i] + (1-tau) * target_theta[i]
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau)) for i in range(len(self.target_network_params))]

        # critic network에게 제공받을 placeholder입니다. action의 gradient입니다.
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # 최적화 부분
        #optimization part
        self.optimize = \
            tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        #tf.trainable_variables()의 반환값이 무엇인지 먼저 알아야 한다.
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    # Define actor neural network
    def create_actor_network(self):

        inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        w1 = tf.Variable(tf.random_normal(shape=[self.s_dim, 10], mean=0., stddev=0.1), name='w1')
        l1 = tf.matmul(inputs, w1)
        l1 = tf.nn.relu(l1)

        w2 = tf.Variable(tf.random_normal(shape=[10, 10], mean=0., stddev=0.1), name='w2')
        l2 = tf.matmul(l1, w2)
        l2 = tf.nn.relu(l2)

        w3 = tf.Variable(tf.random_normal(shape=[10, 6], mean=0., stddev=0.1), name='w3')
        l3 = tf.matmul(l2, w3)
        l3 = tf.nn.relu(l3)

        w4 = tf.Variable(tf.random_normal(shape=[6, self.a_dim], mean=0., stddev=0.1), name='w4')
        l4 = tf.matmul(l3, w4)
        out = tf.nn.tanh(l4)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out


    # action의 gradient 와 inputs(state)를 입력으로 받아 self.optimize를 돌려서 학습합니다.
    # Train by running self.optimize which gets the gradient of the action and inputs(state) as a inputs
    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    # input을 받아 예측한 행동을 반환합니다.
    # Choose and return the action of the actor network
    # by getting input(state) as a input
    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    # target network의 행동 예측값을 반환합니다.
    # Choose and return the action of the target actor network
    # by getting input(state) as a input
    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    # target network를 self.update_target_network_params를 이용해 업데이트합니다.
    # Update the target network by using self.update_target_network_params
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    # 정채불명
    # Unconfirmed
    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # critic network를 생성합니다.
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target critic network를 생성합니다.
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        #target critic network에 y_i 값으로 제공될 placeholder입니다.
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # loss를 정의하고 최적화합니다.
        #self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # action에 대해서 신경망의 gradient를 구합니다.
        # 미니배치의 각 critic 출력의 (각 액션에 의해서 구해진)기울기를 합산한다.
        # 모든 출력은 자신이 나눠진 action을 제외한 모든 action에 대해 독립적이다.
        self.action_grads = tf.gradients(self.out, self.action)

    # Critic network를 정의합니다.
    # Define the critic network
    def create_critic_network(self):
        inputs = tf.placeholder(shape=[None, self.s_dim], dtype=tf.float32)
        action = tf.placeholder(shape=[None, self.a_dim], dtype=tf.float32)

        w1 = tf.Variable(tf.random_uniform(shape=[self.s_dim, 400], maxval=0.3, minval=-0.3), dtype=tf.float32)
        l1 = tf.matmul(inputs, w1)
        l1 = tf.nn.relu(l1)
        w2 = tf.Variable(tf.random_uniform(shape=[400, 300], maxval=0.3, minval=-0.3), dtype=tf.float32)

        # action에 가중치를 곱해서 critic network에 더해준다. 경험적으로 좋은 결과를 이끌어냈다고 함.

        w2_a = tf.Variable(tf.random_uniform(shape=[self.a_dim, 300],  maxval=0.3, minval=-0.3), dtype=tf.float32)
        l2 = tf.nn.relu(tf.matmul(l1, w2) + tf.matmul(action, w2_a))

        w3 = tf.Variable(tf.random_uniform(shape=[300, 1], maxval=0.03, minval=-0.03), dtype=tf.float32)
        out = tf.matmul(l2, w3)


        return inputs, action, out


    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py 에서 그대로 가지고 옴
# Brought this class at https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


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


# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())

    #generate tensorboard
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    saver = tf.train.Saver()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    reward_mean = deque(maxlen=10)

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # uo-process 노이즈 추가
            # add uo-process noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(a[0])

            # replay buffer에 추가
            # Add to replay buffer
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                                                                             i, (ep_ave_max_q / float(j))))
                reward_mean.append(ep_reward)

                break

        if i > 10  :
            if int(sum(reward_mean)/10) < -200 :
                saver.save(sess, './results/model_save/model.ckpt')



def main(args):
    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))



        train(sess, env, args, actor, critic, actor_noise)



if __name__ == '__main__':


    # print the parameters on the console
    # and also offer the parametes to the main function
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)