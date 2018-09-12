import tensorflow as tf
import numpy as np
from hover import hover_v1
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 100000
MAX_EP_STEPS = 100
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount`
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 100000

TRAIN_START = 1200
BATCH_SIZE = 1024


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, sess, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = sess

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)


    def choose_action(self, s):
        s = np.asarray(s)
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            l2 = tf.layers.dense(net, 40, activation=tf.nn.relu, name='l2', trainable=trainable)
            l3 = tf.layers.dense(l2, 20, activation=tf.nn.relu, name='l3', trainable=trainable)
            l4 = tf.layers.dense(l3, 10, activation=tf.nn.relu, name='l4', trainable=trainable)
            a = tf.layers.dense(l4, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)

            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            merge = tf.matmul(s, w1_s) + tf.matmul(a, w1_a)
            c_l1 = tf.layers.dense(merge, 30, activation=tf.nn.relu, name='c_l1', trainable=trainable)
            c_l2 = tf.layers.dense(c_l1, 40, activation=tf.nn.relu, name='c_l2', trainable=trainable)
            c_l3 = tf.layers.dense(c_l2, 20, activation=tf.nn.relu, name='c_l3', trainable=trainable)
            c_l4 = tf.layers.dense(c_l3, 10, activation=tf.nn.relu, name='c_l4', trainable=trainable)
            return tf.layers.dense(c_l4, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################
def main() :

    with tf.Session() as sess :
        env = hover_v1(sas = True, max_altitude=300, max_step=MAX_EP_STEPS)

        s_dim = env.observation_space + 1 # Add last thrust
        a_dim = env.action_space
        a_bound = env.action_max

        ddpg = DDPG(sess, a_dim, s_dim, a_bound)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1000)

        writer = tf.summary.FileWriter('./results/tensorboard/', sess.graph)
        writer.flush()

        var = 3  # control exploration
        for i in range(MAX_EPISODES):
            s = env.reset()
            ep_reward = 0
            last_thrust = env.initial_throttle

            # Add exploration noise
            s = np.append(s, last_thrust)
            for j in range(MAX_EP_STEPS+1):

                a = ddpg.choose_action(s)
                a = np.clip(np.random.normal(a, var), 0., 1.)   # add randomness to action selection for exploration
                s_, r, done = env.step(a)
                s_ = np.append(s_, a)

                ddpg.store_transition(s, a, r / 10, s_)


                s = s_
                ep_reward += r/100
                last_thrust = a

                if j == MAX_EP_STEPS:
                    print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                    print()
                    break
            if i % 100 == 0:
                saver.save(sess, './results/model_save/model_%d' % i)

            if ddpg.pointer > TRAIN_START:
                print('training')
                t = time.time()
                var *= .998  # decay the action randomness
                ddpg.learn()
                print('training time : ', time.time() - t)

        #print('Running time: ', time.time() - t1)

if __name__ == "__main__" :
    main()