import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, state_space, action_space):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_space, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(50, 100)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(100, 40)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(40, action_space)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self, args, state_space, action_space):

        self.args = args

        self.epsilon = 1.
        self.state_space = state_space
        self.action_space = action_space

        self.eval_net, self.target_net = \
            Net(state_space, action_space), Net(state_space, action_space)
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.args.memory_capacity, self.state_space * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):

        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]  # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.action_space)
            action = action     # if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.args.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.args.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.args.memory_capacity, self.args.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_space])
        b_a = torch.LongTensor(b_memory[:, self.state_space:self.state_space+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_space+1:self.state_space+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_space:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s)
        _, q_eval_argmax = torch.max(q_eval, 1)
        q_eval = q_eval.gather(1, b_a) # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # q_next = q_next.gather(1, b_a)              #double_dqn
        q_target = b_r + self.args.gamma * q_next.max(1)[0].view(self.args.batch_size, 1)   # shape (batch, 1)
        # q_target = b_r + self.gamma * q_next.view(self.batch_size, 1)  # double_dqn
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def cartpole_train_loop(self, env):
        print('\nCollecting experience...')
        for i_episode in range(self.args.num_episode):
            s = env.reset()
            ep_r = 0
            while True:
                env.render()
                a = self.choose_action(s)

                # take action
                s_, r, done, info = env.step(a)

                # modify the reward
                x, x_dot, theta, theta_dot = s_
                r1 = (env.env.x_threshold - abs(x)) / env.env.x_threshold - 0.8
                r2 = (env.env.theta_threshold_radians - abs(theta)) / env.env.theta_threshold_radians - 0.5
                r = r1 + r2

                self.store_transition(s, a, r, s_)

                ep_r += r
                if self.memory_counter > self.args.memory_capacity:
                    self.learn()
                    if done:
                        if i_episode % self.args.ep_print_iter == 0:
                            print('Ep: ', i_episode,
                                  '| Ep_r: ', round(ep_r, 2))
                            if i_episode % self.args.model_save_iter == 0:
                                torch.save(self.eval_net.state_dict(), './save_model/model_%d' % (i_episode))

                if done:
                    break
                s = s_
