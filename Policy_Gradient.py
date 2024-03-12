"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
from collections import deque

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            n_time,
            n_lstm_features,
            learning_rate=0.01,
            reward_decay=0.95,
            memory_size=500,
            batch_size=32,
            n_lstm_step=10,
            N_lstm=20,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size 
        self.n_time = n_time
        # lstm
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step       # step_size in lstm
        self.n_lstm_state = n_lstm_features  # [fog1, fog2, ...., fogn, M_n(t)]

        self.ep_obs = np.zeros(memory_size, self.n_features)
        self.ep_obs_lstm = np.zeros(memory_size, self.n_lstm_state)
        self.ep_as = np.zeros(memory_size)
        self.ep_rs = np.zeros(memory_size)

        self._build_net()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for ii in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_prob_weights = list()

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_obs_lstm = tf.placeholder(tf.float32, [None, self.n_lstm_step, self.n_features], name="lstm observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        #lstm0
            lstm_dnn = tf.contrib.rnn.BasicLSTMCell(self.n_lstm)
            lstm_dnn.zero_state(1, tf.float32)
            lstm_output, lstm_state = tf.nn.dynamic_rnn(lstm_dnn, self.tf_obs_lstm, dtype=tf.float32)
            lstm_output_reduced = tf.reshape(lstm_output[-1, :], shape=[self.n_lstm])   
            concat_input = tf.concat([self.tf_obs, lstm_output_reduced])     
        # fc1
        layer = tf.layers.dense(
            inputs=concat_input,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        lstm_observation = np.array(self.lstm_history)

        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :], self.tf_obs_lstm: lstm_observation.reshape(self.n_lstm_step,
                                                                                           self.n_lstm_state),
                                                                   })
        
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        
        self.store_prob_weights.append({'observation': observation, 'q_value': prob_weights})

        return action
    
    def choose_action1(self, observation):
        action = 0
        return action

    def store_transition(self, s, lstm_s, a, r):

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        index = self.memory_counter % self.memory_size
        self.ep_obs[index, :] = s
        self.ep_obs_lstm[index, :] = lstm_s
        self.ep_as[index, :] = a
        self.ep_rs[index, :] = r

        self.memory_counter += 1

    def update_lstm(self, lstm_s):

        self.lstm_history.append(lstm_s)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # randomly pick [batch_size] memory from memory np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)

        batch_memory = self.ep_obs[sample_index, :]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state])
        for ii in range(len(sample_index)):
            for jj in range(self.n_lstm_step):
                lstm_batch_memory[ii,jj,:] = self.ep_obs_lstm[sample_index[ii]+jj, :]

        action_memory = self.ep_as[sample_index, :]
        reward_memory = discounted_ep_rs_norm[sample_index, :]

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: batch_memory,  # shape=[None, n_obs]
             self.tf_obs_lstm: lstm_batch_memory,
             self.tf_acts: action_memory,  # shape=[None, ]
             self.tf_vt: reward_memory,  # shape=[None, ]
        })

        # return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    
    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self,episode,time, action):
        while episode >= len(self.action_store):
            self.action_store.append(- np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy):
        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy



