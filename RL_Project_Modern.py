# -*- coding: utf-8 -*-
"""
Modern IoT-Fog Computing Task Offloading with Deep Reinforcement Learning
Updated for TensorFlow 2.12+ with GPU support

This implementation uses Deep Q-Network (DQN) with LSTM for task offloading
decisions in IoT-Fog computing environments.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import random
import math
import queue
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure GPU
def configure_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU detected and configured: {len(gpus)} device(s)")
            return True
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
            return False
    else:
        logger.info("No GPU detected, using CPU")
        return False

class DeepQNetwork:
    def __init__(self,
                 n_actions,                  # the number of actions
                 n_features,
                 n_lstm_features,
                 n_time,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.99,
                 replace_target_iter=200,    # each 200 steps, update target net
                 memory_size=500,            # maximum of memory
                 batch_size=32,
                 e_greedy_increment=0.00025,
                 n_lstm_step=10,
                 dueling=True,
                 double_q=True,
                 N_L1=20,
                 N_lstm=20):

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0
        self.N_L1 = N_L1

        # LSTM parameters
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_features

        # Initialize memory
        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1
                                + self.n_features + self.n_lstm_state + self.n_lstm_state))

        # Build networks
        self._build_net()

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)

        # Storage for training history
        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()

        # LSTM history
        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for ii in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_q_value = list()

    def _build_net(self):
        """Build the neural networks using TensorFlow 2.x Keras API"""
        
        def build_network(name):
            """Build a single network (eval or target)"""
            # Input layers
            state_input = tf.keras.layers.Input(shape=(self.n_features,), name=f'{name}_state_input')
            lstm_input = tf.keras.layers.Input(shape=(self.n_lstm_step, self.n_lstm_state), name=f'{name}_lstm_input')
            
            # LSTM layer
            lstm_layer = tf.keras.layers.LSTM(self.N_lstm, name=f'{name}_lstm')(lstm_input)
            
            # Concatenate state and LSTM features
            concat_layer = tf.keras.layers.Concatenate(name=f'{name}_concat')([state_input, lstm_layer])
            
            # Dense layers
            dense1 = tf.keras.layers.Dense(self.N_L1, activation='relu', name=f'{name}_dense1')(concat_layer)
            dense2 = tf.keras.layers.Dense(self.N_L1, activation='relu', name=f'{name}_dense2')(dense1)
            
            if self.dueling:
                # Dueling DQN architecture
                value_stream = tf.keras.layers.Dense(1, name=f'{name}_value')(dense2)
                advantage_stream = tf.keras.layers.Dense(self.n_actions, name=f'{name}_advantage')(dense2)
                
                # Combine value and advantage
                mean_advantage = tf.keras.layers.Lambda(
                    lambda x: tf.reduce_mean(x, axis=1, keepdims=True),
                    name=f'{name}_mean_advantage'
                )(advantage_stream)
                
                q_values = tf.keras.layers.Add(name=f'{name}_q_values')([
                    value_stream,
                    tf.keras.layers.Subtract(name=f'{name}_subtract')([advantage_stream, mean_advantage])
                ])
            else:
                # Standard DQN
                q_values = tf.keras.layers.Dense(self.n_actions, name=f'{name}_q_values')(dense2)
            
            model = tf.keras.Model(inputs=[state_input, lstm_input], outputs=q_values, name=name)
            return model

        # Build evaluation and target networks
        self.q_eval_model = build_network('eval_net')
        self.q_target_model = build_network('target_net')
        
        # Copy weights from eval to target network
        self.q_target_model.set_weights(self.q_eval_model.get_weights())

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        """Store transition in replay memory"""
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):
        """Update LSTM history"""
        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        """Choose action using epsilon-greedy policy"""
        observation = observation[np.newaxis, :]
        
        if np.random.uniform() < self.epsilon:
            # Exploit: choose best action
            lstm_observation = np.array(self.lstm_history)
            lstm_observation = lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state)
            
            actions_value = self.q_eval_model.predict([observation, lstm_observation], verbose=0)
            self.store_q_value.append({'observation': observation, 'q_value': actions_value})
            action = np.argmax(actions_value)
        else:
            # Explore: choose random action
            action = np.random.randint(0, self.n_actions)
        
        return action

    @tf.function
    def train_step(self, states, lstm_states, actions, targets):
        """Single training step using tf.function for performance"""
        with tf.GradientTape() as tape:
            q_pred = self.q_eval_model([states, lstm_states])
            q_pred_selected = tf.reduce_sum(q_pred * tf.one_hot(actions, self.n_actions), axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_pred_selected))
        
        gradients = tape.gradient(loss, self.q_eval_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_eval_model.trainable_variables))
        return loss

    def learn(self):
        """Train the network"""
        # Check if replace target network parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target_model.set_weights(self.q_eval_model.get_weights())
            logger.info('Target network parameters updated')

        # Sample from memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)

        # Prepare batch data
        batch_memory = self.memory[sample_index, :self.n_features + 1 + 1 + self.n_features]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])
        
        for ii in range(len(sample_index)):
            for jj in range(self.n_lstm_step):
                lstm_batch_memory[ii, jj, :] = self.memory[sample_index[ii] + jj,
                                               self.n_features + 1 + 1 + self.n_features:]

        # Extract states, actions, rewards, next_states
        states = batch_memory[:, :self.n_features]
        actions = batch_memory[:, self.n_features].astype(int)
        rewards = batch_memory[:, self.n_features + 1]
        next_states = batch_memory[:, -self.n_features:]
        
        current_lstm = lstm_batch_memory[:, :, :self.n_lstm_state]
        next_lstm = lstm_batch_memory[:, :, self.n_lstm_state:]

        # Get Q-values
        q_next = self.q_target_model.predict([next_states, next_lstm], verbose=0)
        
        if self.double_q:
            q_eval4next = self.q_eval_model.predict([next_states, next_lstm], verbose=0)
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[np.arange(self.batch_size), max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        # Compute targets
        targets = rewards + self.gamma * selected_q_next

        # Train
        loss = self.train_step(
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(current_lstm, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(targets, dtype=tf.float32)
        )

        # Update epsilon
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        self.learn_step_counter += 1

    def do_store_reward(self, episode, time, reward):
        """Store reward for analysis"""
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self, episode, time, action):
        """Store action for analysis"""
        while episode >= len(self.action_store):
            self.action_store.append(-np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        """Store delay for analysis"""
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay


class Offload:
    """IoT-Fog Computing Environment"""
    
    def __init__(self, num_iot, num_fog, num_time, max_delay):
        # Environment parameters
        self.n_iot = num_iot
        self.n_fog = num_fog
        self.n_time = num_time
        self.duration = 0.1

        # Performance counters
        self.drop_trans_count = 0
        self.drop_fog_count = 0
        self.drop_iot_count = 0

        # Capacity settings
        self.comp_cap_iot = 2.5 * np.ones(self.n_iot) * self.duration
        self.comp_cap_fog = 41.8 * np.ones([self.n_fog]) * self.duration
        self.tran_cap_iot = 14 * np.ones([self.n_iot, self.n_fog]) * self.duration
        self.comp_density = 0.297 * np.ones([self.n_iot])
        self.max_delay = max_delay

        # Task arrival settings - more challenging
        self.task_arrive_prob = 0.4  # Higher task arrival rate
        self.max_bit_arrive = 8      # Larger tasks
        self.min_bit_arrive = 3      # Higher minimum load
        self.bitArrive_set = np.arange(self.min_bit_arrive, self.max_bit_arrive, 0.1)
        self.bitArrive = np.zeros([self.n_time, self.n_iot])

        # Action and state space
        self.n_actions = 1 + num_fog
        self.n_features = 1 + 1 + 1 + num_fog
        self.n_lstm_state = self.n_fog

        # Initialize environment state
        self._initialize_environment()

    def _initialize_environment(self):
        """Initialize queues and state variables"""
        self.time_count = int(0)

        # Initialize queues
        self.Queue_iot_comp = [queue.Queue() for _ in range(self.n_iot)]
        self.Queue_iot_tran = [queue.Queue() for _ in range(self.n_iot)]
        self.Queue_fog_comp = [[queue.Queue() for _ in range(self.n_fog)] for _ in range(self.n_iot)]

        # Initialize state variables
        self.t_iot_comp = -np.ones([self.n_iot])
        self.t_iot_tran = -np.ones([self.n_iot])
        self.b_fog_comp = np.zeros([self.n_iot, self.n_fog])

        # Task indicators
        self.task_on_process_local = [{'size': np.nan, 'time': np.nan, 'remain': np.nan} 
                                      for _ in range(self.n_iot)]
        self.task_on_transmit_local = [{'size': np.nan, 'time': np.nan, 'fog': np.nan, 'remain': np.nan} 
                                       for _ in range(self.n_iot)]
        self.task_on_process_fog = [[{'size': np.nan, 'time': np.nan, 'remain': np.nan} 
                                     for _ in range(self.n_fog)] for _ in range(self.n_iot)]
        
        self.fog_iot_m = np.zeros(self.n_fog)
        self.fog_iot_m_observe = np.zeros(self.n_fog)

        # Delay tracking
        self.process_delay = np.zeros([self.n_time, self.n_iot])
        self.process_delay_unfinish_ind = np.zeros([self.n_time, self.n_iot])
        self.process_delay_trans = np.zeros([self.n_time, self.n_iot])
        self.fog_drop = np.zeros([self.n_iot, self.n_fog])

    def reset(self, bitArrive):
        """Reset environment for new episode"""
        # Reset counters
        self.drop_trans_count = 0
        self.drop_fog_count = 0
        self.drop_iot_count = 0

        # Set bit arrival pattern
        self.bitArrive = bitArrive

        # Re-initialize environment
        self._initialize_environment()

        # Initial observations
        observation_all = np.zeros([self.n_iot, self.n_features])
        for iot_index in range(self.n_iot):
            if self.bitArrive[self.time_count, iot_index] != 0:
                observation_all[iot_index, :] = np.hstack([
                    self.bitArrive[self.time_count, iot_index],
                    self.t_iot_comp[iot_index],
                    self.t_iot_tran[iot_index],
                    np.squeeze(self.b_fog_comp[iot_index, :])
                ])

        lstm_state_all = np.zeros([self.n_iot, self.n_lstm_state])
        return observation_all, lstm_state_all

    def step(self, action):
        """Execute one time step in the environment"""
        # Extract actions for each IoT device
        iot_action_local = np.zeros([self.n_iot], np.int32)
        iot_action_fog = np.zeros([self.n_iot], np.int32)
        
        for iot_index in range(self.n_iot):
            iot_action = action[iot_index]
            iot_action_fog[iot_index] = int(iot_action - 1)
            if iot_action == 0:
                iot_action_local[iot_index] = 1

        # Process computation queues (local processing)
        self._process_local_computation(iot_action_local)
        
        # Process fog computation queues
        self._process_fog_computation()
        
        # Process transmission queues
        self._process_transmission(iot_action_local, iot_action_fog)
        
        # Update fog load information
        self._update_fog_load()
        
        # Update time
        self.time_count += 1
        done = (self.time_count >= self.n_time - self.max_delay)

        # Generate next observations
        observation_all_ = np.zeros([self.n_iot, self.n_features])
        lstm_state_all_ = np.zeros([self.n_iot, self.n_lstm_state])
        
        if not done:
            for iot_index in range(self.n_iot):
                if self.bitArrive[self.time_count, iot_index] != 0:
                    observation_all_[iot_index, :] = np.hstack([
                        self.bitArrive[self.time_count, iot_index],
                        self.t_iot_comp[iot_index] - self.time_count + 1,
                        self.t_iot_tran[iot_index] - self.time_count + 1,
                        self.b_fog_comp[iot_index, :]
                    ])
                lstm_state_all_[iot_index, :] = np.hstack(self.fog_iot_m_observe)

        return observation_all_, lstm_state_all_, done

    def _process_local_computation(self, iot_action_local):
        """Process local computation tasks"""
        for iot_index in range(self.n_iot):
            iot_bitarrive = self.bitArrive[self.time_count, iot_index]
            iot_comp_cap = self.comp_cap_iot[iot_index]
            iot_comp_density = self.comp_density[iot_index]

            # Add new tasks to local computation queue
            if iot_action_local[iot_index] == 1 and iot_bitarrive > 0:
                task = {'size': iot_bitarrive, 'time': self.time_count}
                self.Queue_iot_comp[iot_index].put(task)

            # Process current task
            if math.isnan(self.task_on_process_local[iot_index]['remain']) and not self.Queue_iot_comp[iot_index].empty():
                task = self.Queue_iot_comp[iot_index].get()
                if task['size'] != 0 and self.time_count - task['time'] + 1 <= self.max_delay:
                    self.task_on_process_local[iot_index] = {
                        'size': task['size'],
                        'time': task['time'],
                        'remain': task['size']
                    }

            # Execute processing
            if self.task_on_process_local[iot_index]['remain'] > 0:
                self.task_on_process_local[iot_index]['remain'] -= iot_comp_cap / iot_comp_density
                
                if self.task_on_process_local[iot_index]['remain'] <= 0:
                    # Task completed
                    task_time = self.task_on_process_local[iot_index]['time']
                    self.process_delay[task_time, iot_index] = self.time_count - task_time + 1
                    self.task_on_process_local[iot_index]['remain'] = np.nan

            # Update queue information
            if iot_bitarrive != 0:
                self.t_iot_comp[iot_index] = min(
                    max(self.t_iot_comp[iot_index] + 1, self.time_count) + 
                    math.ceil(iot_bitarrive * iot_action_local[iot_index] / (iot_comp_cap / iot_comp_density)) - 1,
                    self.time_count + self.max_delay - 1
                )

    def _process_fog_computation(self):
        """Process fog computation tasks"""
        for iot_index in range(self.n_iot):
            iot_comp_density = self.comp_density[iot_index]
            
            for fog_index in range(self.n_fog):
                # Start new task if processor is free
                if (math.isnan(self.task_on_process_fog[iot_index][fog_index]['remain']) and 
                    not self.Queue_fog_comp[iot_index][fog_index].empty()):
                    
                    task = self.Queue_fog_comp[iot_index][fog_index].get()
                    if self.time_count - task['time'] + 1 <= self.max_delay:
                        self.task_on_process_fog[iot_index][fog_index] = {
                            'size': task['size'],
                            'time': task['time'],
                            'remain': task['size']
                        }

                # Process current task
                if self.task_on_process_fog[iot_index][fog_index]['remain'] > 0 and self.fog_iot_m[fog_index] > 0:
                    processing_power = self.comp_cap_fog[fog_index] / iot_comp_density / self.fog_iot_m[fog_index]
                    self.task_on_process_fog[iot_index][fog_index]['remain'] -= processing_power
                    
                    if self.task_on_process_fog[iot_index][fog_index]['remain'] <= 0:
                        # Task completed
                        task_time = self.task_on_process_fog[iot_index][fog_index]['time']
                        self.process_delay[task_time, iot_index] = self.time_count - task_time + 1
                        self.task_on_process_fog[iot_index][fog_index]['remain'] = np.nan

                # Update fog queue information
                if self.fog_iot_m[fog_index] != 0:
                    reduction = (self.comp_cap_fog[fog_index] / iot_comp_density / self.fog_iot_m[fog_index] +
                                 self.fog_drop[iot_index, fog_index])
                    self.b_fog_comp[iot_index, fog_index] = max(
                        self.b_fog_comp[iot_index, fog_index] - reduction, 0
                    )

    def _process_transmission(self, iot_action_local, iot_action_fog):
        """Process transmission tasks"""
        for iot_index in range(self.n_iot):
            iot_tran_cap = self.tran_cap_iot[iot_index, :]
            iot_bitarrive = self.bitArrive[self.time_count, iot_index]

            # Add new transmission tasks
            if iot_action_local[iot_index] == 0 and iot_bitarrive > 0:
                task = {
                    'size': iot_bitarrive,
                    'time': self.time_count,
                    'fog': iot_action_fog[iot_index]
                }
                self.Queue_iot_tran[iot_index].put(task)

            # Start new transmission if channel is free
            if (math.isnan(self.task_on_transmit_local[iot_index]['remain']) and
                not self.Queue_iot_tran[iot_index].empty()):
                
                task = self.Queue_iot_tran[iot_index].get()
                if task['size'] != 0 and self.time_count - task['time'] + 1 <= self.max_delay:
                    self.task_on_transmit_local[iot_index] = {
                        'size': task['size'],
                        'time': task['time'],
                        'fog': int(task['fog']),
                        'remain': task['size']
                    }

            # Process transmission
            if self.task_on_transmit_local[iot_index]['remain'] > 0:
                fog_idx = self.task_on_transmit_local[iot_index]['fog']
                self.task_on_transmit_local[iot_index]['remain'] -= iot_tran_cap[fog_idx]

                if self.task_on_transmit_local[iot_index]['remain'] <= 0:
                    # Transmission completed, add to fog queue
                    fog_task = {
                        'size': self.task_on_transmit_local[iot_index]['size'],
                        'time': self.task_on_transmit_local[iot_index]['time']
                    }
                    self.Queue_fog_comp[iot_index][fog_idx].put(fog_task)
                    
                    # Update fog queue size
                    self.b_fog_comp[iot_index, fog_idx] += fog_task['size']
                    
                    # Record transmission delay
                    task_time = self.task_on_transmit_local[iot_index]['time']
                    self.process_delay_trans[task_time, iot_index] = self.time_count - task_time + 1
                    
                    self.task_on_transmit_local[iot_index]['remain'] = np.nan

            # Update transmission queue information
            if iot_bitarrive != 0:
                self.t_iot_tran[iot_index] = min(
                    max(self.t_iot_tran[iot_index] + 1, self.time_count) +
                    math.ceil(iot_bitarrive * (1 - iot_action_local[iot_index]) / 
                             iot_tran_cap[iot_action_fog[iot_index]]) - 1,
                    self.time_count + self.max_delay - 1
                )

    def _update_fog_load(self):
        """Update fog node load information"""
        # Count active IoT devices per fog node
        self.fog_iot_m = np.zeros(self.n_fog)
        for fog_index in range(self.n_fog):
            for iot_index in range(self.n_iot):
                if not math.isnan(self.task_on_process_fog[iot_index][fog_index]['remain']):
                    self.fog_iot_m[fog_index] += 1
        
        # Set minimum load to avoid division by zero
        self.fog_iot_m = np.maximum(self.fog_iot_m, 1)
        self.fog_iot_m_observe = self.fog_iot_m.copy()


def reward_fun(delay, max_delay, unfinish_indi):
    """Reward function for RL training"""
    penalty = -max_delay * 2
    if unfinish_indi:
        return penalty
    else:
        return -delay


def train(iot_RL_list, NUM_EPISODE, env):
    """Training loop for the RL agents"""
    RL_step = 0
    Total_tasks = []
    Total_dropped = []

    for episode in range(NUM_EPISODE):
        # Show progress every episode
        logger.info(f"Episode {episode+1}/{NUM_EPISODE}, Epsilon: {iot_RL_list[0].epsilon:.4f}")
        
        # Generate bit arrival pattern
        bitarrive = np.random.uniform(env.min_bit_arrive, env.max_bit_arrive, 
                                      size=[env.n_time, env.n_iot])
        task_prob = env.task_arrive_prob
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_iot]) < task_prob)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_iot])

        # Count tasks in episode
        tasks_ep = np.count_nonzero(bitarrive)
        Total_tasks.append(tasks_ep)

        # Initialize episode
        observation_all, lstm_state_all = env.reset(bitarrive)

        # Episode loop
        while True:
            # Choose actions for all IoT devices
            action_all = np.zeros([env.n_iot], dtype=int)
            for iot_index in range(env.n_iot):
                observation = observation_all[iot_index, :]
                lstm_state = lstm_state_all[iot_index, :]
                
                # Update LSTM history and choose action
                iot_RL_list[iot_index].update_lstm(lstm_state)
                action_all[iot_index] = iot_RL_list[iot_index].choose_action(observation)

            # Execute actions in environment
            observation_all_, lstm_state_all_, done = env.step(action_all)

            # Store transitions and compute rewards
            for iot_index in range(env.n_iot):
                if env.bitArrive[env.time_count - 1, iot_index] != 0:
                    # Compute reward based on delay
                    delay = env.process_delay[env.time_count - 1, iot_index]
                    unfinish_ind = env.process_delay_unfinish_ind[env.time_count - 1, iot_index]
                    reward = reward_fun(delay, env.max_delay, unfinish_ind)
                    
                    # Store transition
                    iot_RL_list[iot_index].store_transition(
                        observation_all[iot_index, :],
                        lstm_state_all[iot_index, :],
                        action_all[iot_index],
                        reward,
                        observation_all_[iot_index, :],
                        lstm_state_all_[iot_index, :]
                    )
                    
                    # Store for analysis
                    iot_RL_list[iot_index].do_store_reward(episode, env.time_count - 1, reward)
                    iot_RL_list[iot_index].do_store_action(episode, env.time_count - 1, action_all[iot_index])
                    iot_RL_list[iot_index].do_store_delay(episode, env.time_count - 1, delay)

            RL_step += 1

            # Update observations
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # Learning
            if (RL_step > 200) and (RL_step % 10 == 0):
                for iot in range(env.n_iot):
                    iot_RL_list[iot].learn()

            if done:
                break

        # Count dropped tasks
        unfinish_indi = env.process_delay_unfinish_ind
        tasks_drop_ep = np.count_nonzero(unfinish_indi)
        Total_dropped.append(tasks_drop_ep)
        
        # Calculate episode rewards for all agents
        episode_reward = 0
        for iot in range(env.n_iot):
            if episode < len(iot_RL_list[iot].reward_store):
                episode_reward += np.sum(iot_RL_list[iot].reward_store[episode])
        
        # Show results for every episode
        drop_rate = tasks_drop_ep / max(tasks_ep, 1) * 100
        logger.info(f'Tasks: {tasks_ep}, Dropped: {tasks_drop_ep}, Drop Rate: {drop_rate:.1f}%, Total Reward: {episode_reward:.1f}')

    return Total_tasks, Total_dropped


def plot_results(iot_RL_list, Total_tasks, Total_dropped, NUM_EPISODE, NUM_IOT):
    """Plot training results"""
    # Calculate episode rewards
    episode_rewards = []
    for episode in range(NUM_EPISODE):
        episode_rewards.append(np.zeros([NUM_IOT]))
        for iot in range(NUM_IOT):
            if episode < len(iot_RL_list[iot].reward_store):
                rew = np.sum(iot_RL_list[iot].reward_store[episode])
                episode_rewards[episode][iot] = rew

    avg_rewards = [np.mean(episode_rewards[episode]) for episode in range(NUM_EPISODE)]

    # Calculate episode delays
    episode_delays = []
    for episode in range(NUM_EPISODE):
        episode_delays.append(np.zeros([NUM_IOT]))
        for iot in range(NUM_IOT):
            if episode < len(iot_RL_list[iot].delay_store):
                delay = np.sum(iot_RL_list[iot].delay_store[episode])
                episode_delays[episode][iot] = delay

    avg_delays = [np.mean(episode_delays[episode]) for episode in range(NUM_EPISODE)]

    # Calculate drop ratios
    Total_dropped_float = [float(i) for i in Total_dropped]
    Total_tasks_float = [float(i) for i in Total_tasks]
    drop_ratio = [Total_dropped_float[i] / max(Total_tasks_float[i], 1) for i in range(len(Total_dropped_float))]

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Average rewards plot
    ax1.plot(range(len(avg_rewards)), avg_rewards)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Rewards')
    ax1.set_title('Average Rewards vs Episodes')
    ax1.grid(True)

    # Average delays plot
    ax2.plot(range(len(avg_delays)), avg_delays)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Average Delay')
    ax2.set_title('Average Delay vs Episodes')
    ax2.grid(True)

    # Task drop ratio plot
    ax3.plot(range(len(drop_ratio)), drop_ratio)
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Ratio of Dropped Tasks')
    ax3.set_title('Task Drop Ratio vs Episodes')
    ax3.grid(True)

    # Tasks per episode
    ax4.plot(range(len(Total_tasks)), Total_tasks, label='Total Tasks', alpha=0.7)
    ax4.plot(range(len(Total_dropped)), Total_dropped, label='Dropped Tasks', alpha=0.7)
    ax4.set_xlabel('Episodes')
    ax4.set_ylabel('Number of Tasks')
    ax4.set_title('Tasks per Episode')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('RL_Project/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Configure GPU
    gpu_available = configure_gpu()
    
    # Environment parameters
    NUM_IOT = 50
    NUM_FOG = 5
    NUM_EPISODE = 100  # Reduced for testing
    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    logger.info(f"Starting training with {NUM_IOT} IoT devices, {NUM_FOG} fog nodes")
    logger.info(f"GPU available: {gpu_available}")

    # Create environment
    env = Offload(NUM_IOT, NUM_FOG, NUM_TIME, MAX_DELAY)

    # Create RL agents for each IoT device
    iot_RL_list = []
    for iot in range(NUM_IOT):
        agent = DeepQNetwork(
            env.n_actions, 
            env.n_features, 
            env.n_lstm_state, 
            env.n_time,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.99,
            replace_target_iter=200,
            memory_size=500
        )
        iot_RL_list.append(agent)

    # Train the system
    logger.info("Starting training...")
    Total_tasks, Total_dropped = train(iot_RL_list, NUM_EPISODE, env)
    logger.info("Training completed!")

    # Plot and save results
    plot_results(iot_RL_list, Total_tasks, Total_dropped, NUM_EPISODE, NUM_IOT)
    
    # Save models
    for i, agent in enumerate(iot_RL_list):
        agent.q_eval_model.save(f'RL_Project/models/iot_agent_{i}_eval.h5')
        agent.q_target_model.save(f'RL_Project/models/iot_agent_{i}_target.h5')
    
    logger.info("Models saved successfully!")
