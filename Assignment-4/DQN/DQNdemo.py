import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import losses,optimizers,metrics
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from tqdm import tqdm

env = gym.make("MountainCar-v0")
env.reset()

# Parameter Settings
ACTION_SPACE_SIZE = env.action_space.n
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
DISCOUNT = 0.99
EPISODES =100
epsilon = 1 
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

ep_rewards = []

AGGREGATE_STATS_EVERY = 50
MIN_EPSILON = 0.001

recorder = {"epsode":[],"epsilon":[]}

for epsode in range(EPISODES):
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    recorder["epsode"].append(epsode)
    recorder["epsilon"].append(epsilon)

def create_model():
    model = models.Sequential()
    model.add(Dense(16 ,input_shape=(env.observation_space.shape)))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(ACTION_SPACE_SIZE))
    model.add(Activation('linear'))
    model.compile(loss = 'mse', optimizer=Adam(lr=0.001),metrics=['accuracy'])
    return model

class DQNAgent:
  def __init__(self):
    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    self.model_prediction = create_model()
    self.model_target = create_model()
    self.model_target.set_weights(self.model_prediction.get_weights())
    self.target_update_counter = 0
    
  def update_replay_memory(self, transition):
    self.replay_memory.append(transition)
  
  def get_qs(self, state):
    return self.model_prediction.predict(np.array(state).reshape(-1, *state.shape))[0]
  
  def train(self, terminal_state, step):
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
      return
    minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
    current_states = np.array([transition[0] for transition in minibatch])
    current_qs_list = self.model_prediction.predict(current_states)   
    next_states = np.array([transition[3] for transition in minibatch])
    target_qs_list = self.model_target.predict(next_states)
    
    X = []
    y = []
    for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):
        if not done:
            max_target_q = np.max(target_qs_list[index])
            new_q = reward + DISCOUNT * max_target_q
        else:
            new_q = reward
        current_qs = current_qs_list[index]
        current_qs[action] = new_q
        X.append(current_state)
        y.append(current_qs)    
    self.model_prediction.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False if terminal_state else None)
    if terminal_state:
        self.target_update_counter +=1
    if self.target_update_counter > UPDATE_TARGET_EVERY:
      self.model_target.set_weights(self.model_prediction.get_weights())
      self.target_update_counter = 0


agent = DQNAgent()
agent.model_prediction=keras.models.load_model('dqn_1.h5')

done = False
state = env.reset()
while not done:
    qs_list = agent.get_qs(state)
    action = np.argmax(qs_list)
    next_state, _, done, _ = env.step(action)
    state = next_state
    env.render()
env.close()