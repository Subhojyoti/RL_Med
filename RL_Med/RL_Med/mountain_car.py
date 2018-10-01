'''
Created on Sep 24, 2018

@author: subhojyotimukherjee
'''

import numpy
import gym
import random

      
class Main(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    def init_policy(self, env):
        
        state_range = env.max_position - env.min_position
        tiling = (state_range*1.0)*200.0
        self.Q = [[0.0, 0.0] for i in range(0,tiling)]   
    
    def update_greedy_policy(self, row1, col1, row2, col2, ini_action, final_action, t, reward):
        
        discount_factor = 0.9
        alpha = 0.01
        #self.Q[row1][col1][ini_action] += pow(discount_factor,t)*reward
        
        new_action1 = [i[0] for i in sorted(enumerate(self.Q[row2][col2]), reverse = True, key=lambda x:x[1])]
        num = random.uniform(0,1)
        self.epsilon = 0.01
        if num < self.epsilon:
            while True:
                action1 = random.randint(0,self.actions-1)
                if action1 != new_action1[0]:
                    break
            new_action = action1
        
        else:
            new_action = new_action1[0]

env = gym.make('MountainCar-v0')
print(env.action_space)
print(env.observation_space)


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(i_episode, t, observation)
        
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(reward, action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break