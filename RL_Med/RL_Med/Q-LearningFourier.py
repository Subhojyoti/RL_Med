'''
Created on Oct 1, 2018

@author: subhojyotimukherjee
'''






import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
import math
import random
from copy import deepcopy

from RL_Med.fourier_basis import FourierSampler

if "../" not in sys.path:
  sys.path.append("../") 

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')





env = gym.envs.make("MountainCar-v0")



        

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurized represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("f1", FourierSampler(constant=5.0, n_components=100)),
        ("f2", FourierSampler(constant=2.0, n_components=100)),
        ("f3", FourierSampler(constant=1.0, n_components=100)),
        ("f4", FourierSampler(constant=3.0, n_components=100))
        #("rbf5", RBFSampler(gamma=0.25, n_components=1000)),
        #("rbf6", RBFSampler(gamma=0.05, n_components=100)),
        #("rbf7", RBFSampler(gamma=0.01, n_components=100)),
        #("rbf8", RBFSampler(gamma=1.25, n_components=100)),
        #("rbf9", RBFSampler(gamma=1.75, n_components=100)),
        #("rbf10", RBFSampler(gamma=2.25, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))



class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant", shuffle=True, max_iter = 5, tol = None)
            
            model.partial_fit([self.featurize_state(env.reset())], [0])
            #model.fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        #print(featurized[0], len(featurized[0]))
        return featurized[0]
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        features = self.featurize_state(s)
        #print(features, len(features))
        
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        
        
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])
        #self.models[a].fit([features], [y])




def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def run_last_epsiode(stats, estimator):
    
    epsilon = 0.0
    epsilon_decay = 1.0
    discount_factor = 1.0
    num_episodes = 10
    
    
    for i_episode in range(num_episodes):
        
        policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        
        # Only used for SARSA, not Q-Learning
        next_action = None
        state = env.reset()
        
        observation = env.reset()
        last_reward = stats.episode_rewards[i_episode - 1]
        # One step in the environment        
        for t in itertools.count():
                        
            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            env.render()
            
            #epsilon = 1.0/math.sqrt(t+1.0)
            if next_action is None:
                action_probs = policy(state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action
            
            # Take a step
            next_state, reward, done, _ = env.step(action)
            
            
            # TD Update
            q_values_next = estimator.predict(next_state)
            
            # Use this code for Q-Learning
            # Q-Value TD Target
            
            td_target = (reward + discount_factor * np.max(q_values_next))
            #print(estimator.predict(state)[action])
            
            # Use this code for SARSA TD Target for on policy-training:
            # next_action_probs = policy(next_state)
            # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)             
            # td_target = reward + discount_factor * q_values_next[next_action]
            
            # Update the function approximator using our target
            
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")
                
            if done:
                break
                
            state = next_state
            env.render()
            print(observation)
        
        '''
        observation, reward, done, info = env.step(action)
        print(reward, action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        '''

def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.0, epsilon_decay=1.0):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_tderror=np.zeros(num_episodes))
        
    
    
    
    flag = True
    
    
    estimator_episode = []
    for i_episode in range(num_episodes):
        
        estimator_episode.append(Estimator())
        estimator_episode[i_episode] = deepcopy(estimator)
        # The policy we're following
        policy = make_epsilon_greedy_policy(estimator_episode[i_episode], epsilon * epsilon_decay**i_episode, env.action_space.n)
        
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # Only used for SARSA, not Q-Learning
        next_action = None
        
        
        history = []
        tderror = []
        path_reward = 0
        
        # One step in the environment        
        for t in itertools.count():
                        
            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            
            #epsilon = 1.0/math.sqrt(t+1.0)
            if next_action is None:
                action_probs = policy(state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action
            
            # Take a step
            next_state, reward, done, _ = env.step(action)
            
            
            
            #Save history
            history.append([state,action,reward,next_state])
            
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            #print(next_state)
            
            if next_state[0] > 0.4:
                path_reward += 0.1
                print(path_reward)
            
            # TD Update
            q_values_next = estimator_episode[i_episode].predict(next_state)
            
            # Use this code for Q-Learning
            # Q-Value TD Target
            
            td_target = (reward + discount_factor * np.max(q_values_next) + path_reward)
            #print(estimator.predict(state)[action])
            #tderror.append(abs(td_target - estimator_episode[i_episode].predict(state)[action]))
            
            # Use this code for SARSA TD Target for on policy-training:
            # next_action_probs = policy(next_state, epsilon)
            # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)             
            # td_target = reward + discount_factor * q_values_next[next_action]
            
            tderror.append(abs(td_target - estimator_episode[i_episode].predict(state)[action]))
            
            # Update the function approximator using our target
                 
            estimator_episode[i_episode].update(state, action, td_target)
            
            
            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")
                
            if done:
                break
                
            state = next_state
            
        stats.episode_tderror[i_episode] = np.mean(tderror)   
        
        
        take = []
        slice = 10
        for i in range(slice,slice*2):
            take.append(stats.episode_tderror[i_episode - i-1])
                 
        take1 = []
        for i in range(0,slice):
            take1.append(stats.episode_tderror[i_episode - i])
        
        
        #if stats.episode_tderror[i_episode - 1] < stats.episode_tderror[i_episode]:
        #if (np.mean(take) > np.mean(take1)) and flag == True:
        #if (np.mean(take) > np.mean(take1)) and i_episode < 1500:
        if i_episode < 1500:    
            
            num = random.uniform(0,1)
            
            if num < 0.05:
                #+ 1.0/math.sqrt(1 + i_episode)
                #eps = random.randint(0,slice-1)
                take2 = []
                
                for i in range(0,i_episode+1):
                    take2.append(stats.episode_tderror[i])
                print(take2)
                eps = np.argmin(take2)
            else:
                eps = i_episode
                
            print("update:", np.mean(take), np.mean(take1), eps)
            
            estimator = deepcopy(estimator_episode[eps])
            
            #estimator.clone(estimator_episode)
            
            #print(estimator.models[0].coef_)
            #print(estimator_episode.models[0].coef_)
            estimator.update(state, action, td_target)
        
        elif(i_episode >= 1500 and flag == True):
              
            take2 = []
            for i in range(0,i_episode+1):
                take2.append(stats.episode_rewards[i])
                
            eps = np.argmax(take2)
            
            estimator = deepcopy(estimator_episode[eps])  
            flag = False
        
        '''
        take = []
        slice = 5
        if i_episode > 100:
            for i in range(0,slice):
                take.append(stats.episode_rewards[i_episode - i])
            
            if np.mean(take) >= -110.0:
                print("Reached")
                
                flag = False
        '''
        f = open('history/QLearning/Eps'+str(i_episode + 1)+'.txt', 'w')
        for r in range(0,len(history)):
            f.write(str(list(history[r][0])) + ", " + str(history[r][1]) + ", " + str(history[r][2]) + ", " + str(list(history[r][3]))+"\n")
        f.close()
        
    
    #estimator = deepcopy(estimator)
    
    take2 = []
    for i in range(0,i_episode+1):
        take2.append(stats.episode_rewards[i])
        
    np.argmax(take2)
    
    estimator = deepcopy(estimator_episode[eps])
    
    run_last_epsiode(stats, estimator)    
    
    
    return stats, estimator





estimator = Estimator()





# Note: For the Mountain Car we don't actually need an epsilon > 0.0
# because our initial estimate for all states is too "optimistic" which leads
# to the exploration of all states.
stats, estimator = q_learning(env, estimator, 2000, epsilon=0.0)





plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)





