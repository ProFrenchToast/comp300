import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class RewardNetwork (nn.Module):
    #setup the nn by initialising the layers plus other variables
    def __init__(self):
        super.__init__()
        #add the nn layers

    #use the nn to predict the reward for a given trajectory
    def predict_reward(self, trajectory):
        return

    #use the nn on two trajectories to find the better one
    def forward(self, trajecory_i, trajectory_J):
        return #add the forward method on two trajectories


#a generic class to hold all types of agents
class Agent (object):
    def __init__(self, env):
        return

    def act(self, observation, reward, done):
        return

class RandomAgent(Agent):
    #a simple agent that just takes a random action
    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

#a base class for ppo2 agents
#currently uses random agents as a stub
class PPO2Agent(RandomAgent):
    def __init__(self, env, env_type, stochastic):
        super(PPO2Agent,self).__init__(env)

    def load(self, path):
        print("loaded model:" + path)

    def act(self, observation, reward, done):
        return super(PPO2Agent, self).act(observation, reward, done)


#todo: implement both of them