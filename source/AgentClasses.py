import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class RewardNetwork (nn.Module):
    #setup the nn by initialising the layers plus other variables
    def __init__(self, lossFunction):
        super().__init__()

    #todo: try different network configurations
        #just use the layers from the paper to start
        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)

        self.lossFunction = lossFunction

    #use the nn to predict the reward for a given trajectory
    def predict_reward(self, trajectory):
        #first change the trajectpory into the NCHW format
        #number of samples * channels * height * width
        x = trajectory.permute(0,3,2,1)

        #now pass the trajectory through the network
        x = functional.leaky_relu((self.conv1(x)))
        x = functional.leaky_relu((self.conv2(x)))
        x = functional.leaky_relu((self.conv3(x)))
        x = functional.leaky_relu((self.conv4(x)))
        x = x.view(-1, 784)
        x = functional.leaky_relu(self.fc1(x))
        x = self.fc2(x) # this gives a single value FOR EACH OBSERVATION GIVEN and so must be summed
        reward = torch.sum(x) #sum up the reward for each state in the trajectory
        abs_reward = torch.sum(torch.abs(x))
        return reward, abs_reward

    #use the nn on two trajectories to find the better one
    def forward(self, trajectory_i, trajectory_j):
        reward_i, abs_reward_i = self.predict_reward(trajectory_i)
        reward_j, abs_reward_j = self.predict_reward(trajectory_j)
        #now use the predicted rewards for each trajectory as a probability distribution for that traj being better
        #note the distribution needs to be unsqueezed to have an extra dimension because it need to be a batch of size 1
        return torch.cat((reward_i.unsqueeze(0), reward_j.unsqueeze(0)), 0), (abs_reward_i + abs_reward_j)


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