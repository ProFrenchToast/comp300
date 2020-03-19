import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from baselines.common.vec_env import VecEnvWrapper

#this is a copy of the reward network from AgentClasses.py
from gym.envs.atari import atari_env


class AtariRewardNetwork (nn.Module):
    # setup the nn by initialising the layers plus other variables
    def __init__(self, env=atari_env.AtariEnv()):
        super().__init__()
        # check to see if this is a visual domain that convolutions would make sense for
        self.is_visual = isinstance(env, atari_env.AtariEnv)

        if self.is_visual:  # if atari env
            # todo: try different network configurations
            # just use the layers from the paper to start
            self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
            self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
            self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
            self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
            self.fc1 = nn.Linear(784, 64)
            self.fc2 = nn.Linear(64, 1)
        else:
            # todo: make the size and network topology change based on the shapes of the obs
            # first calculate the full size of the observations
            self.obs_size = 0
            for i in range(len(env.observation_space.shape)):
                self.obs_size += env.observation_space.shape[i]

            # now make fc layers that fit the size of the observations
            self.fc1 = nn.Linear(self.obs_size, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, 1)



    # use the nn to predict the reward for a given trajectory
    def predict_reward(self, trajectory):
        if self.is_visual:
            # first change the trajectpory into the NCHW format
            # number of samples * channels * height * width
            x = trajectory.permute(0, 3, 2, 1)

            # now pass the trajectory through the network
            x = functional.leaky_relu((self.conv1(x)))
            x = functional.leaky_relu((self.conv2(x)))
            x = functional.leaky_relu((self.conv3(x)))
            x = functional.leaky_relu((self.conv4(x)))
            x = x.view(-1, 784)
            x = functional.leaky_relu(self.fc1(x))
            x = self.fc2(x)  # this gives a single value FOR EACH OBSERVATION GIVEN and so must be summed
            reward = torch.sum(x)  # sum up the reward for each state in the trajectory
            abs_reward = torch.sum(torch.abs(x))
            return reward, abs_reward
        else:
            x = trajectory.view(-1, self.obs_size)
            x = functional.leaky_relu((self.fc1(x)))
            x = functional.leaky_relu((self.fc2(x)))
            x = functional.leaky_relu((self.fc3(x)))
            reward = torch.sum(x)
            abs_reward = torch.sum(torch.abs(x))
            return reward, abs_reward

    # use the nn on two trajectories to find the better one
    def forward(self, trajectory_i, trajectory_j):
        reward_i, abs_reward_i = self.predict_reward(trajectory_i)
        reward_j, abs_reward_j = self.predict_reward(trajectory_j)
        # now use the predicted rewards for each trajectory as a probability distribution for that traj being better
        # note the distribution needs to be unsqueezed to have an extra dimension because it need to be a batch of size 1
        return torch.cat((reward_i.unsqueeze(0), reward_j.unsqueeze(0)), 0), (abs_reward_i + abs_reward_j)

class VecPytorchRewardEnv(VecEnvWrapper):
    def __init__(self, venv, reward_net_path, env_name, env_type):
        VecEnvWrapper.__init__(self, venv)
        self.env_name = env_name
        self.reward_net = AtariRewardNetwork(venv)

        self.reward_net.load_state_dict(torch.load(reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        with torch.no_grad():
            customReward, absCustomReward = self.reward_net.predict_reward(torch.from_numpy(np.array(obs)).float().to(self.device))
        customReward = customReward.cpu().numpy().squeeze()
        return obs, customReward, news, infos

    def reset(self):
        obs = self.venv.reset()

        return obs