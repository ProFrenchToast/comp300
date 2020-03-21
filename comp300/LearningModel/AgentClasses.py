import torch
import torch.nn as nn
import torch.nn.functional as functional
from baselines.common.policies import build_policy
from baselines.ppo2.model import Model
from gym.envs.atari import atari_env

class RewardNetwork (nn.Module):
    """The neural network used to approximate the reward function of demonstrations."""
    def __init__(self, lossFunction, env_type, env=atari_env.AtariEnv()):
        """
        The constructor that initialises the neural network using the environment.

        Parameters
        ----------
        lossFunction : torch.nn.loss
            The loss function to be used during the training of the network.
        env_type : str
            The type of environment the network needs to approximate.
        env : gym.env
            The actual environment the network is interacting with. only needed when is it not a visual env like atari.
        """
        super().__init__()
        #check to see if this is a visual domain that convolutions would make sense for
        self.is_visual = env_type == "atari"

        if self.is_visual: #if atari env
            #todo: try different network configurations
            #just use the layers from the paper to start
            self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
            self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
            self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
            self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
            self.fc1 = nn.Linear(784, 64)
            self.fc2 = nn.Linear(64, 1)
        else:
            #first calculate the full size of the observations
            self.obs_size = 0
            for i in range(len(env.observation_space.shape)):
                self.obs_size += env.observation_space.shape[i]

            #now make fc layers that fit the size of the observations
            self.fc1 = nn.Linear(self.obs_size, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, 1)

        self.lossFunction = lossFunction


    def predict_reward(self, trajectory):
        """
        Predict the reward of the trajectory by applying the network to each frame stack.

        Parameters
        ----------
        trajectory : tensor
            The trajectory which is just an array of observations directly from the env.

        Returns
        -------
        Tuple
            A tuple of floats containing the reward, which is the sum of the reward in each observation, and the
            abs_reward, which is the sum of absolute rewards in each observation.
        """
        if self.is_visual:
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
        else:
            x = trajectory.view(-1, self.obs_size)
            x = functional.leaky_relu((self.fc1(x)))
            x = functional.leaky_relu((self.fc2(x)))
            x = functional.leaky_relu((self.fc3(x)))
            reward = torch.sum(x)
            abs_reward=torch.sum(torch.abs(x))
            return reward, abs_reward

    #use the nn on two trajectories to find the better one
    def forward(self, trajectory_i, trajectory_j):
        """
        Classify which of the two trajectories has higher reward.

        Parameters
        ----------
        trajectory_i : tensor
            A tensor containing the first trajectory corresponding to class 0.
        trajectory_j : tensor
            A tensor containing the second trajectory corresponding to class 1.

        Returns
        -------
        Tuple
            A tuple containing a tensor, which contains the predicted rewards of each trajectory used as the
            classification values, and the sum of absolute rewards of both trajectories.

        """
        reward_i, abs_reward_i = self.predict_reward(trajectory_i)
        reward_j, abs_reward_j = self.predict_reward(trajectory_j)
        #now use the predicted rewards for each trajectory as a probability distribution for that traj being better
        #note the distribution needs to be unsqueezed to have an extra dimension because it need to be a batch of size 1
        return torch.cat((reward_i.unsqueeze(0), reward_j.unsqueeze(0)), 0), (abs_reward_i + abs_reward_j)



class Agent (object):
    """The generic class to hold all types of agents"""
    def __init__(self, env):
        return

    def act(self, observation, reward, done):
        return

class RandomAgent(Agent):
    """a simple agent that just takes a random action."""
    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class PPO2Agent(Agent):
    """The agent used to load all agents trained from baselines"""
    def __init__(self, env, env_type, stochastic):
        """
        The constructor that uses the environment to constuct the network build policy and then build the agent.

        Parameters
        ----------
        env : gym.env
            The env the agent needs to interact with.
        env_type : str
            The type of env.
        stochastic : bool
            A bool describing if the behavior of the agent is stochastic (random in simple terms).
        """
        ob_space = env.observation_space
        ac_space = env.action_space
        self.stochastic = stochastic

        #now find the correct build policy
        if env_type == 'atari':
            policy = build_policy(env, 'cnn')
        elif env_type == "ChessWrapper":
            policy = build_policy(env, 'mlp', {'num_layers':5})
        else:
            policy = build_policy(env, 'mlp')

        #construct the agent model using the build model
        make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=1,
                                   nsteps=1, ent_coef=0., vf_coef=0.,
                                   max_grad_norm=0.)
        self.model = make_model()

    def load(self, path):
        """
        Loads a set of weights from a file path.

        Parameters
        ----------
        path : str
            The path to the agent file.

        Returns
        -------

        """
        self.model.load(path)

    def act(self, observation, reward, done):
        """
        Given the current observations calculate the correct action.

        Parameters
        ----------
        observation : numpy array
            The current observations as a subset of the observation space.
        reward : float
            The reward given the previous timestep.
        done : bool
            A bool telling if the current episode is finished.

        Returns
        -------
        Action
            return the best action the agent can find from the action space.
        """
        if self.stochastic:
            a, v, state, neglogp = self.model.step(observation)
        else:
            a = self.model.act_model.act(observation)
        return a


