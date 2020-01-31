from baselines.common.vec_env import VecFrameStack

from AgentClasses import *
from baselines.common.cmd_util import make_vec_env
from statistics import stdev

def getReward(agent, env):
    totalReward = 0

    obs = env.reset()
    r = 0
    done = False

    while True:
        action = agent.act(obs, r, done)
        obs, r, done, info = env.step(action)
        totalReward += r

        if done:
            break
    return  totalReward

def getAvgReward(agent, env, iterations):
    rewards = []

    for i in range(iterations):
        rewards.append(float(getReward(agent, env)))

    mean = sum(rewards) / len(rewards)
    minR = min(rewards)
    maxR = max(rewards)
    std = stdev(rewards)

    return mean, minR, maxR, std

if __name__ == '__main__':
    model_path = "/home/patrick/models/breakout-reward-RL2/breakout_50M_ppo2"
    env_id = 'BreakoutNoFrameskip-v4'
    env_type = 'atari'

    env = make_vec_env(env_id, env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })
    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, 'atari', True)
    agent.load(model_path)

    meanR, minR, maxR, std = getAvgReward(agent, env, 20)

    print("the 2e7 timestep model on ground truth achived mean: {}, min: {}, max: {}, std: {}"
          .format(meanR, minR,maxR, std))


