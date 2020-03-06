from baselines.common.vec_env import VecFrameStack
from gym import register

from LearningModel.AgentClasses import *
from baselines.common.cmd_util import make_vec_env
from statistics import stdev

def getReward(agent, env):
    totalReward = 0

    obs = env.reset()
    r = 0
    done = False

    while True:
        env.render()
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
    register(id='ChessSelf-v0',
             entry_point='Chess.ChessWrapper:ChessEnv',
             max_episode_steps=1000)
    model_path = "/home/patrick/models/chessTest/chess20Mppo2"
    env_id = 'ChessSelf-v0'
    env_type = 'ChessWrapper'

    env = make_vec_env(env_id, env_type, 1, 0,
                       flatten_dict_observations=True,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })
    #env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, True)
    agent.load(model_path)

    meanR, minR, maxR, std = getAvgReward(agent, env, 20)

    print("the 2e7 timestep model on ground truth achived mean: {}, min: {}, max: {}, std: {}"
          .format(meanR, minR,maxR, std))


