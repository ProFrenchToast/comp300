from baselines.common.vec_env import VecFrameStack
from gym import register
import sys
from comp300.LearningModel.AgentClasses import *
from baselines.common.cmd_util import make_vec_env
from statistics import stdev
from comp300.LearningModel.cmd_utils import getAverageParser

def getReward(agent, env, render=False):
    totalReward = 0

    obs = env.reset()
    r = 0
    done = False

    while True:
        if render:
            env.render()
        action = agent.act(obs, r, done)
        obs, r, done, info = env.step(action)
        totalReward += r

        if done:
            break
    return  totalReward

def getAvgReward(agent, env, iterations, render):
    rewards = []

    for i in range(iterations):
        rewards.append(float(getReward(agent, env, render)))

    mean = sum(rewards) / len(rewards)
    minR = min(rewards)
    maxR = max(rewards)
    std = stdev(rewards)

    return mean, minR, maxR, std

if __name__ == '__main__':
    register(id='ChessSelf-v0',
             entry_point='Chess.ChessWrapper:ChessEnv',
             max_episode_steps=1000)
    parser = getAverageParser()
    args, unknown_args = parser.parse_known_args(sys.argv)
    model_path = args.model_dir
    env_id = args.env
    env_type = args.env_type

    if env_type == "atari":
        env = make_vec_env(env_id, env_type, 1, 0,
                           wrapper_kwargs={
                               'clip_rewards': False,
                               'episode_life': 10000,
                           })
        env = VecFrameStack(env, 4)
    else:
        env = make_vec_env(env_id, env_type, 1, 0,
                           flatten_dict_observations=True,
                           wrapper_kwargs={
                               'clip_rewards': False,
                               'episode_life': 10000,
                           })
    agent = PPO2Agent(env, env_type, True)
    agent.load(model_path)

    meanR, minR, maxR, std = getAvgReward(agent, env, args.num_episodes, args.render)

    print("Tested model {} for {} episodes, ground truth reward: {}, min: {}, max: {}, std: {}"
          .format(model_path, args.num_episodes, meanR, minR,maxR, std))


