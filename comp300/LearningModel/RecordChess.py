import random
import sys
import cv2

import chess
import chess.engine
import torch
from baselines.common.cmd_util import make_vec_env
from gym import register
from torch import nn, optim

from comp300.LearningModel.AgentClasses import RewardNetwork
from comp300.LearningModel.LearnReward import create_training_test_labels, train_reward_network, calc_accuracy
from comp300.LearningModel.cmd_utils import chessLearnRewardParser


def generate_chess_demos(env, num_demos, save_path):
    """
    Generate a number of demonstrations using stockfish.

    Parameters
    ----------
    env : ChessEnv
        The chess env that is used for creating the demonstrations.
    num_demos : int
        The number of demonstrations to be created.

    Returns
    -------
    Tuple
        A tuple containing the demonstrations, as an array of observations, and the rewards, an array of int showing
        the reward for each demonstration.

    """
    trajectories = []
    rewards = []
    engine = chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish')
    rootEnv = env.envs[0].env.env.env #who the fuck though this was a good arcitecture to make?
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, (400, 400))

    for i in range(num_demos):
        done = False

        obs = env.reset()


        while not done:
            img = rootEnv.render(mode='rgb_array')
            out.write(img)
            cv2.waitKey(delay=1)
            moves = list(rootEnv.board.generate_legal_moves())

            result = engine.play(rootEnv.board, limit=chess.engine.Limit(depth=1))
            move = moves.index(result.move)

            obs, reward, done, _ = env.step(move)

    out.release()


if __name__ == '__main__':
    register(id='ChessSelf-v0',
             entry_point='comp300.Chess.ChessWrapper:ChessEnv',
             max_episode_steps=1000)
    env_id = 'ChessSelf-v0'
    env_type = 'ChessWrapper'

    env = make_vec_env(env_id, env_type, 1, 0,
                       flatten_dict_observations=True,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })

    generate_chess_demos(env, 3, "/home/patrick/chessDemo.mp4")

    print("done recording")