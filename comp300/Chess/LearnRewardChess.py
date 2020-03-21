import random
import sys

import chess
import chess.engine
import torch
from baselines.common.cmd_util import make_vec_env
from gym import register
from torch import nn, optim

from comp300.LearningModel.AgentClasses import RewardNetwork
from comp300.LearningModel.LearnReward import create_training_test_labels, train_reward_network, calc_accuracy
from comp300.LearningModel.cmd_utils import chessLearnRewardParser


def generate_chess_demos(env, num_demos):
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

    for i in range(num_demos):
        traj = []
        done = False
        skill = random.choice(range(0, 21))
        if skill != 21: #random
            engine.configure({"Skill Level": skill})

        obs = env.reset()
        traj.append(obs)

        while not done:
            moves = list(rootEnv.board.generate_legal_moves())
            if skill == 21:
                move = random.choice(range(len(moves)))
            else:
                result = engine.play(rootEnv.board, limit=chess.engine.Limit(depth=1))
                move = moves.index(result.move)

            obs, reward, done, _ = env.step(move)
            traj.append(obs)

        print("  {}:finished demo with skill level {}, Result = {}".format(i,skill, reward))
        trajectories.append(traj)
        if skill ==21:
            rewards.append(reward - 5)
        else:
            rewards.append(reward + (skill*2))

    return trajectories, rewards

if __name__ == '__main__':
    register(id='ChessSelf-v0',
             entry_point='comp300.Chess.ChessWrapper:ChessEnv',
             max_episode_steps=1000)

    parser = chessLearnRewardParser()
    args, unknown_args = parser.parse_known_args(sys.argv)

    env_id = 'ChessSelf-v0'
    env_type = 'ChessWrapper'

    env = make_vec_env(env_id, env_type, 1, 0,
                       flatten_dict_observations=True,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })

    trajectories, rewards = generate_chess_demos(env, args.num_demonstrations)
    training_obs, training_labels, test_obs, test_labels = create_training_test_labels(0.5, trajectories, rewards,
                                                                                       args.num_full_demonstrations * 2,
                                                                                       args.num_sub_demonstrations * 2,
                                                                                       args.min_snippet_length,
                                                                                       args.max_snippet_length)
    trajectories = 0
    rewards = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    learning_rate = args.learning_rate
    weight_decay = 0
    network = RewardNetwork(loss, env=env, env_type=args.env_type)
    network.to(device)
    optimiser = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_reward_network(network, optimiser, training_obs, training_labels, args.training_epochs,
                         checkpoint_dir=args.checkpoint_dir)
    torch.save(network.state_dict(), args.save_path)
    accuracy = calc_accuracy(network, test_obs, test_labels)
    print("accuracy of test network is {}%".format(accuracy))