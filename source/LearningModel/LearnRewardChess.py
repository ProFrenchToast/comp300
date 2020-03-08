from LearningModel.AgentClasses import *
from LearningModel.LearnReward import *
from gym import register
import chess
import chess.engine
import random

def generate_chess_demos(env, num_demos):
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
             entry_point='Chess.ChessWrapper:ChessEnv',
             max_episode_steps=1000)

    env_id = 'ChessSelf-v0'
    env_type = 'ChessWrapper'

    env = make_vec_env(env_id, env_type, 1, 0,
                       flatten_dict_observations=True,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })

    trajectories, rewards = generate_chess_demos(env, 10000)
    trajectories = np.array(trajectories)
    rewards = np.array(rewards)
    training_obs, training_labels, test_obs, test_labels = create_training_test_labels(0.5, trajectories, rewards, 120000,
                                                                                       0, 50, 150)
    trajectories = 0
    rewards = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    learning_rate = 0.00005
    weight_decay = 0
    network = RewardNetwork(loss, env=env)
    network.to(device)
    optimiser = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_reward_network(network, optimiser, training_obs, training_labels,5)
    torch.save(network.state_dict(), "/home/patrick/models/chess-reward/fullTest.params")
    accuracy = calc_accuracy(network, test_obs, test_labels)
    print("accuracy of test network is {}%".format(accuracy))