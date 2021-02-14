import sys

import cv2
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env import VecFrameStack
from gym import register

from comp300.LearningModel.AgentClasses import PPO2Agent
from comp300.LearningModel.cmd_utils import recordAgentParser


def RecordAgent(env, agent, num_episodes, save_path, render=False):
    """
    Runs the agent in the environment and records an mp4 video.

    Parameters
    ----------
    env : gym.env
        The env the agent was trained for.
    agent : comp300.LearningModel.AgentClasses.PPO2Agent
        The agent that will be recorded.
    num_episodes : int
        The number of episodes to run through in this recording.
    save_path : str
        The path to save the video to.
    render : bool
        If the env should render to the screen while recording.

    Returns
    -------

    """

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 15, (500,500))

    for i in range(num_episodes):
        totalReward = 0

        obs = env.reset()
        r = 0
        done = False

        while True:
            img = env.render(mode='rgb_array')
            out.write(img)
            if render:
                cv2.imshow(env_type, img)
            cv2.waitKey(delay=1)
            action = agent.act(obs, r, done)
            obs, r, done, info = env.step(action)
            totalReward += r

            if done:
                break


if __name__ == '__main__':
    register(id='ChessSelf-v0',
             entry_point='comp300.Chess.ChessWrapper:ChessEnv',
             max_episode_steps=1000)

    parser = recordAgentParser()
    args, unknown_args = parser.parse_known_args(sys.argv)

    model_path = args.model_path
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
    num_episodes = args.num_episodes
    save_path = args.save_path
    render = args.render

    RecordAgent(env, agent, num_episodes, save_path, render)
