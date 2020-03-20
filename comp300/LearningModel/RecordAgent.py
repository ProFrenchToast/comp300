from baselines.common.vec_env import VecFrameStack
import cv2
from comp300.LearningModel.AgentClasses import *
from comp300.LearningModel.cmd_utils import recordAgentParser
from baselines.common.cmd_util import make_vec_env
from gym import register
import sys

def load_basePPO_and_Display(args):
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

    out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 15, (500,500))

    for i in range(args.num_episodes):
        totalReward = 0

        obs = env.reset()
        r = 0
        done = False

        while True:
            img = env.render(mode='rgb_array')
            out.write(img)
            if args.render:
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

    load_basePPO_and_Display(args)
