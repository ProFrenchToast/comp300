from baselines.common.vec_env import VecFrameStack
import cv2
from LearningModel.AgentClasses import *
from baselines.common.cmd_util import make_vec_env

def load_basePPO_and_Display():

    from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
    model_path = "/home/patrick/models/halfcheetah-reward-rl/halfcheetah_2M_ppo2"
    env_id = 'HalfCheetah-v2'
    env_type = 'mujoco'
    record = False

    env = make_vec_env(env_id, env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })

    out = cv2.VideoWriter('../videos/HalfCheetah_agent2M.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (500,500))
    if record:
        env = VecVideoRecorder(env, '../videos/', lambda steps: True, 2000000)
    #env = VecFrameStack(env, 4)

    agent = PPO2Agent(env, env_type, True)
    agent.load(model_path)

    for i in range(2):
        totalReward = 0

        obs = env.reset()
        r = 0
        done = False

        while True:
            img = env.render(mode='rgb_array')
            out.write(img)
            cv2.imshow('window', img)
            cv2.waitKey(delay=1)
            action = agent.act(obs, r, done)
            obs, r, done, info = env.step(action)
            totalReward += r

            if done:
                break


if __name__ == '__main__':
    load_basePPO_and_Display()
