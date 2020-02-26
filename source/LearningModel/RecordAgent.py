from baselines.common.vec_env import VecFrameStack

from LearningModel.AgentClasses import *
from baselines.common.cmd_util import make_vec_env

def load_basePPO_and_Display():

    from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
    model_path = "/home/patrick/logs/Agent/IRLTest/checkpoints/00001"
    env_id = 'BreakoutNoFrameskip-v4'
    env_type = 'atari'
    record = True

    env = make_vec_env(env_id, env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })

    if record:
        env = VecVideoRecorder(env, '../videos/', lambda steps: True, 2000000)
    env = VecFrameStack(env, 4)

    agent = PPO2Agent(env, 'atari', True)
    agent.load(model_path)

    for i in range(1):
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


if __name__ == '__main__':
    load_basePPO_and_Display()