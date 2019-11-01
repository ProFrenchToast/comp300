from baselines.common.vec_env import VecFrameStack

from AgentClasses import *
from baselines.common.cmd_util import make_vec_env


#todo: load a directory of checkpointed models
#a method to find all the models in a given dir that are just numbers
def Find_all_Models(model_dir):
    from os import listdir
    from os.path import isfile, join
    import re

    checkpoints = []
    allFiles = [file for file in listdir(model_dir) if isfile(join(model_dir, file))]
    for file in allFiles:
        if re.match('^[0-9]+$',file.title()):
            checkpoints.append(file.title())

    return checkpoints

#given a dir and an environment plus an agent load every checkpointed model and run it for some demos
def generate_demos_from_checkpoints(env, agent, model_dir, demosPerModel):
    checkpoints = Find_all_Models(model_dir)
    print('found models: ' + checkpoints)


    TrajectoryObservations = [] # the set of observations for each demo
    TrajectoryTotalRewards = [] #the total reward of each demo

    #now loop over each model and load it
    for model in checkpoints:
        model_path = model_dir + '/' + model
        agent.load(model_path)

        for demo in range(demosPerModel):
            observations = []
            totalReward = 0

            currentReward = 0
            currentObservation = env.reset()
            timeSteps = 0
            done = False

            #run the demo
            while True:
                observations.append(currentObservation)
                totalReward += currentReward

                action = agent.act(currentObservation,  currentReward, done)
                currentObservation, currentReward, done = env.step(action)
                timeSteps += 1

                if done:
                    observations.append(currentObservation)
                    totalReward += currentReward
                    print("generated demo from model at {}, demo length: {}, total reward: {}".
                          format(model_path, timeSteps, totalReward))
                    break

            #save the results
            TrajectoryObservations.append(observations)
            TrajectoryTotalRewards.append(totalReward)

    print("finished trajectory generation, Total demos: {}, min reward: {}, max reward: {}".format(
        len(TrajectoryObservations), min(TrajectoryTotalRewards), max(TrajectoryTotalRewards)))

    return TrajectoryObservations, TrajectoryTotalRewards


#todo: run each model and save the demonstrations

#todo: make nn for the demos

#todo: train the nn on the saved demos and display acc


if __name__ == '__main__':
    model_path = "~/Downloads/breakout_25"
    env_id = 'BreakoutNoFrameskip-v4'
    env_type = 'atari'

    env = make_vec_env(env_id, env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })
    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env)

    trajectories, rewards = generate_demos_from_checkpoints(env, agent, model_path, 1)
