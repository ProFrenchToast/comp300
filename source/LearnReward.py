from baselines.common.vec_env import VecFrameStack

from AgentClasses import *
from baselines.common.cmd_util import make_vec_env


#a method to find all the models in a given dir that are just numbers
def Find_all_Models(model_dir):
    from os import listdir
    from os.path import isfile, join
    import re

    checkpoints = []
    filesandDirs = listdir(model_dir)
    allFiles = []
    for i in filesandDirs:
        if isfile(join(model_dir, i)):
            allFiles.append(i)

    for file in allFiles:
        if re.match('^[0-9]+$',file.title()):
            checkpoints.append(file.title())

    return checkpoints

#given a dir and an environment plus an agent load every checkpointed model and run it for some demos
def generate_demos_from_checkpoints(env, agent, model_dir, demosPerModel):
    checkpoints = Find_all_Models(model_dir)
    print('found models: ' + str(checkpoints))


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
            #the shape of the observations is (1,84,84, 4) so take only the first slice to remove the first dimension
            shapedObservations = currentObservation[0]
            timeSteps = 0
            done = False

            #run the demo
            while True:
                observations.append(shapedObservations)
                totalReward += currentReward

                action = agent.act(currentObservation,  currentReward, done)
                currentObservation, currentReward, done, info = env.step(action)
                shapedObservations = currentObservation[0]
                timeSteps += 1

                if done:
                    observations.append(shapedObservations)
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

#todo: make a method to split the trajectories into smaller subsets + make the labels

#todo: train the nn on the saved demos and display acc


if __name__ == '__main__':
    model_path = "/home/patrick/Downloads/breakout_25"
    env_id = 'BreakoutNoFrameskip-v4'
    env_type = 'atari'

    env = make_vec_env(env_id, env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })
    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, 'atari', True)

    trajectories, rewards = generate_demos_from_checkpoints(env, agent, model_path, 1)

    #now try to create the network and see the output
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewardNetwork = RewardNetwork()
    rewardNetwork.to(device)
    trajectory_i = np.array(trajectories[0])
    trajectory_j = np.array(trajectories[1])
    traj_i = torch.from_numpy(trajectory_i).float().to(device)
    traj_j = torch.from_numpy(trajectory_j).float().to(device)

    output, abs_rewards = rewardNetwork.forward(traj_i, traj_j)

    print(str(output))
