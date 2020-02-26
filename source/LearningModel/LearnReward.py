from baselines.common.vec_env import VecFrameStack

from LearningModel.AgentClasses import *
from baselines.common.cmd_util import make_vec_env
import torch.optim as optim
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import re


#a method to find all the models in a given dir that are just numbers
def Find_all_Models(model_dir):

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
        model_path = join(model_dir, model)
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


def create_labels(demonstrations, demo_rewards, num_full_trajectories, num_sub_trajectories,
                         min_snippet_length, max_snippet_length):
    trajectory_observations = []
    trajectory_labels = []
    num_demos = len(demonstrations)

    if (len(demonstrations) != len(demo_rewards)) | (num_demos <= 1):
        print("Error: {} demos but {} reward labels".format(len(demonstrations), len(demo_rewards)))
        exit(1)

    #first make the full trajectories
    for n in range(num_full_trajectories):
        ti_index = 0
        tj_index = 0
        #find 2 different trajectories
        while (ti_index == tj_index):
            ti_index = np.random.randint(num_demos)
            tj_index = np.random.randint(num_demos)

        traj_i = demonstrations[ti_index]
        traj_j = demonstrations[tj_index]
        reward_i = demo_rewards[ti_index]
        reward_j = demo_rewards[tj_index]

        if reward_i > reward_j:
            label = 0
        else:
            label = 1

        trajectory_observations.append((traj_i, traj_j))
        trajectory_labels.append(label)

    print("created {} full demos".format(len(trajectory_observations)))

    #now make all the sub trajectories
    for n in range(num_sub_trajectories):
        ti_index = 0
        tj_index = 0
        # find 2 different trajectories
        while (ti_index == tj_index):
            ti_index = np.random.randint(num_demos)
            tj_index = np.random.randint(num_demos)

        #now find the start and end points for the sub trajectories
        sub_trajectory_length = np.random.randint(min_snippet_length, max_snippet_length)
        #find the latest starting point in each demo to avoid out of bounds
        ti_latest_start = len(demonstrations[ti_index]) - sub_trajectory_length -1
        tj_latest_start = len(demonstrations[tj_index]) - sub_trajectory_length -1

        ti_start = np.random.randint(0, ti_latest_start)
        tj_start = np.random.randint(0, tj_latest_start)

        ti_end = ti_start + sub_trajectory_length
        tj_end = tj_start + sub_trajectory_length

        #possibly change to skip half the frames
        traj_i = demonstrations[ti_index][ti_start:ti_end]
        traj_j = demonstrations[tj_index][tj_start:tj_end]
        reward_i = demo_rewards[ti_index]
        reward_j = demo_rewards[tj_index]

        if reward_i > reward_j:
            label = 0
        else:
            label = 1

        if len(traj_i) ==0 or len(traj_j) == 0:
            print("found size of 0")
        trajectory_observations.append((traj_i, traj_j))
        trajectory_labels.append(label)

    print("created {} sub demos".format(num_sub_trajectories))

    return trajectory_observations, trajectory_labels

#split the demos and create a test and training labels
def create_training_test_labels(ratio, demonstrations, demo_rewards, num_full_trajectories, num_sub_trajectories,
                         min_snippet_length, max_snippet_length):
    assert len(demonstrations) == len(demo_rewards)
    shufflePermutation = np.random.permutation(len(demonstrations))
    copyDemonstrations = demonstrations[shufflePermutation]
    copyDemo_rewards = demo_rewards[shufflePermutation]

    numTrainingDemos = np.int(np.floor(ratio * len(demonstrations)) +1)
    numTestDemos = len(demonstrations) - numTrainingDemos

    training_demos = copyDemonstrations[0:numTrainingDemos]
    testing_demos = copyDemonstrations[numTrainingDemos:numTrainingDemos + numTestDemos]

    training_rewards = copyDemo_rewards[0:numTrainingDemos]
    testing_rewards = copyDemo_rewards[numTrainingDemos:numTrainingDemos + numTestDemos]

    training_full_traj = np.int(np.floor(ratio * num_full_trajectories) +1)
    testing_full_traj = num_full_trajectories - training_full_traj

    training_sub_traj = np.int(np.floor(ratio * num_sub_trajectories) +1)
    testing_sub_traj = num_sub_trajectories - training_sub_traj

    training_observations, training_labels = create_labels(training_demos, training_rewards, training_full_traj,
                                                           training_sub_traj, min_snippet_length, max_snippet_length)
    testing_observations, testing_labels = create_labels(testing_demos, testing_rewards, testing_full_traj,
                                                         testing_sub_traj, min_snippet_length, max_snippet_length)

    return training_observations, training_labels, testing_observations, testing_labels



def train_reward_network(rewardNetwork, optimiser, training_trajectories, training_labels, training_epochs, checkpoint_dir = ""):
    #first check if gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("training using {}".format(device))

    saveCheckpoints = True
    cumulative_loss = 0
    if checkpoint_dir == "":
        print("no checkpoint directory set, no checkpoints will be saved")
        saveCheckpoints = False

    #zip the inputs and labels together to shuffle them for each epoch
    training_data = list(zip(training_trajectories, training_labels))
    for epoch in range(training_epochs):
        np.random.shuffle(training_data)
        shuffled_trajectories, shuffled_labels = zip(*training_data)
        epoch_loss = 0

        #now loop over every trajectory in the dataset
        for i in range(len(shuffled_labels)):
            traj_i, traj_j = shuffled_trajectories[i]
            labels = np.array([shuffled_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #print("traj_I:{}, Traj_J: {}".format(traj_i.size(), traj_j.size()))

            #zero out the gradient before applying to netwrok
            optimiser.zero_grad()

            #apply forwards on the trajectories then apply backwards to get the gradient from the loss tensor
            output, abs_reward = rewardNetwork.forward(traj_i, traj_j)
            output = output.unsqueeze(0)
            loss = rewardNetwork.lossFunction(output, labels)
            loss.backward()
            optimiser.step()

            loss_value = loss.item()
            cumulative_loss += loss_value
            epoch_loss += loss_value

        epoch_avg_loss = epoch_loss / len(shuffled_labels)
        print("Epoch: {}, Cumulative loss: {}, loss this epoch: {}". format(epoch, cumulative_loss, epoch_avg_loss))
        if saveCheckpoints:
            print("saving checkpoint {} to dir: {}".format(epoch, checkpoint_dir))
            torch.save(rewardNetwork.state_dict((), checkpoint_dir+ "/" + epoch))
    print("finished training reward network")


def calc_accuracy(reward_network, test_trajectories, testing_lables):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(test_trajectories)):
            label = testing_lables[i]
            traj_i, traj_j = test_trajectories[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(test_trajectories)


if __name__ == '__main__':
    model_path = "/home/patrick/models/fullGuiTest"
    env_id = 'BreakoutNoFrameskip-v4'
    env_type = 'atari'

    env = make_vec_env(env_id, env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })
    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, 'atari', True)
    #agent.load("/home/patrick/Downloads/breakout_25/00001")

    trajectories, rewards = generate_demos_from_checkpoints(env, agent, model_path, 1)
    trajectories = np.array(trajectories)
    rewards = np.array(rewards)
    training_obs, training_labels, test_obs, test_labels = create_training_test_labels(0.5, trajectories, rewards, 0, 12000, 50, 150)
    trajectories = 0
    rewards = 0


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    learning_rate = 0.00005
    weight_decay = 0
    network = RewardNetwork(loss)
    network.to(device)
    optimiser = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_reward_network(network, optimiser, training_obs, training_labels, 5)
    torch.save(network.state_dict(), "/home/patrick/models/fullGuiTest/fullTest.params")
    accuracy = calc_accuracy(network, test_obs, test_labels)
    print("accuracy of test network is {}%".format(accuracy))