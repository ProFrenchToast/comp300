import numpy as np
import torch
from baselines.common.vec_env import VecFrameStack
from baselines.common.cmd_util import make_vec_env
import torch.optim as optim
from gym import register
import tensorflow as tf

from os import listdir
from os.path import isfile, join
import re

from torch import nn

from comp300.LearningModel.AgentClasses import RewardNetwork, PPO2Agent
from comp300.LearningModel.cmd_utils import learnRewardParser
import sys
import pickle

def getObsFrommp4(videoPath):
        """
        Converts a .mp4 file to and numpy array of the observations for a demonstration.

        Returns
        -------
        An array of observations each frame.
        """
        #open the video file
        video = cv2.VideoCapture(videoPath)
        frames = []
        ret = True

        #get each frame of the video in order and add it to a normal array
        while ret:
            ret ,currentFrame = video.read()

            if not ret:
                break

            #this is the conversion to observations from a frame
            currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2RGB)
            currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_RGB2GRAY)
            currentFrame = cv2.resize(
                currentFrame, (84, 84), interpolation=cv2.INTER_AREA
            )
            currentFrame = np.expand_dims(currentFrame, -1)
            frames.append(np.array(currentFrame))

        #then stack each set of 4 frames so that it matches the expected input
        frames = np.array(frames)
        frame_stack = 4
        obs = []
        stacked_obs = np.zeros((1, 84, 84, frame_stack), dtype=np.uint8)

        for currentFrame in range(len(frames)):
            stacked_obs[..., -frames[currentFrame].shape[-1]:] = frames[currentFrame]
            obs.append(copy.deepcopy(stacked_obs))

        #note the conversion is not totally accurate because the observations are not encoded into the mp4 file
        #perfectly by the emulator
        return np.array(obs)

def Find_all_Models(model_dir):
    """
    Finds all of the files in a directory that are just numbers if trained agents.

    Parameters
    ----------
    model_dir : str
        The path to the directory containing the demonstrators.

    Returns
    -------
    [str]
        An array of strings that are the filenames of each demonstrator found.

    """

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


def Find_all_Videos(video_dir):
    """
    Finds all .mp4 videos in a given directory that are just numbers ie video demonstrations.
    Parameters
    ----------
    video_dir

    Returns
    -------
    [str]
        An array of strings that are the filenames of each demonstration found.

    """

    checkpoints = []
    filesandDirs = listdir(video_dir)
    allFiles = []
    for i in filesandDirs:
        if isfile(join(video_dir, i)):
            allFiles.append(i)

    for file in allFiles:
        if re.match('^[0-9]+\.mp4$',file):
            checkpoints.append(file)

    return checkpoints


def generate_demos_from_checkpoints(env, agent, model_dir, demosPerModel):
    """
    Generates demonstrations from a set of demonstators in a directory.

    Parameters
    ----------
    env : gym.env
        The env the demonstrators were trained for.
    agent : comp300.LearningModel.AgentClasses.PPO2Agent
        The agent object that the demonstrators will be loaded into.
    model_dir : str
        The path to the directory containing the demonstrators.
    demosPerModel : int
        The number of demonstrations to make per model.

    Returns
    -------
    Tuple
        A tuple containing an array of the demonstrations and an array of the reward for each demo.

    """
    checkpoints = Find_all_Models(model_dir)
    print('found models: ' + str(checkpoints))


    TrajectoryObservations = [] # the set of observations for each demo
    TrajectoryTotalRewards = [] #the total reward of each demo

    #now loop over each model and load it
    for model in checkpoints:
        tf.keras.backend.clear_session()
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
    """
    Creates the input pairs and labels from a set of demonstrations and rewards.

    Parameters
    ----------
    demonstrations : numpy array
        The set of demonstrations to be used to make the input pairs.
    demo_rewards : [float]
        The set of rewards used to make the labels for each pair of trajectories.
    num_full_trajectories : int
        The number of full trajectories in the dataset.
    num_sub_trajectories : int
        The number of sub-trajectories in the dataset.
    min_snippet_length : int
        The minimum length of the sub-trajectories.
    max_snippet_length : int
        The maximum length of the sub-trajectories.

    Returns
    -------
    Tuple
        A tuple containing and array of the input pairs of trajectories and an array of the class labels for the pairs.

    """
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
    """
    Creates the test and training dataset given a set of input data and a ratio.

    Parameters
    ----------
    ratio : float
        The proportion of the demonstrations used to create the test dataset.
    demonstrations : numpy array
        The set of demonstrations used to make pairs of trajectories.
    demo_rewards : [float]
        The array of rewards for each demonstration.
    num_full_trajectories : int
        The number of full demonstrations across the test and training dataset.
    num_sub_trajectories : int
        The number of sub-demonstrations (snippets) across the test and training dataset.
    min_snippet_length : int
        The minimum length of the sub-trajectories.
    max_snippet_length : int
        The maximum length of the sub-trajectories.

    Returns
    -------
    Tuple
        A tuple containing the training data, training labels, test data and test labels.

    """
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

    training_full_traj = np.int(np.floor(ratio * num_full_trajectories) +0)
    testing_full_traj = num_full_trajectories - training_full_traj

    training_sub_traj = np.int(np.floor(ratio * num_sub_trajectories) +0)
    testing_sub_traj = num_sub_trajectories - training_sub_traj

    training_observations, training_labels = create_labels(training_demos, training_rewards, training_full_traj,
                                                           training_sub_traj, min_snippet_length, max_snippet_length)
    testing_observations, testing_labels = create_labels(testing_demos, testing_rewards, testing_full_traj,
                                                         testing_sub_traj, min_snippet_length, max_snippet_length)

    return training_observations, training_labels, testing_observations, testing_labels



def train_reward_network(rewardNetwork, optimiser, training_trajectories, training_labels, training_epochs,
                         checkpoint_dir = ""):
    """
    Train the reward network to correctly classify the training data.

    Parameters
    ----------
    rewardNetwork : comp300.LearningModel.AgentClasses.RewardNetwork
        The reward network to be trained.
    optimiser : tourch.optim
        The optimiser to use during gradient decent.
    training_trajectories : numpy array
        The array of pairs of trajectories as input data.
    training_labels : [(1,0)]
        The array of labels for the pairs of trajectories for which has higher reward.
    training_epochs : int
        The number of training epochs.
    checkpoint_dir : str
        The path to the directory to save the network after each epoch

    Returns
    -------

    """
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

            #zero out the gradient before applying to network
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


def calc_accuracy(reward_network, test_trajectories, testing_labels):
    """
    Calculate the accuracy of the network on a test dataset.

    Parameters
    ----------
    reward_network : comp300.LearningModel.AgentClasses.RewardNetwork.
        The trained reward network to be tested.
    test_trajectories : numpy array
        The array of pairs of trajectories as test input.
    testing_labels : [(1,0)]
        The array of labels for each of the pairs of trajectories.

    Returns
    -------

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(test_trajectories)):
            label = testing_labels[i]
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

def generate_demos_from_videos(video_dir):
    """
    Generates a set of demonstrations from a directory containing .mp4 videos as demonstrations.

    Parameters
    ----------
    video_dir : str
        The path to the directory containing the video demonstrations.

    Returns
    -------
    Tuple
        A tuple containing an array of the demonstrations and an array of the reward for each demo.

    """
    videos = Find_all_Videos(video_dir)
    trajectories = []
    rewards = []

    for file in videos:
        traj = getObsFrommp4(join(video_dir,file))
        traj_shaped = traj[:, 0, :, :, :]
        trajectories.append(traj_shaped)
        splitName = file.split(".")
        reward = int(splitName[0])
        rewards.append(reward)

    return trajectories, rewards

if __name__ == '__main__':
    register(id='ChessSelf-v0',
             entry_point='comp300.Chess.ChessWrapper:ChessEnv',
             max_episode_steps=1000)

    args_parser = learnRewardParser()
    args, unknown_args = args_parser.parse_known_args(sys.argv)

    env_id = args.env
    env_type = args.env_type
    trajectories = []
    rewards = []
    env = make_vec_env(env_id, env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })
    env = VecFrameStack(env, 4)


    if not args.model_dir is None:
        model_path = args.model_dir
        if env_type != "atari":
            env = make_vec_env(env_id, env_type, 1, 0,
                               flatten_dict_observations=True,
                               wrapper_kwargs={
                                   'clip_rewards': False,
                                   'episode_life': False,
                               })
        agent = PPO2Agent(env, env_type, True)

        trajectories, rewards = generate_demos_from_checkpoints(env, agent, model_path, args.demos_per_model)
    elif not args.video_dir is None:
        trajectories, rewards = generate_demos_from_videos(args.video_dir)
    elif not args.dateset_path is None:
        dataset = pickle.load(open(args.dataset_path, "rb"))
        trajectories = dataset[0]
        rewards = dataset[1]
    else:
        print("need to provide either a model directory, video directory or a dataset path to create the dataset with")
        exit(1)

    trajectories = np.array(trajectories)
    rewards = np.array(rewards)
    training_obs, training_labels, test_obs, test_labels = create_training_test_labels(0.5, trajectories, rewards,
                                                                                       args.num_full_demonstrations*2,
                                                                                       args.num_sub_demonstrations*2,
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
