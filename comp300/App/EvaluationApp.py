from tkinter import *
from tkinter import filedialog
from comp300.App.Utils import getAvailableEnvs, makeDemoFromAgent
from os import path, listdir
from os.path import isfile, join
import re
import pickle

from baselines.common.vec_env import VecFrameStack
from comp300.LearningModel.AgentClasses import *
from baselines.common.cmd_util import make_vec_env
import tensorflow as tf

import matplotlib
from matplotlib.figure import Figure
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random


"""This is the gui for creating demonstrations and demonstrators for the learning"""
class EvaluationGUI:
    def __init__(self, master):
        self.master = master
        master.title("Evaluate performance")

        #first make the frame to hold the graphs of the learned reward
        self.learned_rewardFrame = Frame(master)
        self.learned_rewardFrame.pack()

        #now fill it with buttons and a canvas
        self.learned_rewardButtonFrame = Frame(self.learned_rewardFrame)
        self.learned_rewardButtonFrame.pack(side=LEFT)

        self.demos = []
        self.learnedRewards = []
        self.reward_network = RewardNetwork("")
        self.set_demosButton = Button(self.learned_rewardButtonFrame, text='Set demonstratons', command=self.setDemos)
        self.set_demosButton.grid(row=0, column=0)

        self.clear_demosButton = Button(self.learned_rewardButtonFrame, text='Clear demonstrations', command=self.clearDemos)
        self.clear_demosButton.grid(row=0, column=1)

        self.add_learned_rewardButton = Button(self.learned_rewardButtonFrame, text='Add learned reward', command=self.addLearnedReward)
        self.add_learned_rewardButton.grid(row=1, column=0)

        self.clear_learned_rewardButton = Button(self.learned_rewardButtonFrame, text='Clear learned reward', command=self.clearLearnedReward)
        self.clear_learned_rewardButton.grid(row=1, column=1)

        self.learnedRewardFigure = Figure( figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
        self.learned_rewardCanvas = FigureCanvasTkAgg(self.learnedRewardFigure ,self.learned_rewardFrame)
        #self.learned_rewardCanvas.pack(side=RIGHT)

        #second create the frame to hold the graphs of the training over time
        self.training_logsFrame = Frame(master)
        self.training_logsFrame.pack()

        #now fill the frame with the canvas and buttons
        self.training_logsButtonFrame = Frame(self.training_logsFrame)
        self.training_logsButtonFrame.pack(side=LEFT)

        self.add_training_logsButton = Button(self.training_logsButtonFrame, text='Add training logs', command=self.addTrainingLogs)
        self.add_training_logsButton.grid(row=0, column=0)

        self.clear_training_logsButton = Button(self.training_logsButtonFrame, text='Clear training logs', command=self.clearTrainingLogs)
        self.clear_training_logsButton.grid(row=0, column=1)

        self.add_trained_agentButton = Button(self.training_logsButtonFrame, text='Add trained agent', command=self.addTrainedAgent)
        self.add_trained_agentButton.grid(row=1, column=0)

        self.clear_trained_agentButton = Button(self.training_logsButtonFrame, text='Clear trained agent', command=self.clearTrainedAgent)
        self.clear_trained_agentButton.grid(row=1, column=1)

        self.training_logsCanvas = Canvas(self.training_logsFrame)
        self.training_logsCanvas.pack(side=RIGHT)

        #finally add a frame to hold the trained agent
        self.trained_agentFrame = Frame(master)
        self.trained_agentFrame.pack()
         #now add a canvas to the frame
        self.trained_agentCanvas = Canvas(self.trained_agentFrame)
        self.trained_agentCanvas.pack()

    def setDemos(self):
        print("add a set of demonstrations")
        demo_dir = filedialog.askdirectory(initialdir="~/", title="select folder of demos")
        files = self.Find_all_Models(demo_dir)
        files = random.sample(files, 10)
        for filename in files:
            self.demos.append(pickle.load(open(filename, "rb")))


    def calculateLearnedReward(self):
        print("calculating reward graph")

        ground_truth_reward = []
        for demo in self.demos:
            ground_truth_reward.append(demo[1])
            maxReward = max(ground_truth_reward)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        all_predicted_rewards = []
        for network in self.learnedRewards:
            #load the network weights
            self.reward_network.load_state_dict(torch.load(network))
            #apply the network to each demo
            predicted_rewards = []
            for demo in self.demos:
                observations = demo[0]
                observations = torch.from_numpy(np.array(observations)).float().to(device)
                reward, abs_reward = self.reward_network.predict_reward(observations)
                predicted_rewards.append(reward.tolist())
            tf.keras.backend.clear_session()
            minLearnedReward = min(predicted_rewards)
            maxLearnedReward = max(predicted_rewards - minLearnedReward)
            normalisedRewards = (predicted_rewards - minLearnedReward) * (maxReward / maxLearnedReward)
            all_predicted_rewards.append(normalisedRewards)

        #now pl;ot the data
        fig = self.learnedRewardFigure.add_subplot(111) #no idea why the 111 os needed
        fig.plot(np.arange(500), np.arange(500))

        for rewards in all_predicted_rewards:
            fig.plot(reward, ground_truth_reward)

        self.learned_rewardCanvas.draw()
        self.learned_rewardCanvas.get_tk_widget().pack()


    def Find_all_Models(self, model_dir):

        checkpoints = []
        filesandDirs = listdir(model_dir)
        allFiles = []
        for i in filesandDirs:
            if isfile(join(model_dir, i)):
                allFiles.append(i)

        for file in allFiles:
            if re.match('^[0-9]+\.obs$', file):
                checkpoints.append(join(model_dir, file))

        return checkpoints


    def clearDemos(self):
        self.demos = []
        self.learnedRewards = []

        self.calculateLearnedReward()

    def addLearnedReward(self):
        learned_reward_file = filedialog.askopenfilename(initialdir="~/", title="select the learned reward")
        self.learnedRewards.append((learned_reward_file))

        self.calculateLearnedReward()

    def clearLearnedReward(self):
        print("clear the set of learned rewards")

    def addTrainingLogs(self):
        print("add the logs of an agent training")

    def clearTrainingLogs(self):
        print("clear the agent training logs")

    def addTrainedAgent(self):
        print("add a fuly trained agent")

    def clearTrainedAgent(self):
        print("clear all of the trained agent")

if __name__ == '__main__':
    root = Tk()
    gui = EvaluationGUI(root)
    root.mainloop()