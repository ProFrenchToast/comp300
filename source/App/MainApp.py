import tkinter
from tkinter import Tk, Label, Button, Frame
from App.CreateDemosApp import CreateDemosGUI
from App.RewardLearningApp import SetupRewardLearning
from App.TrainPolicyApp import SetupTrainPolicy


""" This is the top level gui class that will be used to open the other three steps"""
class MainGUI:
    def __init__(self, master):
        self.master = master
        master.title("Inverse reinforcement learning")

        self.description_label = Label(master, text="Inverse reinforcement learning is a method to learn to perform\n"
                                                    "a task only using a set of demonstrations. Please have a look\n"
                                                    "around and try out the different sections",
                                       highlightthickness=10)
        self.description_label.pack()

        self.ButtonFrame = Frame(master, highlightthickness=10)
        self.ButtonFrame.pack()
        self.demonstration_button = Button(self.ButtonFrame, text="Create Demonstrations", command=self.create_demonstration,
                                           highlightthickness=5)
        self.demonstration_button.grid(row=0, column=0)

        self.rewardLearning_button = Button(self.ButtonFrame, text="Learn Reward Function", command=self.create_LearnReward,
                                            highlightthickness=5)
        self.rewardLearning_button.grid(row=0, column=1)

        self.reinforcementLearning_button = Button(self.ButtonFrame, text="Learn Policy", command=self.create_ReinforcementLearning,
                                                   highlightthickness=5)
        self.reinforcementLearning_button.grid(row=0, column=2)

        #self.evaluation_button = Button(master, text="Evaluate a trained model", command=self.create_Evaluation)

    def create_demonstration(self):
        newWindow = tkinter.Toplevel(self.master)
        gui = CreateDemosGUI(newWindow)
        gui.load_config('/home/patrick/models/fullGuiTest/createDemos.config')

    def create_LearnReward(self):
        newWindow= tkinter.Toplevel(self.master)
        gui = SetupRewardLearning(newWindow)
        gui.loadConfig('/home/patrick/models/fullGuiTest/learnReward.config')

    def create_ReinforcementLearning(self):
        newWindow = tkinter.Toplevel(self.master)
        gui = SetupTrainPolicy(newWindow)
        gui.loadConfig('/home/patrick/models/fullGuiTest/trainPolicy.config')

    #def create_Evaluation(self):
        #print("creating evaluation of existing model")

if __name__ == '__main__':
    rootWindow = Tk()
    Gui = MainGUI(rootWindow)
    rootWindow.mainloop()
