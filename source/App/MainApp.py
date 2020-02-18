import tkinter
from tkinter import Tk, Label, Button
from App.CreateDemosApp import CreateDemosGUI
from App.RewardLearningApp import SetupRewardLearning
from App.TrainPolicyApp import SetupTrainPolicy


""" This is the top level gui class that will be used to open the other three steps"""
class MainGUI:
    def __init__(self, master):
        self.master = master
        master.title("Inverse reinforcement learning")

        self.description_label = Label(master, text="inverse reinforcement learning is a method to learn to perform\n"
                                                    "a task only using a set of demonstrations. please have a look\n"
                                                    "around and try out the different sections")
        self.description_label.pack()

        self.demonstration_button = Button(master, text="Create demonstrations", command=self.create_demonstration)
        self.demonstration_button.pack()

        self.rewardLearning_button = Button(master, text="Learn from Demonstrations", command=self.create_LearnReward)
        self.rewardLearning_button.pack()

        self.reinforcementLearning_button = Button(master, text="Learn to perform task", command=self.create_ReinforcementLearning)
        self.reinforcementLearning_button.pack()

        self.evaluation_button = Button(master, text="Evaluate a trained model", command=self.create_Evaluation)

    def create_demonstration(self):
        newWindow = tkinter.Toplevel(self.master)
        CreateDemosGUI(newWindow)

    def create_LearnReward(self):
        newWindow= tkinter.Toplevel(self.master)
        SetupRewardLearning(newWindow)

    def create_ReinforcementLearning(self):
        newWindow = tkinter.Toplevel(self.master)
        SetupTrainPolicy(newWindow)

    def create_Evaluation(self):
        print("creating evaluation of existing model")

if __name__ == '__main__':
    rootWindow = Tk()
    Gui = MainGUI(rootWindow)
    rootWindow.mainloop()
