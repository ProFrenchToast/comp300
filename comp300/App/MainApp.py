import tkinter
from tkinter import Tk, Label, Button, Frame

from comp300.App.CreateDemosApp import CreateDemosGUI
from comp300.App.RewardLearningApp import SetupRewardLearning
from comp300.App.TrainPolicyApp import SetupTrainPolicy


class MainGUI:
    """ This is the top level gui class that will be used to open the other three steps"""
    def __init__(self, master):
        """
        The constructor that initialises the widgets in the window

        Parameters
        ----------
        master : TK
            The window containing the gui.
        """
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


    def create_demonstration(self):
        """
        Opens the create demos gui in a new window & loads a simple config.

        Returns
        -------

        """
        newWindow = tkinter.Toplevel(self.master)
        gui = CreateDemosGUI(newWindow)
        gui.load_config('/home/patrick/models/fullGuiTest/createDemos.config')

    def create_LearnReward(self):
        """
        Opens the learn reward gui in a new window and loads a simpel config.

        Returns
        -------

        """
        newWindow= tkinter.Toplevel(self.master)
        gui = SetupRewardLearning(newWindow)
        gui.loadConfig('/home/patrick/models/fullGuiTest/learnReward.config')

    def create_ReinforcementLearning(self):
        """
        Opens the learn reward gui in a new window and loads a simpel config.

        Returns
        -------

        """
        newWindow = tkinter.Toplevel(self.master)
        gui = SetupTrainPolicy(newWindow)
        gui.loadConfig('/home/patrick/models/fullGuiTest/trainPolicy.config')


if __name__ == '__main__':
    rootWindow = Tk()
    Gui = MainGUI(rootWindow)
    rootWindow.mainloop()
