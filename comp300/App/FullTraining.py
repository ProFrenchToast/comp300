import tkinter
from comp300.App.TrainPolicyApp import ActiveTrainPolicy
import pickle

if __name__ == '__main__':
    root = tkinter.Tk()
    config = pickle.load(open('/home/patrick/models/fullGuiTest/fullPolicy.config', "rb"))
    gui = ActiveTrainPolicy(root, "something-something", 0, "", "", "", config=config)
    root.mainloop()