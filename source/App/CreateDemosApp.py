from tkinter import *
from tkinter import filedialog
from App.Utils import getAvailableEnvs
from os import path

"""This is the gui for creating demonstrations and demonstrators for the learning"""
class CreateDemosGUI:
    def __init__(self, master):
        self.master = master
        master.title("Creating demonstrations")

        #set up the environment choice
        self.env_label = Label(master, text="Environment: ")
        self.env_label.grid(row=0, column=0)

        self.env_variable = StringVar(master)
        self.env_variable.set('')
        # not sure if this is needed: self.env_variable.trace()
        self.env_options = getAvailableEnvs()

        self.env_menu = OptionMenu(master, self.env_variable, *self.env_options)
        self.env_menu.grid(row=0, column=1)

        #set up the steps between demos
        self.steps_label = Label(master, text="Training steps between demos: ")
        self.steps_label.grid(row=1, column=0)

        self.steps_input = Spinbox(master, from_=1, to=1000000)
        self.steps_input.grid(row=1, column=1)

        #setup the number of demos needed
        self.numDemos_label = Label(master, text="Number of demos: ")
        self.numDemos_label.grid(row=2, column=0)

        self.numDemos_input = Spinbox(master, from_=1, to=100)
        self.numDemos_input.grid(row=2, column=1)

        #setup the save directory
        self.saveDir_label = Label(master, text="Save directory: ")
        self.saveDir_label.grid(row=3, column=0)

        self.saveDir_variable = ""
        self.saveDir_button = Button(master, text="select folder", command=self.setSaveDir)
        self.saveDir_button.grid(row=3, column=1)

        self.saveDirDisplay_label = Label(master, text="no folder selected")
        self.saveDirDisplay_label.grid(row=4, column=0, columnspan=2)

        # setup the log directory
        self.logDir_label = Label(master, text="Log directory: ")
        self.logDir_label.grid(row=5, column=0)

        self.logDir_variable = ""
        self.logDir_button = Button(master, text="select folder", command=self.setLogDir)
        self.logDir_button.grid(row=5, column=1)

        self.logDirDisplay_label = Label(master, text="no folder selected")
        self.logDirDisplay_label.grid(row=6, column=0, columnspan=2)

        #set up the cancel and run buttons
        self.cancel_button = Button(master, text="Cancel", command=self.cancel)
        self.cancel_button.grid(row=7, column=0)

        self.run_button = Button(master, text="Run", command=self.tryRun)
        self.run_button.grid(row=7, column=1)

    def setSaveDir(self):
        self.saveDir_variable = filedialog.askdirectory(initialdir="~/", title="select folder to save demos")
        self.saveDirDisplay_label.config(text="Save Dir: {}".format(self.saveDir_variable))

    def setLogDir(self):
        self.logDir_variable = filedialog.askdirectory(initialdir="~/", title="select folder to log demos")
        self.logDirDisplay_label.config(text="Log Dir: {}".format(self.logDir_variable))

    def cancel(self):
        self.master.destroy()

    def tryRun(self):
        #first try to gather the inputs from the user and check they are all valid
        valid = True

        stepsStr = self.steps_input.get()
        numDemosStr = self.numDemos_input.get()
        try:
            steps = int(stepsStr)
            numDemos = int(numDemosStr)
        except ValueError:
            print("Error need integer inputs")
            valid = False

        if not path.exists(self.saveDir_variable):
            valid = False
            print("Error save directory is is unvalid")

        if self.logDir_variable != "" and not path.exists(self.logDir_variable):
            valid = False
            print("Error log directory is invalid")




if __name__ == '__main__':
    rootWindow = Tk()
    Gui = CreateDemosGUI(rootWindow)
    rootWindow.mainloop()
