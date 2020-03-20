import threading
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Progressbar
from comp300.App.Utils import getAvailableEnvs, makeDemoFromAgent
from os import path
import subprocess
import pickle


"""This is the gui for creating demonstrations and demonstrators for the learning"""
class CreateDemosGUI:
    def __init__(self, master):
        self.master = master
        master.title("Creating demonstrations")

        #set up the environment choice
        self.env_label = Label(master, text="Environment: ", highlightthickness=5)
        self.env_label.grid(row=0, column=0, sticky=W)

        self.env_variable = StringVar(master)
        self.env_variable.set('')
        # not sure if this is needed: self.env_variable.trace()
        self.env_options = getAvailableEnvs()

        self.env_menu = OptionMenu(master, self.env_variable, *self.env_options)
        self.env_menu.grid(row=0, column=1,sticky=N+S+E+W)

        #set up the steps between demos
        self.steps_label = Label(master, text="Steps per demo: ", highlightthickness=5)
        self.steps_label.grid(row=1, column=0, sticky=W)

        self.steps_input = Spinbox(master, from_=1, to=1000000, highlightthickness=5)
        self.steps_input.grid(row=1, column=1, sticky=N+S+E+W)

        #setup the number of demos needed
        self.numDemos_label = Label(master, text="Number of demos: ", highlightthickness=5)
        self.numDemos_label.grid(row=2, column=0, sticky=W)

        self.numDemos_input = Spinbox(master, from_=1, to=100, highlightthickness=5)
        self.numDemos_input.grid(row=2, column=1, sticky=N+S+E+W)

        #setup the save directory
        self.saveDir_label = Label(master, text="Save directory: ", highlightthickness=5)
        self.saveDir_label.grid(row=3, column=0, sticky=W)

        self.saveDir_variable = ""
        self.saveDir_button = Button(master, text="select folder", command=self.setSaveDir, highlightthickness=5)
        self.saveDir_button.grid(row=3, column=1, sticky=N+S+E+W)

        self.saveDirDisplay_label = Label(master, text="no folder selected", highlightthickness=5)
        self.saveDirDisplay_label.grid(row=4, column=0, columnspan=2)

        # setup the log directory
        self.logDir_label = Label(master, text="Log directory: ", highlightthickness=5)
        self.logDir_label.grid(row=5, column=0, sticky=W)

        self.logDir_variable = ""
        self.logDir_button = Button(master, text="select folder", command=self.setLogDir, highlightthickness=5)
        self.logDir_button.grid(row=5, column=1, sticky=N+S+E+W)

        self.logDirDisplay_label = Label(master, text="no folder selected", highlightthickness=5)
        self.logDirDisplay_label.grid(row=6, column=0, columnspan=2)

        #set up the cancel and run buttons
        self.cancel_button = Button(master, text="Cancel", command=self.cancel, highlightthickness=5)
        self.cancel_button.grid(row=7, column=0)

        self.run_button = Button(master, text="Run", command=self.tryRun, highlightthickness=5)
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

        if self.env_variable.get() == '':
            valid = False
            print("Please select the environment you want to generate demos for")

        self.run(self.env_variable.get(), steps, numDemos, self.saveDir_variable, self.logDir_variable)

    def run(self, envName, stepsPerDemo, numDemos, saveDir, logDir):
        newWindow = Toplevel(self.master)
        gui = ProgressWindow(newWindow)
        # start training on a seperate thread
        self.training_thread = threading.Thread(target=gui.run,
                                                args=(envName, stepsPerDemo, numDemos, saveDir, logDir))
        self.training_thread.start()

    def load_config(self, filename):
        try:
            config = pickle.load(open(filename, "rb"))
            self.env_variable.set(config.get('env'))

            self.steps_input.delete(0, "end")
            self.steps_input.insert(0, config.get('steps_per_demo'))

            self.numDemos_input.delete(0, "end")
            self.numDemos_input.insert(0, config.get('num_demos'))

            self.saveDir_variable = config.get('save_dir')
            self.saveDirDisplay_label.config(text="Save Dir: {}".format(self.saveDir_variable))
        except:
            return

class ProgressWindow():
    def __init__(self, master):
        self.master = master
        master.title("Creating demonstrations")

        self.desc = Label(master, text='Creating Demos......', highlightthickness=10)
        self.desc.pack()

        self.progress = Progressbar(master, orient = HORIZONTAL, length = 200, mode='determinate')
        self.progress.pack()


    def run(self, envName, stepsPerDemo, numDemos, saveDir, logDir):
        progress_per_demo = 100 / numDemos
        total_progress = 0
        algorithm = "ppo2"
        splitname = envName.split("-")
        fullEnvName = splitname[0] + "NoFrameskip-" + splitname[1]
        stepSize = stepsPerDemo
        # first generate the initial step
        if logDir != "":
            p = subprocess.Popen(
                "python -m baselines.run --alg={} --env={} --num_timesteps={} --save_path={}/{} --log_path={}/{}"
                    .format(algorithm, fullEnvName, stepSize, saveDir, stepSize, logDir, stepSize), shell=True)
            p.wait()
        else:
            p = subprocess.Popen(
                "python -m baselines.run --alg={} --env={} --num_timesteps={} --save_path={}/{}"
                    .format(algorithm, fullEnvName, stepSize, saveDir, stepSize), shell=True)
            p.wait()
        lastTrained = stepSize
        # with tf.Graph().as_default():
        agent = makeDemoFromAgent(saveDir + "/" + str(lastTrained), fullEnvName)
        for checkpoint in range(1, numDemos):
            total_progress += progress_per_demo
            self.progress['value'] = total_progress
            nextTrained = lastTrained + stepSize
            if logDir != "":
                p = subprocess.Popen(
                    "python -m baselines.run --alg={} --env={} --num_timesteps={} --save_path={}/{} --load_path={}/{} --log_path={}/{}"
                        .format(algorithm, fullEnvName, stepSize, saveDir, nextTrained, saveDir, lastTrained, logDir,
                                nextTrained),
                    shell=True)
                p.wait()
            else:
                p = subprocess.Popen(
                    "python -m baselines.run --alg={} --env={} --num_timesteps={} --save_path={}/{} --load_path={}/{}"
                        .format(algorithm, fullEnvName, stepSize, saveDir, nextTrained, saveDir, lastTrained),
                    shell=True)
                p.wait()

            self.desc.config(text="trained checkpoint {} to {}".format(lastTrained, nextTrained))
            # with tf.Graph().as_default():
            makeDemoFromAgent(saveDir + "/" + str(nextTrained), fullEnvName, agent=agent)
            lastTrained = nextTrained
        self.desc.config(text="finished training")
        self.master.destroy()


if __name__ == '__main__':
    rootWindow = Tk()
    Gui = CreateDemosGUI(rootWindow)
    rootWindow.mainloop()
