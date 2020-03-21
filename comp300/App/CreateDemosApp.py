import threading
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Progressbar
from comp300.App.Utils import getAvailableEnvs, makeDemoFromAgent
from os import path
import subprocess
import pickle



class CreateDemosGUI:
    """
    This is the gui window for creating demonstrations and demonstrators that can be used for learning the reward
    function.
    """
    def __init__(self, master):
        """
        The constructor method that initialises all of the widgets for the window.

        Parameters
        ----------
        master : TK
            The master window to render this window inside of usually TK()
        """
        self.master = master
        master.title("Creating demonstrations")

        #set up the environment choice
        self.env_label = Label(master, text="Environment: ", highlightthickness=5)
        self.env_label.grid(row=0, column=0, sticky=W)

        self.env_variable = StringVar(master)
        self.env_variable.set('')
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
        """
        Lets the user set the save directory.

        Opens a select directory dialog and saves the selected directory & updates the label.

        Returns
        -------

        """
        self.saveDir_variable = filedialog.askdirectory(initialdir="~/", title="select folder to save demos")
        self.saveDirDisplay_label.config(text="Save Dir: {}".format(self.saveDir_variable))

    def setLogDir(self):
        """
        Lets the user set the log directory.

        Opens a select directory dialog and saves the selected directory & updates the label.

        Returns
        -------

        """
        self.logDir_variable = filedialog.askdirectory(initialdir="~/", title="select folder to log demos")
        self.logDirDisplay_label.config(text="Log Dir: {}".format(self.logDir_variable))

    def cancel(self):
        """
        Closes the window containing this GUI.

        Returns
        -------

        """
        self.master.destroy()

    def tryRun(self):
        """
        Reads all of the users input and checks that each input is valid.

        Returns
        -------

        """
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

        if valid:
            self.run(self.env_variable.get(), steps, numDemos, self.saveDir_variable, self.logDir_variable)

    def run(self, envName, stepsPerDemo, numDemos, saveDir, logDir=""):
        """
        Takes the training parameters and creates demonstrations.

        Uses the given parameters to open a new thread to do the training in to provide progress checks.

        Parameters
        ----------
        envName : str
            The case sensitive id of the environment such as BreakoutNoFrameskip-v4.
        stepsPerDemo : int
            The number of training steps between each demonstration that is created.
        numDemos : int
            The number of demonstrations to be created.
        saveDir : str
            The full path to the directory to save the demonstrations.
        logDir : str
            The full path to the directory to save the training logs.

        Returns
        -------

        """
        newWindow = Toplevel(self.master)
        gui = ProgressWindow(newWindow)
        # start training on a seperate thread
        self.training_thread = threading.Thread(target=gui.run,
                                                args=(envName, stepsPerDemo, numDemos, saveDir, logDir))
        self.training_thread.start()

    def load_config(self, filename):
        """
        Load a pickled dictionary containing preset values for the user input.

        Parameters
        ----------
        filename : str
            The full path to the configuration file.

        Returns
        -------

        """
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
    """
    This is a GUI window used to actually run the training and creation of demonstrations.
    """
    def __init__(self, master):
        """
        The constructor method that initialises all of the widgets to display the progress.

        Parameters
        ----------
        master : TK
            The window that contains this GUI.
        """
        self.master = master
        master.title("Creating demonstrations")

        self.desc = Label(master, text='Creating Demos......', highlightthickness=10)
        self.desc.pack()

        self.progress = Progressbar(master, orient = HORIZONTAL, length = 200, mode='determinate')
        self.progress.pack()


    def run(self, envName, stepsPerDemo, numDemos, saveDir, logDir=""):
        """
        Creates a set of demonstrations in a given directory.

        Parameters
        ----------
        envName : str
            The case sensitive id of the environment such as BreakoutNoFrameskip-v4.
        stepsPerDemo : int
            The number of training steps between each demonstration that is created.
        numDemos : int
            The number of demonstrations to be created.
        saveDir : str
            The full path to the directory to save the demonstrations.
        logDir : str
            The full path to the directory to save the training logs.

        Returns
        -------

        """
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
