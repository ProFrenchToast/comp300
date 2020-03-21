from tkinter import *
import tkinter.scrolledtext as ScrolledText
from tkinter import filedialog
from comp300.App.Utils import *
import PIL.Image, PIL.ImageTk
from os import path
import re
import threading

class SetupRewardLearning:
    """This is the gui to set up the parameters for the reward learning """
    def __init__(self, master):
        """
        The constructor that initialises the widgets in the window.

        Parameters
        ----------
        master : TK
            The window that contains the gui.
        """
        self.master = master
        master.title("Learning from demonstrations")

        self.leftFrame = Frame(master)
        self.leftFrame.pack(fill=Y, expand=TRUE, side=LEFT)

        self.param_title = Label(self.leftFrame, text="Parameters:", highlightthickness=5)
        self.param_title.pack(anchor=N)

        self.parameter_frame = Frame(self.leftFrame)
        self.parameter_frame.pack(anchor=CENTER)

        self.demo_frame = Frame(master)
        self.demo_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        #first make the parameter frame becuase it is the simplest

        # set up the number of training epochs
        self.epoch_label = Label(self.parameter_frame, text="Number of training epochs: ", highlightthickness=5)
        self.epoch_label.grid(row=1, column=0, sticky=W)

        self.epoch_input = Spinbox(self.parameter_frame, from_=1, to=100, highlightthickness=5)
        self.epoch_input.grid(row=1, column=1)

        # set up the min snippet size
        self.minSnipp_label = Label(self.parameter_frame, text="Min snippet length: ", highlightthickness=5)
        self.minSnipp_label.grid(row=2, column=0, sticky=W)

        self.minSnipp_input = Spinbox(self.parameter_frame, from_=1, to=1000000, highlightthickness=5)
        self.minSnipp_input.grid(row=2, column=1)

        # set up the max snippet size
        self.maxSnipp_label = Label(self.parameter_frame, text="Max snippet length: ", highlightthickness=5)
        self.maxSnipp_label.grid(row=3, column=0, sticky=W)

        self.maxSnipp_input = Spinbox(self.parameter_frame, from_=1, to=1000000, highlightthickness=5)
        self.maxSnipp_input.grid(row=3, column=1)

        # set up the number of snippets
        self.noSnipp_label = Label(self.parameter_frame, text="Training set size: ", highlightthickness=5)
        self.noSnipp_label.grid(row=4, column=0, sticky=W)

        self.noSnipp_input = Spinbox(self.parameter_frame, from_=1, to=100000000, highlightthickness=5)
        self.noSnipp_input.grid(row=4, column=1)

        # setup the save directory
        self.saveDir_label = Label(self.parameter_frame, text="Save directory: ", highlightthickness=5)
        self.saveDir_label.grid(row=5, column=0, sticky=W)

        self.saveDir_variable = ""
        self.saveDir_button = Button(self.parameter_frame, text="select folder", command=self.setSaveDir, highlightthickness=5)
        self.saveDir_button.grid(row=5, column=1)

        self.saveDirDisplay_label = Label(self.parameter_frame, text="no folder selected", highlightthickness=5)
        self.saveDirDisplay_label.grid(row=6, column=0, columnspan=2)

        # set up the start training button
        self.start_button = Button(self.parameter_frame, text="Start training", command=self.tryStart, highlightthickness=5)
        self.start_button.grid(row=7, column=1)

        #setup the close button
        self.close_button = Button(self.parameter_frame, text="Cancel", command=self.emptyAndClose, highlightthickness=5)
        self.close_button.grid(row=7, column=0)

        #Now set up the demo frame
        #set up the demos label
        self.demo_label = Label(self.demo_frame, text="Demonstrations selected:", highlightthickness=5)
        self.demo_label.pack()

        #set up the frame to hold the demos and play buttons
        self.list_frame =ScrollableFrame(self.demo_frame, highlightthickness=5)
        self.list_frame.pack(fill=BOTH, expand=True)

        #setup the list box to order the demos
        self.demos = []
        self.demo_variable = Variable(master=self.list_frame.scrollable_frame, value=self.demos)

        self.demo_listBox = DragDropListbox(self.list_frame.scrollable_frame, listvariable=self.demo_variable)
        self.demo_listBox.pack(side=LEFT, fill=BOTH, expand=True)

        #set up the frame to list all of the buttons
        self.playButton_Frame = Frame(self.list_frame.scrollable_frame)
        self.playButton_Frame.pack(side=RIGHT, anchor="n")

        #now fill the frame with buttons
        self.playButton_array = []
        self.makeButtons()

        #finally add the buttons for adding demos and clearing the demos
        self.addDemo_button = Button(self.demo_frame, text="Add demonstration", command=self.addDemo, highlightthickness=10)
        self.addDemo_button.pack(side=RIGHT)

        self.clear_button = Button(self.demo_frame, text="Clear demonstrations", command=self.clearDemos, highlightthickness=10)
        self.clear_button.pack(side=LEFT)

    def makeButtons(self):
        """
        Remakes the set of play buttons to match the current set of demos.

        Returns
        -------

        """
        #first remove any existing buttons
        for i in reversed(range(len(self.playButton_array))):
            self.playButton_array[i].pack_forget()
            del self.playButton_array[i]

        #then get the set of demonstrations
        listOfDemos = self.demo_variable.get()
        demosStr = []
        for demo in self.demos:
            demosStr.append(str(demo))
        #then make a button for each demo
        self.pixel = PhotoImage(width=1, height=1)  #needed to be able to resize the buttons to exact pixel values
        for i in range(len(self.demos)):
            filename = listOfDemos[i]
            indexInDemos = demosStr.index(filename)
            self.playButton_array.append(Button(self.playButton_Frame, text="Play",
                                                command=self.demos[indexInDemos].play,
                                                image=self.pixel, compound='c',
                                                height=5, width=30))
            self.playButton_array[i].pack()


    def setSaveDir(self):
        """
        Lets the user set the save directory.

        Opens a select directory dialog and saves the selected directory & updates the label.

        Returns
        -------

        """
        self.saveDir_variable = filedialog.asksaveasfilename(initialdir="~/", title="Where to save learned reward",
                                                             initialfile="learnedReward")
        self.saveDirDisplay_label.config(text="Save Dir: {}".format(self.saveDir_variable))

    def tryStart(self):
        """
        Reads all of the users input and checks that each input is valid.

        Returns
        -------

        """
        valid = True

        epochStr = self.epoch_input.get()
        min_snippetStr = self.minSnipp_input.get()
        max_snippetStr = self.maxSnipp_input.get()
        training_sizeStr = self.noSnipp_input.get()

        try:
            epochs = int(epochStr)
            min_snippet = int(min_snippetStr)
            max_snippet = int(max_snippetStr)
            training_size = int(training_sizeStr)

            if min_snippet > max_snippet:
                valid = False
                print("Error the minimum snippet size needs to be smaller or equal to the maximum  snippet size")
        except ValueError:
            print("Error the values need to be integers")
            valid = False

        if self.saveDir_variable == "":
            valid = False
            print("Error need to select the save directory")

        if len(self.demos) < 2:
            valid = False
            print("Error needs at least 2 demos to perform learning")
        else:
            #create the ordered demos
            orderedDemos = []
            listOfDemos = self.demo_variable.get()
            demosStr = []
            for demo in self.demos:
                demosStr.append(str(demo))

            for filename in listOfDemos:
                indexInDemos = demosStr.index(filename)
                obs = self.demos[indexInDemos]
                orderedDemos.append(obs)

        if valid:
            print("trying to start training")
            #now make a new window
            self.activeWindow = tkinter.Toplevel(self.master)
            ActiveRewardLearning(self.activeWindow, orderedDemos, epochs, min_snippet, max_snippet,
                                 training_size, self.saveDir_variable)

    def emptyAndClose(self):
        """
        Clears the current set of demos and closes the window.

        Returns
        -------

        """
        for demo in self.demos:
            del demo
        self.master.destroy()

    def addDemo(self):
        """
        Lets the user add a new demonstration to the current set.

        Opens a select file dialog and lets the user select from .mp4 or .obs files. the selected file is then
        used to create a demonstration and is added ot the set of current demonstrations.

        Returns
        -------

        """
        newDemoPath = filedialog.askopenfilename(initialdir="~/", title="Select the demo to open",
                                                 filetypes=(("mp4 video files","*.mp4"),
                                                            ("raw observation files","*.obs"),
                                                            ("all files","*.*")))

        #first check file exisits
        if not path.exists(newDemoPath):
            print("Error that file does not exist")
            return
        else:
            #create a demo from the file and save it
            newDemo = DemoObsAndVideo(newDemoPath)
            self.demos.append(newDemo)
            self.demo_variable.set(self.demos)
            #remake the buttons to reflect the new addition
            self.makeButtons()

    def clearDemos(self):
        """
        Clears all of the current demos and updates the buttons.

        Returns
        -------

        """
        print("clearing all demos")
        for demo in self.demos:
            del demo

        self.demos = []
        self.demo_variable.set(self.demos)
        self.makeButtons()

    def loadConfig(self, filename):
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

            self.epoch_input.delete(0, "end")
            self.epoch_input.insert(0, config.get('training_epochs'))

            self.minSnipp_input.delete(0, "end")
            self. minSnipp_input.insert(0, config.get('min_snippet'))

            self.maxSnipp_input.delete(0, "end")
            self.maxSnipp_input.insert(0, config.get('max_snippet'))

            self.noSnipp_input.delete(0, "end")
            self.noSnipp_input.insert(0, config.get('no_snippets'))

            self.saveDir_variable = config.get('save_dir')
            self.saveDirDisplay_label.config(text="Save Dir: {}".format(self.saveDir_variable))

            self.clearDemos()
            demoNames = config.get('demos')
            for name in demoNames:
                newDemo = DemoObsAndVideo(name)
                self.demos.append(newDemo)

            self.demo_variable.set(self.demos)
            self.makeButtons()
        except:
            return





class ActiveRewardLearning:
    """This is the Gui that actually runs the reward learning and trains the reward network."""
    def __init__(self, master, demos, trainingEpochs, min_snippet_size, max_snippet_size, no_snippets, save_dir):
        """
        The constructor that initialises the widgets and then starts the training thread.

        Parameters
        ----------
        master : TK
            The window to contain the gui.
        demos : [DemoObsAndVideo]
            An array of demonstrations that are ordered with the highest ranked first.
        trainingEpochs : int
            The number of epochs to train for.
        min_snippet_size : int
            The minimum size of the snippets from demonstrations.
        max_snippet_size : int
            The maximum size of the snippets from demonstrations.
        no_snippets : int
            The number of snippets to make to create the dataset.
        save_dir : str
            The path to save the trained network to.
        """
        self.master = master
        master.title("Learning from demonstrations")

        self.demos = []
        for demo in demos:
            self.demos.append(demo.obs[:, 0, :, :])
        self.demo_ranking = list(range(len(self.demos)))
        self.demo_ranking.reverse()
        self.training_epochs = trainingEpochs
        self.min_snippet_size = min_snippet_size
        self.max_snippet_size = max_snippet_size
        self.no_snippets = no_snippets
        self.save_dir = save_dir

        #set up the frame that will hold the video feeds of the training snippets
        self.video_frame = Frame(master)
        self.video_frame.pack()

        self.left_videoFrame = Frame(self.video_frame, highlightthickness=10)
        self.left_videoFrame.pack(side=LEFT, fill=X, expand=TRUE)

        self.right_videoFrame = Frame(self.video_frame, highlightthickness=10)
        self.right_videoFrame.pack(side=RIGHT, fill=X, expand=TRUE)

        #now add the canvases to hold the videos
        self.leftLabel = Label(self.left_videoFrame, text="Worse Demo:")
        self.leftLabel.pack()
        self.video_canvas1 = Canvas(master=self.left_videoFrame, width=160, height=200)
        self.video_canvas1.pack()
        self.video1 = MyVideoCapture("/home/patrick/PycharmProjects/comp300/comp300/videos/Agent50MTrain.mp4")

        self.rightLabel = Label(self.right_videoFrame, text="Better Demo:")
        self.rightLabel.pack()
        self.video_canvas2 = Canvas(master=self.right_videoFrame, width=160, height=200)
        self.video_canvas2.pack()
        self.video2 = MyVideoCapture("/home/patrick/PycharmProjects/comp300/comp300/videos/Agent50MTrain2.mp4")

        #now add the output box
        self.output_frame = Frame(self.master)
        self.output_frame.pack()

        self.output_variable = StringVar(self.output_frame, value="some interesting output about how "
                                                                                   "the training is going.")
        self.output_box = ScrolledText.ScrolledText(self.output_frame, state='disabled', font='TkFixedFont')
        self.output_box.pack()

        text_handler = TextHandler(self.output_box)
        # Logging configuration
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Add the handler to logger
        logger = logging.getLogger()
        logger.addHandler(text_handler)

        #finally add the cancel and finish buttons
        self.button_frame = Frame(self.master)
        self.button_frame.pack()

        self.cancel_button = Button(self.button_frame, text="Cancel", command=self.__del__)
        self.cancel_button.pack(side=LEFT)

        self.finish_button = Button(self.button_frame, text="Finish", state=DISABLED, command=master.destroy)
        self.finish_button.pack(side=RIGHT)

        #make the demos and rankings into numpy arrays
        self.demos = np.array(self.demos)
        self.demo_ranking = np.array(self.demo_ranking)
        #create the labels and snippets
        self.training_trajectories, self.training_labels = create_training_labels(self.demos, self.demo_ranking,
                                                                                  0, self.no_snippets,
                                                                                  self.min_snippet_size,
                                                                                  self.max_snippet_size)
        #start training on a seperate thread
        self.training_thread = threading.Thread(target=train_network, args=(self.training_trajectories, self.training_labels,
                                                                self.training_epochs, self.save_dir, self))
        self.training_thread.start()

        # add the update method to the main window loop
        self.delay = 34
        self.update()

    def update(self):
        """
        Change the current frame to the next one for each video.

        Called every 34ms and changes the current frame displayed and clears all layers of the canvas if no
        new frame is available.

        Returns
        -------

        """
        #first update video one
        ret, frame = self.video1.get_frame()

        if ret:
            self.photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.video_canvas1.create_image(0, 0, image=self.photo1, anchor=tkinter.NW)
        else:
            self.video_canvas1.delete("all")


        #now update the second video
        ret, frame = self.video2.get_frame()

        if ret:
            self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.video_canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)
        else:
            self.video_canvas2.delete("all")

        #check if the training is finished
        if not self.training_thread.isAlive():
            self.finish_button.config(state=ACTIVE)

        #rerun this after self.delay ms
        self.master.after(self.delay, self.update)

    def __del__(self):
        """
        Safely clears both videos and then closes the current window.

        Returns
        -------

        """
        self.video1.__del__()
        self.video2.__del__()
        self.master.destroy()


if __name__ == '__main__':
    rootWindow = Tk()
    Gui = SetupRewardLearning(rootWindow)
    rootWindow.mainloop()
