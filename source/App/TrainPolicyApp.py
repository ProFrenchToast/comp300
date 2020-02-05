from tkinter import *
from App.Utils import *
import PIL.Image, PIL.ImageTk
from tkinter import filedialog

class SetupTrainPolicy:
    def __init__(self, master):
        self.master = master
        master.title("Learning to complete task")

        # set up the environment choice
        self.env_label = Label(master, text="Environment: ")
        self.env_label.grid(row=0, column=0)

        self.env_variable = StringVar(master)
        self.env_variable.set('')
        # not sure if this is needed: self.env_variable.trace()
        self.env_options = {'Pong', 'Breakout', 'some other shit'}

        self.env_menu = OptionMenu(master, self.env_variable, *self.env_options)
        self.env_menu.grid(row=0, column=1)

        # setup the reward network to use
        self.reward_label = Label(master, text="Learned reward: ")
        self.reward_label.grid(row=1, column=0)

        self.reward_variable = ""
        self.reward_button = Button(master, text="select folder", command=self.setRewardNetwork)
        self.reward_button.grid(row=1, column=1)

        self.rewardDisplay_label = Label(master, text="no folder selected")
        self.rewardDisplay_label.grid(row=2, column=0, columnspan=2)

        # set up the steps the policy should be trained for
        self.steps_label = Label(master, text="Training steps between demos: ")
        self.steps_label.grid(row=3, column=0)

        self.steps_input = Spinbox(master, from_=1, to=1000000)
        self.steps_input.grid(row=3, column=1)

        # setup the save directory
        self.saveDir_label = Label(master, text="Save directory: ")
        self.saveDir_label.grid(row=4, column=0)

        self.saveDir_variable = ""
        self.saveDir_button = Button(master, text="select folder", command=self.setSaveDir)
        self.saveDir_button.grid(row=4, column=1)

        self.saveDirDisplay_label = Label(master, text="no folder selected")
        self.saveDirDisplay_label.grid(row=5, column=0, columnspan=2)

        # setup the log directory
        self.logDir_label = Label(master, text="log directory: ")
        self.logDir_label.grid(row=6, column=0)

        self.logDir_variable = ""
        self.logDir_button = Button(master, text="select folder", command=self.setLogDir)
        self.logDir_button.grid(row=6, column=1)

        self.logDirDisplay_label = Label(master, text="no folder selected")
        self.logDirDisplay_label.grid(row=7, column=0, columnspan=2)

        # set up the cancel and run buttons
        self.cancel_button = Button(master, text="Cancel", command=self.cancel)
        self.cancel_button.grid(row=8, column=0)

        self.run_button = Button(master, text="Run", command=self.tryRun)
        self.run_button.grid(row=8, column=1)

    def setRewardNetwork(self):
        self.reward_variable = filedialog.askopenfilename(initialdir="~/", title="select the learned reward")
        self.rewardDisplay_label.config(text="Selected reward: {}".format(self.reward_variable))

    def setSaveDir(self):
        self.saveDir_variable = filedialog.askdirectory(initialdir="~/", title="select folder to save demos")
        self.saveDirDisplay_label.config(text="Save Dir: {}".format(self.saveDir_variable))

    def setLogDir(self):
        self.logDir_variable = filedialog.askdirectory(initialdir="~/", title="select folder to log demos")
        self.logDirDisplay_label.config(text="Log Dir: {}".format(self.logDir_variable))

    def cancel(self):
        self.master.destroy()

    def tryRun(self):
        print("test that variables are consistent and then run")

class ActiveTrainPolicy:
    def __init__(self, master):
        self.master =  master
        master.title("Learning to complete task")

        #set up the canvas for the video
        self.video_canvas = Canvas(master)
        self.video_canvas.pack()

        #not sure how i will even do this because it doesnt produce .mp4 files. I guess probably capture the new window or
        #take the obs directly and create images.
        self.video_stream = MyVideoCapture("/home/patrick/PycharmProjects/comp300/source/videos/Agent50MTrain.mp4")

        # now add the output box
        self.output_frame = ScrollableFrame(self.master)
        self.output_frame.pack()

        self.output_variable = StringVar(self.output_frame.scrollable_frame, value="some interesting output about how "
                                                                                   "the training is going.")
        self.output_message = Message(self.output_frame.scrollable_frame, textvariable=self.output_variable)
        self.output_message.pack()

        # finally add the cancel and finish buttons
        self.button_frame = Frame(self.master)
        self.button_frame.pack()

        self.cancel_button = Button(self.button_frame, text="Cancel", command=self.__del__)
        self.cancel_button.pack(side=LEFT)

        self.finish_button = Button(self.button_frame, text="Finish", state=DISABLED)
        self.finish_button.pack(side=RIGHT)

        # add the update method to the main window loop
        self.delay = 15
        self.update()

    def update(self):
        ret, frame = self.video_stream.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.video_canvas.create_image(0,0, image=self.photo, anchor="nw")

        self.master.after(self.delay, self.update)

    def __del__(self):
        self.video_stream.__del__()
        self.master.destroy()

if __name__ == '__main__':
    root = Tk()
    gui = SetupTrainPolicy(root)
    root.mainloop()

    root2 = Tk()
    gui2 = ActiveTrainPolicy(root2)
    root2.mainloop()