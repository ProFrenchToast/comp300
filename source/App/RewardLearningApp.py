from tkinter import *
from tkinter import filedialog
from App.Utils import *
import PIL.Image, PIL.ImageTk

"""This is the gui to set up the parameters for the reward learning """
class SetupRewardLearning:
    def __init__(self, master):
        self.master = master
        master.title("Learning from demonstrations")

        self.parameter_frame = Frame(master)
        self.parameter_frame.pack(side=LEFT)

        self.demo_frame = Frame(master)
        self.demo_frame.pack(side=RIGHT)

        #first make the parameter frame becuase it is the simplest
        # set up the environment choice
        self.env_label = Label(self.parameter_frame, text="Environment: ")
        self.env_label.grid(row=0, column=0)

        self.env_variable = StringVar(self.parameter_frame)
        self.env_variable.set('')
        # not sure if this is needed: self.env_variable.trace()
        self.env_options = {'Pong', 'Breakout', 'some other shit'}

        self.env_menu = OptionMenu(self.parameter_frame, self.env_variable, *self.env_options)
        self.env_menu.grid(row=0, column=1)

        #todo: add the option of different network sizes

        # set up the number of training epochs
        self.epoch_label = Label(self.parameter_frame, text="Number of training epochs")
        self.epoch_label.grid(row=1, column=0)

        self.epoch_input = Spinbox(self.parameter_frame, from_=1, to=100)
        self.epoch_input.grid(row=1, column=1)

        # set up the min snippet size
        self.minSnipp_label = Label(self.parameter_frame, text="Min snippet length")
        self.minSnipp_label.grid(row=2, column=0)

        self.minSnipp_input = Spinbox(self.parameter_frame, from_=1, to=1000000)
        self.minSnipp_input.grid(row=2, column=1)

        # set up the max snippet size
        self.maxSnipp_label = Label(self.parameter_frame, text="Max snippet length")
        self.maxSnipp_label.grid(row=3, column=0)

        self.maxSnipp_input = Spinbox(self.parameter_frame, from_=1, to=1000000)
        self.maxSnipp_input.grid(row=3, column=1)

        # set up the number of snippets
        self.noSnipp_label = Label(self.parameter_frame, text="Number of snippets in training set")
        self.noSnipp_label.grid(row=4, column=0)

        self.noSnipp_input = Spinbox(self.parameter_frame, from_=1, to=100000000)
        self.noSnipp_input.grid(row=4, column=1)

        # setup the save directory
        self.saveDir_label = Label(self.parameter_frame, text="Save directory: ")
        self.saveDir_label.grid(row=5, column=0)

        self.saveDir_variable = ""
        self.saveDir_button = Button(self.parameter_frame, text="select folder", command=self.setSaveDir)
        self.saveDir_button.grid(row=5, column=1)

        self.saveDirDisplay_label = Label(self.parameter_frame, text="no folder selected")
        self.saveDirDisplay_label.grid(row=6, column=0, columnspan=2)

        # set up the start training button
        self.start_button = Button(self.parameter_frame, text="Start training", command=self.tryStart)
        self.start_button.grid(row=7, column=0)

        #setup the close button
        self.close_button = Button(self.parameter_frame, text="Cancel", command=self.emptyAndClose)
        self.close_button.grid(row=7, column=1)

        #Now set up the demo frame
        #set up the demos label
        self.demo_label = Label(self.demo_frame, text="Demonstrations selected:")
        self.demo_label.grid(row=0, columnspan=2)

        #set up the frame to hold the demos and play buttons
        self.list_frame =ScrollableFrame(self.demo_frame)
        self.list_frame.grid(row=1, columnspan=2)

        #setup the list box to order the demos
        self.demos = ["demo1","demo2", "demo3", "demo4"]
        self.demo_variable = Variable(master=self.list_frame.scrollable_frame, value=self.demos)

        self.demo_listBox = DragDropListbox(self.list_frame.scrollable_frame, listvariable=self.demo_variable)
        self.demo_listBox.pack(side=LEFT, fill=BOTH, expand=1)

        #set up the frame to list all of the buttons
        self.playButton_Frame = Frame(self.list_frame.scrollable_frame)
        self.playButton_Frame.pack(side=RIGHT, anchor="n")

        #now fill the frame with buttons
        self.playButton_array = []
        for i in range(len(self.demos)):
            self.playButton_array.append(Button(self.playButton_Frame, text="Play"))
            self.playButton_array[i].pack(anchor="n")

        #finally add the buttons for adding demos and clearing the demos
        self.addDemo_button = Button(self.demo_frame, text="Add demonstration", command=self.addDemo)
        self.addDemo_button.grid(row=2, column=0)

        self.clear_button = Button(self.demo_frame, text="Clear demonstrations", command=self.clearDemos)
        self.clear_button.grid(row=2, column=1)

    def setSaveDir(self):
        self.saveDir_variable = filedialog.askdirectory(initialdir="~/", title="select folder to save demos")
        self.saveDirDisplay_label.config(text="Save Dir: {}".format(self.saveDir_variable))

    def tryStart(self):
        print("trying to start training")

    def emptyAndClose(self):
        self.master.destroy()

    def addDemo(self):
        print("Adding a new demo")

    def clearDemos(self):
        print("clearing all the demos selected")

class ActiveRewardLearning:
    def __init__(self, master):
        self.master = master
        master.title("Learning from demonstrations")

        #set up the frame that will hold the video feeds of the training snippets
        self.video_frame = Frame(master)
        self.video_frame.pack()

        #now add the canvases to hold the videos
        self.video_canvas1 = Canvas(master=self.video_frame)
        self.video_canvas1.pack(side=LEFT)
        self.video1 = MyVideoCapture("/home/patrick/PycharmProjects/comp300/source/videos/Agent50MTrain.mp4")

        self.video_canvas2 = Canvas(master=self.video_frame)
        self.video_canvas2.pack(side=RIGHT)
        self.video2 = MyVideoCapture("/home/patrick/PycharmProjects/comp300/source/videos/Agent50MTrain2.mp4")

        #now add the output box
        self.output_frame = ScrollableFrame(self.master)
        self.output_frame.pack()

        self.output_variable = StringVar(self.output_frame.scrollable_frame, value="some interesting output about how "
                                                                                   "the training is going.")
        self.output_message = Message(self.output_frame.scrollable_frame, textvariable=self.output_variable)
        self.output_message.pack()

        #finally add the cancel and finish buttons
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
        #first update video one
        ret, frame = self.video1.get_frame()

        if ret:
            self.photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.video_canvas1.create_image(0, 0, image=self.photo1, anchor=tkinter.NW)

        #now update the second video
        ret, frame = self.video2.get_frame()

        if ret:
            self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.video_canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)
        self.master.after(self.delay, self.update)

    def __del__(self):
        self.video1.__del__()
        self.video2.__del__()
        self.master.destory()


if __name__ == '__main__':
    rootWindow = Tk()
    Gui = SetupRewardLearning(rootWindow)
    rootWindow.mainloop()

    secondRoot = Tk()
    Gui = ActiveRewardLearning(secondRoot)
    secondRoot.mainloop()