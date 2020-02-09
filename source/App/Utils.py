import tkinter
from tkinter import ttk
import cv2
import gym
import re
import tensorflow as tf
import numpy as np
import torch
import pickle
import atari_py
import copy

from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

#This code was created by Moshe in resopnse to the question https://stackoverflow.com/questions/14459993/tkinter-listbox-drag-and-drop-with-python
class DragDropListbox(tkinter.Listbox):
    """ A Tkinter listbox with drag'n'drop reordering of entries. """
    def __init__(self, master, **kw):
        kw['selectmode'] = tkinter.SINGLE
        tkinter.Listbox.__init__(self, master, kw)
        self.bind('<Button-1>', self.setCurrent)
        self.bind('<B1-Motion>', self.shiftSelection)
        self.curIndex = None

    def setCurrent(self, event):
        self.curIndex = self.nearest(event.y)

    def shiftSelection(self, event):
        i = self.nearest(event.y)
        if i < self.curIndex:
            x = self.get(i)
            self.delete(i)
            self.insert(i+1, x)
            self.curIndex = i
        elif i > self.curIndex:
            x = self.get(i)
            self.delete(i)
            self.insert(i-1, x)
            self.curIndex = i

#this code was made by Jose Salvatierra from https://blog.tecladocode.com/tkinter-scrollable-frames/ as part of a tutorial
#on tkinter
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tkinter.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

#code code was developed by paul from https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
#as part of a tutorial on how to display videos from opencv in a tkinter window
class MyVideoCapture:
     def __init__(self, video_source=0):
         # Open the video source
         self.vid = cv2.VideoCapture(video_source)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)

         # Get video source width and height
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

     def get_frame(self):
         if self.vid.isOpened():
             ret, frame = self.vid.read()
             if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             else:
                 return (ret, None)
         else:
             return (False, None)

     # Release the video source when the object is destroyed
     def __del__(self):
         if self.vid.isOpened():
             self.vid.release()

class DemoObsAndVideo:
    def __init__(self, fileName):
        self.fileName = fileName
        if re.search(".mp4$", fileName):
            print("converting mp4 observations from {}".format(fileName))
            self.video = MyVideoCapture(fileName)
            self.obs = self.getObsFrommp4()
        elif re.search(".obs$", fileName):
            print("loading observations from {} directly".format(fileName))
            self.obs = np.array(pickle.load(open(fileName, "rb"))) #this is at users own risk
            self.video = Array2Video(self.obs)
        else:
            raise ValueError("Error can only load .mp4 and .obs files {} is neither".format(fileName))


    def getObsFrommp4(videoPath):
        #open the video file
        video = cv2.VideoCapture(videoPath)
        frames = []
        ret = True

        #get each frame of the video in order and add it to a normal array
        while ret:
            ret ,currentFrame = video.read()

            if not ret:
                break

            #this is the conversion to observations from a frame
            currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2RGB)
            currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_RGB2GRAY)
            currentFrame = cv2.resize(
                currentFrame, (84, 84), interpolation=cv2.INTER_AREA
            )
            currentFrame = np.expand_dims(currentFrame, -1)
            frames.append(currentFrame)

        #then stack each set of 4 frames so that it matches the expected input
        frame_stack = 4
        obs = []
        stacked_obs = np.zeros((1, 84, 84, 4), dtype=np.uint8)

        for currentFrame in range(len(frames)):
            stacked_obs[..., -frames[currentFrame].shape[-1]:] = frames[currentFrame]
            obs.append(copy.deepcopy(stacked_obs))

        #fuck trying to convert the mp4 array to the observations because the pixels arent aligned
        #obs_path = videoPath.split(".")[0] + ".obs"
        #obs = pickle.load(obs_path)

        #note the conversion is not totally accurate because the observations are not encoded into the mp4 file correctly
        return obs

    def play(self):
        ret = True
        i = 0
        cv2.namedWindow(self.fileName, cv2.WINDOW_AUTOSIZE)
        while ret:
            ret, frame = self.video.get_frame()

            if not ret:
                break

            cv2.imshow(self.fileName, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow(self.fileName)
        return

    def __str__(self):
        return self.fileName

class Array2Video:
    def __init__(self, obs):
        self.frames = []
        for i in range(len(obs)):
            #get the current frame
            currentFrame = obs[i, 0, :, :, 3]
            #resize it and change to rgb to display
            currentFrame = cv2.resize(
                currentFrame, (160, 210), interpolation=cv2.INTER_AREA
            )
            currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_GRAY2RGB)
            self.frames.append(currentFrame)

        self.currentIndex = 0
        self.ret = True

    def get_frame(self):
        if self.currentIndex >= len(self.frames):
            return False, None
        else:
            current_frame = self.frames[self.currentIndex]
            self.currentIndex += 1
            return True, current_frame


def getAvailableEnvs():
    allEnvs = gym.envs.registry.all()
    env_options = []
    for env in allEnvs:
        env_type = env.entry_point.split(':')[0].split('.')[-1]  # no idea??
        if env_type == "atari" and not re.search("ram", env.id) and not re.search("Deterministic", env.id) \
                and not re.search("v0", env.id):
            env_options.append(env.id)
    return env_options

def makeDemoFromAgent(agentPath, envName, agent=None):

    from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
    model_path = agentPath
    env_id = envName
    env_type = 'atari'
    record = True

    env = make_vec_env(env_id, env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })
    if record:
        env = VecVideoRecorder(env, "/", lambda steps: True, 20000, savefile=agentPath)

    env = VecFrameStack(env, 4)

    from LearningModel.AgentClasses import PPO2Agent as realAgent
    if agent == None:
        agent = realAgent(env, env_type, True)
    agent.load(model_path)

    for i_episode in range(1):
        observation = env.reset()
        #image = env.render(mode="rgb_array")
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fps = 60.0
        #writter = cv2.VideoWriter("{}.avi".format(agentPath), fourcc, fps, (image.shape[0], image.shape[1]))
        obsArray = [observation]
        reward = 0
        totalReward = 0
        done = False
        t = 0
        while True:
            t = t + 1
            #writter.write(env.render(mode="rgb_array")[:, :, 0])
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)
            obsArray.append(observation)
            totalReward += reward
            if done:
                #print("Episode finished after {} timesteps with total reward:{}".format(t + 1, totalReward))
                break
    env.close()
    env.venv.close()
    tf.keras.backend.clear_session()
    #writter.release()
    pickle.dump(obsArray, open(agentPath+".obs", "wb"))
    return agent
    #the agent needs to be returned so that the graph can reuse it

