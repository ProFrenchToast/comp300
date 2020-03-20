import tensorflow as tf
import numpy as np

import gym
import torch

from baselines.common.policies import build_policy
from baselines.ppo2.model import Model
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from comp300.LearningModel.AgentClasses import *

def train_on_mnist():
    import matplotlib.pyplot as plt
    # create the dataset
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test =  x_train/255.0 , x_test/255.0

    # build the model to classify the images
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    model.fit(x_train,y_train, epochs= 3)
    model.evaluate(x_test, y_test, verbose=2)
    predictions = model.predict(x_test)

    for i in range(5):
        predictedValue = np.argmax(predictions[i])
        plt.figure()
        plt.title("Actual:" + str(y_test[i]), loc='left')
        plt.title("prediction: " + str(predictedValue))
        plt.imshow(x_test[i])
        plt.colorbar()
        plt.grid(False)
        plt.show()


def displaygym():
    #create an env
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()

#* actually load a pretrained agent and then render 10 runs of the env
