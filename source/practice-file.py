# TODO: make a cnn that classifies handwritting

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import baselines

def train_on_mnist():

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

    plt.figure()
    plt.imshow(x_train[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()


# Todo: make a cnn that approximates the reward for a given observation
def train_on_gym():
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

# Todo: make a base ppo agent and train on an env and save demos
def train_basePPO():
    env = gym.make('CartPole-v0')
    model = baselines.PPO2(baselines.common.policies.MlpPolicy, env)
    model.learn(total_timesteps=1000)

# Todo: make a cnn that can predict the ranking of demos



train_on_gym()