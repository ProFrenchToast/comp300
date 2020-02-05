import tensorflow as tf
import numpy as np

import gym
import torch

from baselines.common.policies import build_policy
from baselines.ppo2.model import Model
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
##cloned
class PPO2Agent(object):
    def __init__(self, env, env_type, stochastic):
        ob_space = env.observation_space
        ac_space = env.action_space
        self.stochastic = stochastic

        if env_type == 'atari':
            policy = build_policy(env,'cnn')
        elif env_type == 'mujoco':
            policy = build_policy(env,'mlp')

        make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1, nbatch_train=1,
                        nsteps=1, ent_coef=0., vf_coef=0.,
                        max_grad_norm=0.)
        self.model = make_model()

    def load(self, path):
        self.model.load(path)

    def act(self, observation, reward, done):
        if self.stochastic:
            a,v,state,neglogp = self.model.step(observation)
        else:
            a = self.model.act_model.act(observation)
        return a

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
def load_basePPO_and_Display():

    seed = 0
    # set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
    model_path = "~/models/breakout-reward-RL2/breakout_50M_ppo2"
    env_id = 'BreakoutNoFrameskip-v4'
    env_type = 'atari'
    record = False

    env = make_vec_env(env_id, env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })
    if record:
        env = VecVideoRecorder(env, './videos/', lambda steps: True, 2000000)

    env = VecFrameStack(env, 4)

    from LearningModel.AgentClasses import PPO2Agent as realAgent
    agent = realAgent(env, env_type, True)
    agent.load(model_path)

    for i_episode in range(1):
        observation = env.reset()
        reward = 0
        totalReward = 0
        done = False
        t = 0
        while True:
            t = t+1
            env.render()
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                print("Episode finished after {} timesteps with total reward:{}".format(t + 1, totalReward))
                break
    env.close()
    env.venv.close()



load_basePPO_and_Display()