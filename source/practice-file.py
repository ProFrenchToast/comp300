import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch

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
#* actually load a pretrained agent and then render 10 runs of the env
def train_basePPO():
    from baselines.common.policies import build_policy
    from baselines.ppo2.model import Model

    #tensor flow stuff i dont understand
    graph = tf.Graph()
    config = tf.ConfigProto(device_count={'GPU':0})

    session = tf.Session(graph=graph, config = config)
    with graph.as_default():
        with session.as_default():
            #make the env and the build policy and the input and output spaces
            env = gym.make('BreakoutNoFrameskip-v0')
            policy = build_policy(env, 'cnn')
            observation_space = env.observation_space
            action_space = env.action_space

            #now make the method to build the network
            make_model = lambda : Model(policy=policy, ob_space=observation_space, ac_space=action_space, nbatch_act=1,
                                        nbatch_train=1, nsteps=1, ent_coef=0, vf_coef=0, max_grad_norm=0)
            #make and learn the model
            model = make_model()

    model_path = "~/Downloads/03600"
    model.load(model_path)

    for i_episode in range(5):
        observation = env.reset()
        reward = 0
        done = False
        t = 0
        while True:
            t = t+1
            env.render()
            action, _, _, _ = model.act_model.step(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()




    #try it out and render
# Todo: make a cnn that can predict the ranking of demos



train_basePPO()