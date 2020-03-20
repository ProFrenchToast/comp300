import tkinter as tk
from comp300.App.Utils import ScrollableFrame, DemoObsAndVideo
import pickle

env = 'Breakout-v4'
steps_per_demo = 25
num_demos = 4
save_dir = '/home/patrick/models/fullGuiTest'
create_demos_config = {
    'env':env,
    'steps_per_demo':steps_per_demo,
    'num_demos':num_demos,
    'save_dir':save_dir
}
pickle.dump(create_demos_config, open('/home/patrick/models/fullGuiTest/createDemos.config', "wb"))

root = tk.Tk()

frame = ScrollableFrame(root)

for i in range(50):
    tk.Button(frame.scrollable_frame, text="Sample scrolling label").pack(fill=tk.BOTH, expand=True)

frame.pack(fill=tk.BOTH, expand=True)
root.mainloop()

training_epochs = 1
min_snippet = 50
max_snippet = 100
no_snippets = 10
save_dir = '/home/patrick/models/fullGuiTest/learnedRewardDemo.params'
demos = ['/home/patrick/models/fullGuiTest/25.mp4',
         '/home/patrick/models/fullGuiTest/50.mp4',
         '/home/patrick/models/fullGuiTest/75.obs',
         '/home/patrick/models/fullGuiTest/100.obs']

learn_reward_config = {
    'training_epochs':training_epochs,
    'min_snippet':min_snippet,
    'max_snippet':max_snippet,
    'no_snippets':no_snippets,
    'save_dir':save_dir,
    'demos':demos
}

pickle.dump(learn_reward_config, open('/home/patrick/models/fullGuiTest/learnReward.config', "wb"))

env = 'Breakout-v4'
learned_reward = '/home/patrick/models/breakout-reward/fullTest.params'
training_steps = 50000000 #50M
save_dir = '/home/patrick/models/fullGuiTest/agent50MDemo'
train_policy_config = {
    'env':env,
    'learned_reward':learned_reward,
    'training_steps':training_steps,
    'save_dir':save_dir
}

pickle.dump(train_policy_config, open('/home/patrick/models/fullGuiTest/trainPolicy.config', "wb"))


alg = 'ppo2'
env = 'BreakoutNoFrameskip-v4'
num_timesteps = 50000000
save_path = '/home/patrick/models/fullGuiTest/agent50MDemo'
load_path = '/home/patrick/logs/Agent/IRLTest/checkpoints/100000'
custom_reward_path = '/home/patrick/models/fullGuiTest/fullTest.params'
intermediate_policy_config = {
    'alg':alg,
    'env':env,
    'num_timesteps':num_timesteps,
    'save_path':save_path,
    'load_path':load_path,
    'custom_reward_path':custom_reward_path
}

pickle.dump(intermediate_policy_config, open('/home/patrick/models/fullGuiTest/intermediatePolicy.config', "wb"))

alg = 'ppo2'
env = 'BreakoutNoFrameskip-v4'
num_timesteps = 50000000
save_path = '/home/patrick/models/fullGuiTest/agent50MDemo'
load_path = '/home/patrick/models/breakout-reward-RL/breakout_50M_ppo2'
custom_reward_path = '/home/patrick/models/fullGuiTest/fullTest.params'
full_policy_config = {
    'alg':alg,
    'env':env,
    'num_timesteps':num_timesteps,
    'save_path':save_path,
    'load_path':load_path,
    'custom_reward_path':custom_reward_path
}

pickle.dump(full_policy_config, open('/home/patrick/models/fullGuiTest/fullPolicy.config', "wb"))

'''
from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env import VecFrameStack
from LearningModel.AgentClasses import PPO2Agent
from LearningModel.LearnReward import Find_all_Models
from os.path import join
import tensorflow as tf
model_dir = "/home/patrick/models/BreakoutNoFrameskip-v4-demonstrator"
env_id = 'BreakoutNoFrameskip-v4'
env_type = 'atari'

env = make_vec_env(env_id, env_type, 1, 0,
                   wrapper_kwargs={
                       'clip_rewards': False,
                       'episode_life': False,
                   })
env = VecFrameStack(env, 4)
agent = PPO2Agent(env, 'atari', True)
#agent.load("/home/patrick/Downloads/breakout_25/00001")

checkpoints = Find_all_Models(model_dir)
print('found models: ' + str(checkpoints))


TrajectoryObservations = [] # the set of observations for each demo
TrajectoryTotalRewards = [] #the total reward of each demo

#now loop over each model and load it
for model in checkpoints:

    model_path = join(model_dir, model)
    agent.load(model_path)

    for demo in range(1):
        observations = []
        totalReward = 0

        currentReward = 0
        currentObservation = env.reset()
        #the shape of the observations is (1,84,84, 4) so take only the first slice to remove the first dimension
        shapedObservations = currentObservation[0]
        timeSteps = 0
        done = False

        #run the demo
        while True:
            observations.append(shapedObservations)
            totalReward += currentReward

            action = agent.act(currentObservation,  currentReward, done)
            currentObservation, currentReward, done, info = env.step(action)
            shapedObservations = currentObservation[0]
            timeSteps += 1

            if done:
                observations.append(shapedObservations)
                totalReward += currentReward
                print("generated demo from model at {}, demo length: {}, total reward: {}".
                      format(model_path, timeSteps, totalReward))
                break

        #save the results
        #TrajectoryObservations.append(observations)
        #TrajectoryTotalRewards.append(totalReward)
        tf.keras.backend.clear_session()
        demonstration = (observations, totalReward)
        pickle.dump(demonstration, open(model_path+".obs", "wb"))

print("finished trajectory generation, Total demos: {}, min reward: {}, max reward: {}".format(
    len(TrajectoryObservations), min(TrajectoryTotalRewards), max(TrajectoryTotalRewards)))
'''