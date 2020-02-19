import tkinter as tk
from App.Utils import ScrollableFrame, DemoObsAndVideo
import pickle

root = tk.Tk()

frame = ScrollableFrame(root)

for i in range(50):
    tk.Button(frame.scrollable_frame, text="Sample scrolling label").pack(fill=tk.BOTH, expand=True)

frame.pack(fill=tk.BOTH, expand=True)
root.mainloop()

training_epochs = 4
min_snippet = 50
max_snippet = 150
no_snippets = 10
save_dir = '/home/patrick/models/fullGuiTest/learnedReward.params'
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
save_dir = '/home/patrick/models/fullGuiTest/agent50M'
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
save_path = '/home/patrick/models/fullGuiTest/agent50M'
load_path = '/home/patrick/logs/'
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
save_path = '/home/patrick/models/fullGuiTest/agent50M'
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