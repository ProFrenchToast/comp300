# COMP300: Inverse Reinforcement Learning from demonstrations

This repository contains the code used to conduct the experiments for my
final year project on inverse reinforcement learning.

Additionally it contains an GUI to let users perform there own experiments 
without the needing to know the technical details.

Finally some of the results have been included to show examples of how to run
and analyse an experiment.


## Setup

To setup this package you will first need to clone the repository and set up a virtual environment to avoid collisions with other projects.

```
git clone https://gitlab.cs.man.ac.uk/f46471pq/comp300.git
cd comp300
```
Next setup and activate the virtual environment using venv.
```
virtualenv --python=python3 venv
. ./venv/bin/activate
```
Now we need to install the required packages and install this package.
```
pip install -r requirements.txt
pip install -e baselines-master/.
pip install -e .
```

Whenever you no longer need to use the package you can deactivate by running the deactivate script.
```
deactivate
```


## Usage

This codebase is split into 3 main sections:

### App

This sections contains a GUI application that makes it easy to get started with inverse reinforcement learning. You can
open each section individually however MainApp.py has everything you will need tyo get started.
```
python -m comp300.App.MainApp
```
From here you can use the instructions on screen to create a set of demonstrations, approximate the reward function 
and use the reward function for standard reinforcement learning.

Note this code only works for atari environments since they use pixel data as the observations. If you wish to try 
other environments please try the learning model.

### LearningModel

This contains the underling code for the reward model and reinforcement learning agent. Additionally it contains the 
generic versions of the methods used in the App section and so it can be used in all environments.

- Just like with the application the first part is to generate demonstrations, this can be done by recording
mp4 videos yourself or by using another ai to generate demonstrations by using checkpointTraining.py.
```
python -m comp300.LearningModel.checkpointTraining --env=<environment name (Breakout, Pong)> --env_type=<environment type (atari, mujoco ect)>
 --checkpoint_dir=<directory to save the demonstrators>
```

- Next we use LearnReward.py to learn to approximate the reward function. This can accept 3 different data sets as 
input: a directory with AI demonstrators created using checkpointTraining.py, a directory containing a set of .mp4
videos to use as demonstrations (note the videos should be named such that those with a higher ranking are given
a larger number (100.mp4 < 194.mp4)) or a dataset of trajectories and rewards pickled into a single file.
```
python -m comp300.LearningModel.LearnReward --save_path=<where to save the trained network>
# pick the one appropriate for your demonstrations
 --env=<environment name (Breakout, Pong)> --env_type=<environment type (atari, mujoco ect)>
 --checkpoint_dir=<directory containing demonstrators>
# or
 --video_dir=<directory containing videos>
# or
 --dataset_path=<path to dataset file>
```

- Finally we can use our learned reward function to perform standard reinforcement learning using our modified 
baselines library.
```
python -m baselines.run --env=<environment name (Breakout, Pong)> --env_type=<environment type (atari, mujoco ect)>
 --alg=<the learning algorithm (ppo2)> --num_timesteps=<training time> --save_path=<where to save trained agent>
 --custom_reward=pytorch --custom_reward_path=<path to learned reward>
```

Additional programmes have been provided to help with evaluating the performance and recording data such as 
getAverageReward.py and RecordAgent.py.

### Chess

This section contains the code for the chess reinforcement learning environment.

- First we need to install Stockfish (https://stockfishchess.org/) an advanced chess engine used as the opponent.
```
sudo apt-get install stockfish
```
 
- Next it contains LearnRewardChess.py. The normal reward learning from LearningModel works for the chess environment
however it takes a very long time to create demonstrations that are able to beat the opponent. Additionally many 
humans are not very good at chess and so will struggle to create demonstrations. To solve this LearnRewardChess.py uses
stockfish to generate high quality demonstrations and uses them to learn the reward function.
```
python -m comp300.Chess.LearnRewardChess --save_path=<where to save the trained network>
```

- Finally we use ChessWrapper.py to register the new learning environment and then use it for standard reinforcement 
learning.
```
python -m comp300.Chess.ChessWrapper --env=ChessSelf-v0 --env_type=ChessWrapper
 --alg=<the learning algorithm (ppo2)> --num_timesteps=<training time> --save_path=<where to save trained agent>
 --custom_reward=pytorch --custom_reward_path=<path to learned reward>
```