import argparse

def arg_parser(desc):
    """
    Create an empty parser that each method will build on.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=desc)
    return parser

def checkpointParser():
    parser = arg_parser("Train an AI with standard reinforcement learning and save a copy of it after and given number "
                        "of training steps")
    parser.add_argument("--env", help="The environment ID you wish to train", type=str, default="BreakoutNoFrameSkip-v4")
    parser.add_argument("--env_type", help="The type of environment ie atari, mujoco ect", type=str, default="atari",
                        choices=["algorithmic","atari", "box2d", "classic_control", "ChessWrapper", "mujoco", "robotics", "toy_text"])
    parser.add_argument("--alg", help="The reinforcement learning algorithim you want to use", type=str, default="ppo2")
    parser.add_argument("--checkpoint_dir", help="The directory the models will be saved to", type=str, required=True)
    parser.add_argument("--step_size", help="The number of training steps between demonstrations", type=float,
                        default=1e5)
    parser.add_argument("--num_checkpoints", help="The number of checkpoints to be trained", type=int, default=20)
    parser.add_argument("--log_dir", help="The directory where the log files will be stored -note this will overwrite "
                                          "existing files in the directory", type=str)

    return parser

def getAverageParser():
    parser = arg_parser("Test the effectiveness of a trained agent in a given environment and display the results")
    parser.add_argument("--env", help="The environment ID the agent was trained for", type=str, default="BreakoutNoFrameskip-4")
    parser.add_argument("--env_type", help="The type of environment ie atari, mujoco ect", type=str, default="atari",
                        choices=["algorithmic", "atari", "box2d", "classic_control", "ChessWrapper", "mujoco",
                                 "robotics", "toy_text"])
    parser.add_argument("--model_dir", help="The directory of the model you wish to test", type=str, required=True)
    parser.add_argument("--render", help="Changes if the environment is rendered to the screen", action='store_true', default=False)
    parser.add_argument("--num_episodes", help="The number of episodes that will be ran to calculate the results",
                        type=int, default=10)
    return parser

def learnRewardParser():
    parser = arg_parser("Learn to approximate the reward function of a set of demonstrations\n"
                        "This can be done in 1 of 3 ways:\n"
                        "  1. specify an environment to use and a directory that contains trained agents and use them\n"
                        "     to create demonstrations and rank based on ground truth reward\n"
                        "     (Use --env, --env_type, --demos_per_model and --model_dir)\n"
                        "\n"
                        "  2. specify a directory containing .mp4 videos to use as demonstrations. The videos need to\n"
                        "     have filenames that describe the reward for each demo (for example 23.mp4, 54.mp4 ect).\n"
                        "     (Use --video_dir)\n"
                        "\n"
                        "  3. specify a dataset files that contains the demonstrations and rewards together as a \n"
                        "     pickeled python object.\n"
                        "     (Use --dataset_path\n")
    parser.add_argument("--env", help="The environment you want to learn the reward model for", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--env_type", help="The type of environment ie atari, mujoco ect", type=str, default="atari",
                        choices=["algorithmic", "atari", "box2d", "classic_control", "ChessWrapper", "mujoco",
                                 "robotics", "toy_text"])
    parser.add_argument("--model_dir", help="The directory that holds all of the models to use as demonstrators"
                                            "(see --help for details)", type=str)
    parser.add_argument("--video_dir", help="The directory that contains the videos to use as demonstrations"
                                            "(see --help for    details", type=str)
    parser.add_argument("--dataset_path", help="The path to the dataset file "
                                               "(see --help for details)", type=str)
    parser.add_argument("--demos_per_model", help="The number of demonstrations to be created per model found", type=int,
                        default=1)
    parser.add_argument("--num_full_demonstrations", help="The number of full demonstrations in the train and test set",
                        type=int, default=0)
    parser.add_argument("--num_sub_demonstrations", help="The number of sub demonstrations (snippets of full demos) in "
                                                         "the train and test set", type=int, default=6000)
    parser.add_argument("--min_snippet_length", help="The minimum length of the sub demonstrations", type=int, default=50)
    parser.add_argument("--max_snippet_length", help="The maximin length of the sub demonstrations", type=int,default=150)
    parser.add_argument("--learning_rate", help="The learning rate used for the network training", type=float, default=0.00005)
    parser.add_argument("--training_epochs", help="The number of epochs used to train the network", type=int, default=5)
    parser.add_argument("--save_path", help="The file the network will be saved to after training", type=str,
                        required=True)
    parser.add_argument("--checkpoint_dir", help="The directory the where log files and checkpoints after each epoch "
                                                 "will be saved", type=str)
    return parser

