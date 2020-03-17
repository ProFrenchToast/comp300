import argparse

def arg_parser():
    """
    Create an empty parser that each method will build on.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def checkpointParser():
    parser = arg_parser()
    parser.add_argument("--env", help="The environment ID you wish to train", type=str, default="BreakoutNoFrameSkip-v4")
    parser.add_argument("--env_type", help="The type of environment ie atari, mujoco ect", type=str, default="atari",
                        choices=["algorithmic","atari", "box2d", "classic_control", "ChessWrapper", "mujoco", "robotics", "toy_text"])
    parser.add_argument("--alg", help="The reinforcement learning algorithim you want to use", type=str, default="ppo2")
    parser.add_argument("--checkpoint_dir", help="The directory the models will be saved to", type=str)
    parser.add_argument("--step_size", help="The number of training steps between demonstrations", type=float,
                        default=1e5)
    parser.add_argument("--num_checkpoints", help="The number of checkpoints to be trained", type=int, default=20)
    parser.add_argument("--log_dir", help="The directory where the log files will be stored -note this will overwrite "
                                          "existing files in the directory", type=str)
    return parser
