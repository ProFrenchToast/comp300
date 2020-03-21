import subprocess
import sys

from comp300.LearningModel.cmd_utils import checkpointParser


def checkpointTraining(args):
    """
    Trains a set of demonstrators and saves them to a given directory.

    Parameters
    ----------
    args : parsed args
        The known args taken from commandline.

    Returns
    -------

    """
    envName = args.env
    envType = args.env_type
    algorithm = args.alg
    if args.checkpoint_dir is None:
        checkpointDir = "~/models/{}-demonstator".format(envName)
    else:
        checkpointDir = args.checkpoint_dir
    if not args.log_dir is None:
        log_dir = "--log_path={}".format(args.log_dir)
    else:
        log_dir = ""
    stepSize = args.step_size
    noCheckpoints = args.num_checkpoints

    run_checkpoint_training(algorithm, checkpointDir, envName, log_dir, noCheckpoints, stepSize)


def run_checkpoint_training(algorithm, checkpointDir, envName, log_dir, noCheckpoints, stepSize):
    """
    Runs the checkpoint training given the correct parameters.

    Parameters
    ----------
    algorithm : str
        The algorithm to be used for learning.
    checkpointDir : str
        The path to the directory to save the demonstrators.
    envName : str
        The environment id to train the demonstrators in.
    log_dir : str
        The directory to save the log data to.
    noCheckpoints : int
        The number of demonstrators to train.
    stepSize : float
        The number of training steps between demonstrators

    Returns
    -------

    """
    # first generate the initial step
    p = subprocess.Popen("python -m baselines.run --alg={} --env={} --num_timesteps={} --save_path={}/{} {}"
                         .format(algorithm, envName, stepSize, checkpointDir, stepSize, log_dir), shell=True)
    p.wait()
    lastTrained = stepSize
    for checkpoint in range(1, noCheckpoints):
        nextTrained = lastTrained + stepSize
        p = subprocess.Popen(
            "python -m baselines.run --alg={} --env={} --num_timesteps={} --save_path={}/{} --load_path={}/{} {}"
                .format(algorithm, envName, stepSize, checkpointDir, nextTrained, checkpointDir, lastTrained, log_dir),
            shell=True)
        p.wait()
        print("trained checkpoint {} to {}".format(lastTrained, nextTrained))
        lastTrained = nextTrained


if __name__ == '__main__':
    args_parser = checkpointParser()
    args, unknown_args = args_parser.parse_known_args(sys.argv)

    checkpointTraining(args)
    print("finished training")



