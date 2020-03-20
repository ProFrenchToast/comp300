import subprocess
import math
import sys
from comp300.LearningModel.cmd_utils import checkpointParser

if __name__ == '__main__':
    args_parser = checkpointParser()
    args, unknown_args = args_parser.parse_known_args(sys.argv)

    envName = args.env
    envType = args.env_type
    algorithm = args.alg
    if args.checkpoint_dir is None:
        checkpointDir = "~/models/{}-demonstator".format(envName)
    else:
        checkpointDir = args.checkpoint_dir
    stepSize = args.step_size
    noCheckpoints = args.num_checkpoints
    if not args.log_dir is None:
        log_dir = "--log_path={}".format(args.log_dir)
    else:
        log_dir = ""

    #first generate the initial step
    p = subprocess.Popen("python -m baselines.run --alg={} --env={} --num_timesteps={} --save_path={}/{} {}"
                         .format(algorithm, envName, stepSize, checkpointDir, stepSize, log_dir), shell=True)
    p.wait()

    lastTrained = stepSize

    for checkpoint in range(1, noCheckpoints):
        nextTrained = lastTrained + stepSize
        p = subprocess.Popen("python -m baselines.run --alg={} --env={} --num_timesteps={} --save_path={}/{} --load_path={}/{} {}"
                             .format(algorithm, envName, stepSize, checkpointDir, nextTrained, checkpointDir, lastTrained, log_dir),
                             shell=True)
        p.wait()
        print("trained checkpoint {} to {}".format(lastTrained, nextTrained))
        lastTrained = nextTrained
    print("finished training")



