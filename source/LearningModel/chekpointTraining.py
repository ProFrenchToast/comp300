import subprocess
import math

envName = "BreakoutNoFrameskip-v4"
algorithm = "ppo2"
checkpointDir = "~/models/{}-groundTruth-test".format(envName)
stepSize = int(math.pow(10, 3))
stepSizeStr = "1e3"
noCheckpoints = 20

#first generate the initial step
p = subprocess.Popen("python -m baselines.run --alg={} --env={} --num_timesteps={} --save_path={}/{}"
                     .format(algorithm, envName, stepSizeStr, checkpointDir, stepSize), shell=True)
p.wait()

lastTrained = stepSize

for checkpoint in range(1, noCheckpoints):
    nextTrained = lastTrained + stepSize
    p = subprocess.Popen("rm ~/nohup.out", shell=True)
    p.wait()
    p = subprocess.Popen("python -m baselines.run --alg={} --env={} --num_timesteps={} --save_path={}/{} --load_path={}/{}"
                         .format(algorithm, envName, stepSizeStr, checkpointDir, nextTrained, checkpointDir, lastTrained),
                         shell=True)
    p.wait()
    print("trained checkpoint {} to {}".format(lastTrained, nextTrained))
    lastTrained = nextTrained
print("finished training")



