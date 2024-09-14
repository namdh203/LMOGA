import sys
import os
import pickle
from dotmap import DotMap

sys.path.append('/home/kasm_user/PycharmProjects/reduceRestart/')


def main():
    GaData = DotMap()
    GaData.bounds = [[1, 2], [0, 5], [3, 12], [0, 3], [1, 7], [-2.5, 3], [1, 10], [1, 4]]  # [[speed range], [actions]]
    GaData.mutationProb = 0.4  # mutation rate
    GaData.crossoverProb = 0.4  # crossover rate
    GaData.popSize = 4
    GaData.numOfNpc = 3
    GaData.numOfTimeSlice = 2
    GaData.maxGen = 600 #600
    isRestart = False
    count = 0

    for x in range(0, GaData.maxGen):
        print("==== This is iteration: {x}th ====".format(x = x))
        if os.path.isfile('result.obj'):
            os.remove("result.obj")
        # Dump genetic algorithm parameters
        s_f = open('scenario.obj', 'wb')
        print("isRestart = {}".format(isRestart))
        pickle.dump([GaData, isRestart], s_f)
        s_f.truncate()
        s_f.close()

        os.system("python3 simulation.py scenario.obj result.obj")
        resultObj = None

        # Read fitness score
        if os.path.isfile('result.obj'):
            f_f = open('result.obj', 'rb')
            resultObj = pickle.load(f_f)
            f_f.close()

        print("********* resultObj here ********", resultObj)
        # Restart of npc lost
        if resultObj is None or resultObj['ttc'] == 0.0 or resultObj['ttc'] == '' or resultObj['fault'] == 'npcTooLong':
            if os.path.exists('GaCheckpointsCrossroads/last_gen.obj'):
                with open('GaCheckpointsCrossroads/last_gen.obj', "rb") as f:
                    if pickle.load(f):
                        isRestart = True
        else:
            print("=============Finish===================", resultObj)
            break


if __name__ == '__main__':
    main()
