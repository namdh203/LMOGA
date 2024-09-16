import pickle
import os
import sys

objPath = sys.argv[1]

if os.path.isfile(objPath):
    f_f = open(objPath, "rb")
    result = pickle.load(f_f)
    f_f.close()
    # print(result)
    for pop in result:
        print("ttc, smoothness, collision", pop.ttc, pop.smoothness, pop.isCollision)
        # for speed in pop.npcSpeed:
        #     print(speed)
        # for action in pop.npcAction:
        #     print(action)

# if os.path.isfile("trajectory(another try).obj"):
#     f_f = open("trajectory(another try).obj", "rb")
#     result = pickle.load(f_f)
#     f_f.close()
#     print(result)