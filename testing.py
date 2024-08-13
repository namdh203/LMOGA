import pickle
import os

# if os.path.isfile("GaCheckpointsCrossroads/last_gen.obj"):
#     f_f = open("GaCheckpointsCrossroads/last_gen.obj", "rb")
#     result = pickle.load(f_f)
#     f_f.close()
#     print(result)

if os.path.isfile("GaCheckpointsCrossroads/last_gen.obj"):
    f_f = open("GaCheckpointsCrossroads/last_gen.obj", "rb")
    result = pickle.load(f_f)
    f_f.close()
    print(result)