import pickle
import os

if os.path.isfile("AccidentScenarioCrossroads/accident-gen-08-30-2024-07-16-50"):
    f_f = open("AccidentScenarioCrossroads/accident-gen-08-30-2024-07-16-50", "rb")
    result = pickle.load(f_f)
    f_f.close()
    print(result.ttc)

# if os.path.isfile("trajectory(another try).obj"):
#     f_f = open("trajectory(another try).obj", "rb")
#     result = pickle.load(f_f)
#     f_f.close()
#     print(result)