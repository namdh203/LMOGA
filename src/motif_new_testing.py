import math
import pickle
from environs import Env
import lgsvl
import random
import sys

def replay(sim):
    if sim.current_scene == "12da60a7-2fc9-474d-a62a-5cc08cb97fe8":
        sim.reset()
    else:
        sim.load("12da60a7-2fc9-474d-a62a-5cc08cb97fe8")
    sim.set_time_of_day(19, fixed=True)

    # create ego
    ego_state = lgsvl.AgentState()
    ego_state.transform = sim.map_point_on_lane(lgsvl.Vector(-540.1, 10.2, 58.8))
    ego = sim.add_agent("SUV", lgsvl.AgentType.NPC,
                            ego_state)

    # sim.set_follow(ego, follow_distance = 5, follow_height = 2)

    npc_location = [
        lgsvl.Vector(-552.4, 10.2, 21.5)
        # lgsvl.Vector(-540.1, 10.2, 58.8),
        # lgsvl.Vector(-576.4, 10.2, 75.4)
        # lgsvl.Vector(-593.4, 10.2, 39.2)
    ]

    # create npc
    n = len(npc_location) #num of npc
    npc = []
    for i in range(n):
        npc_state = lgsvl.AgentState()
        npc_state.transform = sim.map_point_on_lane(npc_location[i])
        name = random.choice(["Sedan", "Jeep", "Hatchback"])
        npc.append(sim.add_agent(name, lgsvl.AgentType.NPC, npc_state))

    # set simulation camera
    cam_tr = ego.state.transform
    up = lgsvl.utils.transform_to_up(cam_tr)
    forward = lgsvl.utils.transform_to_forward(cam_tr)
    cam_tr.position += up * 3 - forward * 5
    sim.set_sim_camera(cam_tr)

    for t in range(20):
        for x in range(4):
            print("------", x)
            for i in range(n):
                npc[i].follow_closest_lane(True, 13.4)
                if (random.random() > 0):
                    left = True
                    # if (random.random() > 0.5):
                    #     left = False
                    npc[i].change_lane(left)
            for j in range(6):
                sim.run(0.25)


def main():
    # initial simulation
    env = Env()
    sim = lgsvl.Simulator("127.0.0.1", 8977)
    replay(sim)


if __name__ == '__main__':
    main()
