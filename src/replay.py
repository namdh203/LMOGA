import math
import pickle
from environs import Env
import lgsvl
import random
import sys

def replay(scenario, sim):

    if sim.current_scene == "12da60a7-2fc9-474d-a62a-5cc08cb97fe8":
        sim.reset()
    else:
        sim.load("12da60a7-2fc9-474d-a62a-5cc08cb97fe8")
    sim.set_time_of_day(19, fixed=True)

    ego_location = scenario.egoLocation
    ego_speed = scenario.egoSpeed
    npc_location = scenario.npcLocation
    npc_speed = scenario.npcSpeed

    # create ego
    ego_state = lgsvl.AgentState()
    ego_state.transform = ego_location[0]
    ego = sim.add_agent("SUV", lgsvl.AgentType.NPC,
                            ego_state)

    # sim.set_follow(ego, follow_distance = 5, follow_height = 2)

    # create npc
    n = len(npc_location) #num of npc
    m = len(ego_location) #num of ego location
    print("num of location", len(npc_location[0]), m)
    npc = []
    for i in range(n):
        npc_state = lgsvl.AgentState()
        npc_state.transform = npc_location[i][0]
        name = random.choice(["Sedan", "Jeep", "Hatchback"])
        npc.append(sim.add_agent(name, lgsvl.AgentType.NPC, npc_state))

    # util function
    def cal_speed(speed):
        return math.sqrt(speed.x ** 2 + speed.y ** 2 + speed.z ** 2)

    def on_waypoint(agent, index):
        print("waypoint {} reached".format(index))

    # ego waypoints
    ego_waypoints = []
    for i in range(1, len(ego_location)):
        wp = lgsvl.DriveWaypoint(position=ego_location[i].position, angle=ego_location[i].rotation,
                                 speed=ego_speed[i])
        ego_waypoints.append(wp)
    ego.follow(ego_waypoints)

    # npc wypoints
    for i in range(n):
        npc_waypoints = []
        for j in range(1, len(npc_location[i])):
            wp = lgsvl.DriveWaypoint(position=npc_location[i][j].position, angle=npc_location[i][j].rotation,
                                     speed=cal_speed(npc_speed[i][j]))
            npc_waypoints.append(wp)
        npc[i].follow(npc_waypoints)

    # set simulation camera
    # cam_tr = ego_location[0]
    # up = lgsvl.utils.transform_to_up(cam_tr)
    # forward = lgsvl.utils.transform_to_forward(cam_tr)
    # cam_tr.position += up * 3 - forward * 5
    # sim.set_sim_camera(cam_tr)

    # run simulation
    cnt = 0
    while cnt < m:
        cam_tr = ego_location[cnt]
        up = lgsvl.utils.transform_to_up(cam_tr)
        forward = lgsvl.utils.transform_to_forward(cam_tr)
        cam_tr.position += up * 5 - forward * 12
        sim.set_sim_camera(cam_tr)
        cnt += 1
        sim.run(1)


def main(path):
    with open(path, 'rb') as f:
        scenario = pickle.load(f)

    # initial simulation
    env = Env()
    sim = lgsvl.Simulator("127.0.0.1", 8977)

    for i in range(4):
        print("scenario[{}]".format(i))
        replay(scenario[i], sim)


if __name__ == '__main__':
    path = sys.argv[1]
    main(path)
