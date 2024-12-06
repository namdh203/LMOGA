import math
import pickle
from environs import Env
import lgsvl
import random
import sys
import os
import time
from lgsvl.dreamview import CoordType
import lgsvl
import numpy as np
from shapely.geometry.polygon import Point, Polygon
import lgsvl.geometry

APOLLO_HOST = ""  # or 'localhost'
PORT = 8977
DREAMVIEW_PORT = 42164
BRIDGE_PORT = 42077
time_offset = 9

SIMULATOR_HOST = os.environ.get("SIMULATOR_HOST", "127.0.0.1")
SIMULATOR_PORT = int(os.environ.get("SIMULATOR_PORT", PORT))
BRIDGE_HOST = os.environ.get("BRIDGE_HOST", APOLLO_HOST)
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", BRIDGE_PORT))
startPos = lgsvl.Vector(-564.6, 10.2, 265.5)
# startPos = lgsvl.Vector(-540.1, 10.2, 58.8)
destPos = lgsvl.Vector(-445.7, 10.2, -22.7)

lanes_map = None
junctions_map = None
lanes_junctions_map = None

def main():
    # init simulator
    env = Env()
    sim = lgsvl.Simulator(SIMULATOR_HOST, SIMULATOR_PORT)

    # load map
    if sim.current_scene == "12da60a7-2fc9-474d-a62a-5cc08cb97fe8":
        sim.reset()
    else:
        sim.load("12da60a7-2fc9-474d-a62a-5cc08cb97fe8")
    sim.set_time_of_day(19, fixed=True)

    # create ego
    ego_state = lgsvl.AgentState()
    ego_state.transform = sim.map_point_on_lane(startPos)
    ego = sim.add_agent("SUV", lgsvl.AgentType.NPC, ego_state)
    # ego = sim.add_agent("8e776f67-63d6-4fa3-8587-ad00a0b41034", lgsvl.AgentType.EGO, ego_state)
    forward = lgsvl.utils.transform_to_forward(ego_state.transform)
    ego_state.velocity = 3 * forward

    # connect brigde

    # ego.connect_bridge(BRIDGE_HOST, BRIDGE_PORT)
    # while not ego.bridge_connected:
    #     time.sleep(1)
    # print("Bridge connected")

    # Dreamview setup

    # dv = lgsvl.dreamview.Connection(sim, ego, APOLLO_HOST, str(DREAMVIEW_PORT))
    # dv.set_destination(destPos.x, destPos.z, 0, CoordType.Unity)

    npc_location = [
        # lgsvl.Vector(-552.4, 10.2, 21.5)
        # lgsvl.Vector(-540.1, 10.2, 58.8)
        # lgsvl.Vector(-576.4, 10.2, 75.4)
        # lgsvl.Vector(-593.4, 10.2, 39.2)
        ## lgsvl.Vector(-564.6, 10.2, 265.5)
        # lgsvl.Vector(-575.6, 10.2, 273.5)
        # lgsvl.Vector(-519.3, 10.2, 228.3)
        # lgsvl.Vector(-575.4, 10.2, 241.2)
        lgsvl.Vector(-522.0, 10.2, 274.0)
    ]

    # create npc
    n = len(npc_location) #num of npc
    npc = []

    def on_stop_line(agent):
        print(agent.name, "reached stop line")

    def on_lane_change(agent):
        print(agent.name, "is changing lanes")

    def transform_apollo_coord_to_lgsvl_coord(apollo_x, apollo_y):
        point = sim.map_from_gps(northing = apollo_y, easting = apollo_x)
        other = sim.map_point_on_lane(point.position)
        point.position.y = other.position.y
        return point

    def load_map_traffic_condition():
        global lanes_map
        global junctions_map
        global lanes_junctions_map

        map_name = "sanfrancisco"
        map_dir = ''
        lanes_map_file = f"{map_dir}map/{map_name}_lanes.pkl"

        with open(lanes_map_file, "rb") as file:
            lanes_map = pickle.load(file)

        file.close()

        junctions_map_file = f"{map_dir}map/{map_name}_junctions.pkl"

        with open(junctions_map_file, "rb") as file2:
            junctions_map = pickle.load(file2)

        file2.close()

        lanes_junctions_map_file = f"{map_dir}map/{map_name}_lanes_junctions.pkl"

        with open(lanes_junctions_map_file, "rb") as file3:
            lanes_junctions_map = pickle.load(file3)

        file3.close()
        print(lanes_map['lane_50'])
    load_map_traffic_condition()

    print(sim.map_point_on_lane(lgsvl.Vector(-592.5, 10.2, 303.0)))

    print("-------------------------------", lanes_map['lane_1548']['central_curve'])

    def get_way_point_on_lane(lane_id, npc):
        wp = []
        points = []
        print("----------------", npc.state.rotation, npc.state.position)
        first_point = sim.map_to_gps(npc.transform)
        for i in range (len(lanes_map[lane_id]['central_curve'])):
            point = lanes_map[lane_id]['central_curve'][i]

            point_on_lane = [point['x'], point['y']]
            points.append(point_on_lane)

        for i in range (len(points)):
            if i + 1 < len(points):
                rotation = math.degrees(math.atan2(points[i + 1][1] - points[i][1],
                    points[i + 1][0]- points[i][0]))
            else:
                rotation = math.degrees(math.atan2(points[i][1] - points[i - 1][1],
                    points[i][0] - points[i - 1][0]))

            rotation = (360 - rotation) % 360
            print("rotation----------", rotation)
            cur_point = transform_apollo_coord_to_lgsvl_coord(points[i][0], points[i][1])
            wp.append(lgsvl.DriveWaypoint(cur_point.position, speed=13.4, angle=lgsvl.Vector(0, rotation, 0)))

        for wpp in wp:
            print(wpp.position, wpp.angle)
        return wp

    def point_convert_to_lane(transform_point):
        res_lane = []
        gps = sim.map_to_gps(transform_point)
        point = Point(np.array([gps.easting, gps.northing]))
        for lane_id in lanes_map:
            points_lane = []
            for point_bound in lanes_map[lane_id]['left_boundary']:
                points_lane.append(([point_bound['x'], point_bound['y']]))
            for point_bound in lanes_map[lane_id]['right_boundary'][::-1]:
                points_lane.append(([point_bound['x'], point_bound['y']]))
            pp = Polygon(points_lane)
            if pp.contains(point):
                res_lane.append(lane_id)
        return res_lane

    def distane_to_lane(lane_id, ohter_lane):
        points_lane = []
        for point_bound in lanes_map[lane_id]['left_boundary']:
            points_lane.append(([point_bound['x'], point_bound['y']]))
        for point_bound in lanes_map[lane_id]['right_boundary'][::-1]:
            points_lane.append(([point_bound['x'], point_bound['y']]))
        pp = Polygon(points_lane)

        ohter_points = []
        for point_bound in lanes_map[ohter_lane]['left_boundary']:
            ohter_points.append(([point_bound['x'], point_bound['y']]))
        for point_bound in lanes_map[ohter_lane]['right_boundary'][::-1]:
            ohter_points.append(([point_bound['x'], point_bound['y']]))
        pp2 = Polygon(ohter_points)
        return pp.distance(pp2)

    for i in range(n):
        npc_state = lgsvl.AgentState()
        npc_state.transform = sim.map_point_on_lane(npc_location[i])
        name = random.choice(["Sedan", "Jeep", "Hatchback"])
        npc.append(sim.add_agent(name, lgsvl.AgentType.NPC, npc_state))
        npc[i].on_stop_line(on_stop_line)
        npc[i].on_lane_change(on_lane_change)

    # set simulation camera

    cam_tr = ego.state.transform
    up = lgsvl.utils.transform_to_up(cam_tr)
    forward = lgsvl.utils.transform_to_forward(cam_tr)
    cam_tr.position += up * 3 - forward * 5
    sim.set_sim_camera(cam_tr)

    # endPointNpc = sim.map_point_on_lane(lgsvl.Vector(-592.5, 10.2, 303.0))
    # endPointNpc = sim.map_point_on_lane(lgsvl.Vector(-592.5, 10.2, 303.0))

    print('lane first', lanes_map['lane_574'])
    print('---------------------------')
    print('lane second', lanes_map['lane_575'])

    for t in range(20):
        for x in range(4):
            print("------", t, x)
            for i in range(n):
                print(npc[i])
                if x == 0:
                    npc[i].follow_closest_lane(True, 13.4)
                    npc[i].change_lane(False)
                elif x == 1:
                    forward = lgsvl.utils.transform_to_forward(npc[i].transform)
                    right = lgsvl.utils.transform_to_right(npc[i].transform)

                    all_overlap_lane = point_convert_to_lane(npc[i].transform)

                    turn_right_lane = point_convert_to_lane(
                        # sim.map_point_on_lane(npc[i].state.position + 35 * forward))
                        sim.map_point_on_lane(npc[i].state.position + 5 * forward + 25 * right))
                        # sim.map_point_on_lane(npc[i].state.position + 35 * forward))

                    print(all_overlap_lane, turn_right_lane)

                    min_lane_distane = 100000
                    cur_lane = None

                    for overlap_lane in all_overlap_lane:
                        next_lane = lanes_map[overlap_lane]['successor']
                        for each_lane in next_lane:
                            for right_lane in turn_right_lane:
                                # print("each right", each_lane, right_lane)
                                d_lane_to_lane = distane_to_lane(right_lane, each_lane)
                                if min_lane_distane > d_lane_to_lane:
                                    min_lane_distane = d_lane_to_lane
                                    cur_lane = overlap_lane

                    wp = get_way_point_on_lane(cur_lane, npc[i])
                    npc[i].follow(wp)
                    print(cur_lane, min_lane_distane)
                else:
                    npc[i].follow_closest_lane(True, 13.4)

            for j in range(12):
                # for i in range(n):
                #     print("position rotation ",npc[i].state.position, npc[i].state.rotation)
                sim.run(0.25)


if __name__ == '__main__':
    main()
