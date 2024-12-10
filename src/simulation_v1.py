import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' #avoid some limit of threads server allow
import lgsvl
import sys
import time
import math
import random
import pickle
import liability
from datetime import datetime
import util
import copy
from MultiObjGeneticAlgorithm import MultiObjGenticAlgorithm
from os import listdir
import numpy as np
from lgsvl.dreamview import CoordType
from shapely.geometry.polygon import Point, Polygon

from MutlChromosome_v1 import MutlChromosome

import json

APOLLO_HOST = "27.65.254.208"  # or 'localhost'
PORT = 8977
DREAMVIEW_PORT = 32428
BRIDGE_PORT = 32487
time_offset = 9

lanes_map = None
junctions_map = None
lanes_junctions_map = None

class LgApSimulation:
    def __init__(self):
        self.SIMULATOR_HOST = os.environ.get("SIMULATOR_HOST", "127.0.0.1")
        self.SIMULATOR_PORT = int(os.environ.get("SIMULATOR_PORT", PORT))
        self.BRIDGE_HOST = os.environ.get("BRIDGE_HOST", APOLLO_HOST)
        self.BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", BRIDGE_PORT))
        self.totalSimTime = 15

        self.sim = None
        self.ego = None  # There is only one ego
        self.initEvPos = lgsvl.Vector(-464.4, 10.2, 330.5)
        self.endEvPos = lgsvl.Vector(-436.3, 10.2, 143.7)
        self.mapName = "12da60a7-2fc9-474d-a62a-5cc08cb97fe8"
        self.roadNum = 1
        self.npcList = []  # The list contains all the npc added
        self.pedetrianList = []
        self.egoSpeed = []
        self.egoLocation = []
        self.initSimulator()
        self.loadMap()
        self.initEV()
        self.isEgoFault = False
        self.isHit = False
        self.connectEvToApollo()
        self.maxint = 130
        self.egoFaultDeltaD = 0
        self.isCollision = 0
        self.load_map_traffic_condition()

    def get_speed(self, vehicle):
        vel = vehicle.state.velocity
        return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def initSimulator(self):
        print("init Simulator")
        sim = lgsvl.Simulator(self.SIMULATOR_HOST, self.SIMULATOR_PORT)
        self.sim = sim

    def loadMap(self):
        print("load Map...")
        sim = self.sim
        print(sim.current_scene)
        if sim.current_scene == self.mapName:
            sim.reset()
        else:
            sim.load(self.mapName)

    def initEV(self):
        print("init EV")
        sim = self.sim
        egoState = lgsvl.AgentState()
        spawn = sim.get_spawn()

        egoState.transform = sim.map_point_on_lane(self.initEvPos)
        forward = lgsvl.utils.transform_to_forward(egoState.transform)
        egoState.velocity = 3 * forward
        ego = sim.add_agent("8e776f67-63d6-4fa3-8587-ad00a0b41034", lgsvl.AgentType.EGO,
                            egoState)
        self.ego = ego
        sim.set_time_of_day((10 + time_offset) % 24, fixed=True)

    def connectEvToApollo(self):
        print("connect to apollo")
        ego = self.ego
        ego.connect_bridge(self.BRIDGE_HOST, self.BRIDGE_PORT)
        while not ego.bridge_connected:
            time.sleep(1)
        print("Bridge connected")
        print(self.BRIDGE_HOST)
        # Dreamview setup
        dv = lgsvl.dreamview.Connection(self.sim, ego, APOLLO_HOST, str(DREAMVIEW_PORT))
        spawns = self.sim.get_spawn()

        dv.set_destination(self.endEvPos.x, self.endEvPos.z, 0, CoordType.Unity)
        time.sleep(5)

    def restartLGSVL(self):
        self.loadMap()
        self.initEV()
        self.connectEvToApollo()

    def addNpcVehicle(self, posVector, vehicleType="SUV"):
        sim = self.sim

        def on_collision(agent1, agent2, contact):
            """
            collision listener function
            """

            if agent2 is None or agent1 is None:
                self.isEgoFault = True
                util.print_debug(" --- Hit road obstacle --- ")
                return

            apollo = agent1
            npcVehicle = agent2
            if agent2.name == "8e776f67-63d6-4fa3-8587-ad00a0b41034":
                apollo = agent2
                npcVehicle = agent1
            else:
                self.isCollision += 1
                print("ego collision")


            util.print_debug(" --- On Collision, ego speed: " + str(apollo.state.speed) + ", NPC speed: " + str(
                npcVehicle.state.speed))
            npcVehicle.follow_closest_lane(True, 0)

            if apollo.state.speed <= 0.005:
                self.isEgoFault = False
                return

        npcList = self.npcList
        npcState = lgsvl.AgentState()
        npcState.transform = sim.map_point_on_lane(posVector)
        npc = sim.add_agent(vehicleType, lgsvl.AgentType.NPC, npcState)
        npc.on_collision(on_collision)
        npcList.append(npc)

    def addPedetrian(self, posVector, peopleType="Bob"):
        sim = self.sim
        pedetrianList = self.pedetrianList
        pedetrianState = lgsvl.AgentState()
        pedetrianState.transform.position = posVector
        pedestrian_rotation = lgsvl.Vector(0.0, 105.0, 0.0)
        pedetrianState.transform.rotation = pedestrian_rotation
        pedestrian = sim.add_agent(peopleType, lgsvl.AgentType.PEDESTRIAN, pedetrianState)
        pedetrianList.append(pedestrian)

    def addFixedMovingNpc(self, posVector, vehicleType="SUV"):
        sim = self.sim
        npcState = lgsvl.AgentState()
        npcState.transform = sim.map_point_on_lane(posVector)
        npc = sim.add_agent(vehicleType, lgsvl.AgentType.NPC, npcState)
        npc.follow_closest_lane(True, 13.4)

    # This function send an instance action command to the NPC at the current time instance
    def setNpcSpeed(self, npc, speed):
        npc.follow_closest_lane(True, speed)

    # Direction is either "LEFT" or "RIGHT"
    def setNpcChangeLane(self, npc, direction):
        if direction == "LEFT":
            npc.change_lane(True)
        elif direction == "RIGHT":
            npc.change_lane(False)

    def setEvThrottle(self, throttle):
        ego = self.ego
        c = lgsvl.VehicleControl()
        c.throttle = throttle
        ego.apply_control(c, True)

    def brakeDist(self, speed):
        dBrake = 0.0467 * pow(speed, 2.0) + 0.4116 * speed - 1.9913 + 0.5
        if dBrake < 0:
            dBrake = 0
        return dBrake

    # ---------------------------
    # Extract map information

    def transform_apollo_coord_to_lgsvl_coord(self, apollo_x, apollo_y):
        sim = self.sim
        point = sim.map_from_gps(northing = apollo_y, easting = apollo_x)
        other = sim.map_point_on_lane(point.position)
        point.position.y = other.position.y
        return point

    def load_map_traffic_condition(self):
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

    def get_way_point_on_lane(self, lane_id, set_speed):
        sim = self.sim
        wp = []
        points = []

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
            cur_point = self.transform_apollo_coord_to_lgsvl_coord(points[i][0], points[i][1])
            wp.append(lgsvl.DriveWaypoint(cur_point.position, speed=set_speed, angle=lgsvl.Vector(0, rotation, 0)))

        return wp

    def point_convert_to_lane(self, transform_point):
        sim = self.sim
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

    def distane_to_lane(self, lane_id, ohter_lane):
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

    def is_in_junction(self, object_transform): #p_point is vector of position lgsvl
        sim = self.sim
        gps = sim.map_to_gps(object_transform)
        p_point = Point(np.array([gps.easting, gps.northing]))
        closet_dis = 999999
        res_id = 0

        for id, lane_polygon in junctions_map.items():
            if lane_polygon.contains(p_point):
                return id, 2

            dis = lane_polygon.boundary.distance(p_point)
            if dis < closet_dis:
                closet_dis = dis
                res_id = id

        print("closet dis", closet_dis)
        if closet_dis <= 0.5:
            return res_id, 2
        elif closet_dis <= 20:
            return res_id, 1
        else:
            return None, 0

    # ----------------------------------------
    # Motif excuted


    # -----------------------------------------
    # Objective for fitness
    def jerk(self, egoSpeedList):
        """
        ego vehicle jerk parameter
        one of the parameters of the fitness function
        """
        aChange = []
        jerkChange = []
        for i in range(1, len(egoSpeedList)):
            aChange.append((egoSpeedList[i] - egoSpeedList[i - 1]) / (0.5))
        for j in range(1, len(aChange)):
            jerkChange.append((aChange[j] - aChange[j - 1]) / 0.5)
        return np.var(jerkChange)

    def findttc(self, ego, npc):
        """
        time to collision
        one of the parameters of the fitness function
        """
        pos_npc = npc.state.transform.position
        pos_ego = ego.state.transform.position
        x2 = pos_npc.x - pos_ego.x
        z2 = pos_npc.z - pos_ego.z
        ego_v_x = ego.state.velocity.x
        ego_v_z = ego.state.velocity.z
        npc_v_x = npc.state.velocity.x
        npc_v_z = npc.state.velocity.z
        A_x = ego_v_z * (x2 * ego_v_z - z2 * npc_v_x)
        B_x = ego_v_x * npc_v_z
        C_x = npc_v_x * ego_v_z
        pre_x = float("inf")
        if B_x - C_x != 0:
            pre_x = A_x / (B_x - C_x)
        A_z = ego_v_x * (z2 * npc_v_x - x2 * npc_v_z)
        B_z = ego_v_z * npc_v_x
        C_z = npc_v_z * ego_v_x
        pre_z = float("inf")
        if B_z - C_z != 0:
            pre_z = A_z / (B_z - C_z)
        ego_speed = ego.state.speed
        if ego_speed == 0:
            ego_speed = float("inf")

        ttc = math.sqrt(pre_x * pre_x + pre_z * pre_z) / ego_speed
        if math.isinf(ttc) or math.isnan(ttc) or ttc == 0:
            ttc = 99999999
        return ttc

    def findPathSimilarity(self, egoPathList, router):
        """
        path similarity of ego vehicle
        one of the parameters of the fitness function
        """
        a = 4.191e-08
        b = - 1.579e-05
        c = 0.003641
        d = 765.5
        egoSimilarity = []
        k = 0
        for egoPath in egoPathList:
            # pred_x = w * egoPath.z + b
            pred_x = a * egoPath.z * egoPath.z * egoPath.z + b * egoPath.z * egoPath.z + c * egoPath.z + d
            # if pred_x-0.5 <= egoPath.x <= pred_x+0.5:
            if egoPath.x < pred_x or pred_x < egoPath.x:
                egoSimilarity.append(math.sqrt((float(router[k][0]) - egoPath.x) ** 2 + (float(router[k][2]) - egoPath.z) ** 2))
            k += 1

        if egoSimilarity == []:
            egoSimilarity.append(1)
        return np.var(egoSimilarity)

    def findFitness(self, deltaDlist, dList, isEgoFault, isHit, hitTime):
        """
        fitness function
        the higher the fitness ranking, the better
        """
        minDeltaD = self.maxint
        for npc in deltaDlist:  # ith NPC
            hitCounter = 0
            for deltaD in npc:
                if isHit == True and hitCounter == hitTime:
                    break
                if deltaD < minDeltaD:
                    minDeltaD = deltaD  # Find the min deltaD over time slices for each NPC as the fitness
                hitCounter += 1
        util.print_debug(deltaDlist)
        util.print_debug(" *** minDeltaD is " + str(minDeltaD) + " *** ")

        minD = self.maxint
        for npc in dList:  # ith NPC
            hitCounter = 0
            for d in npc:
                if isHit == True and hitCounter == hitTime:
                    break
                if d < minD:
                    minD = d
                hitCounter += 1
        util.print_debug(dList)
        util.print_debug(" *** minD is " + str(minD) + " *** ")

        fitness = minDeltaD

        return fitness

    def is_within_distance_ahead(self, target_transform, current_transform):
        """
        return the the relative positional relationship between ego and npc
        """
        target_vector = np.array([target_transform.position.x - current_transform.position.x,
                                  target_transform.position.z - current_transform.position.z])
        norm_target = np.linalg.norm(target_vector)
        if norm_target < 0.001:
            return True
        fwd = lgsvl.utils.transform_to_forward(current_transform)
        forward_vector = np.array([fwd.x, fwd.z])

        d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

        if d_angle == 0:
            return "SameLaneBehind"
        elif d_angle < 37.0:
            return 'behind'
        elif 37.0 <= d_angle <= 143:
            return 'parall'
        elif d_angle == 180:
            return "SameLaneAhead"
        else:
            return 'ahead'

    def is_within_distance_right(self, target_transform, current_transform):
        """
        judge if there is a car around npc
        """
        target_vector = np.array([target_transform.position.x - current_transform.position.x,
                                  target_transform.position.z - current_transform.position.z])
        norm_target = np.linalg.norm(target_vector)
        if norm_target < 0.001:
            return True
        fwd = lgsvl.utils.transform_to_right(current_transform)
        forward_vector = np.array([fwd.x, fwd.z])

        d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

        if d_angle == 0:
            return "right"
        elif d_angle <= 90:
            return 'right'
        else:
            return 'left'

    def is_cross_junction(self, target_transform, current_transform):
        target_transform = sim.map_point_on_lane(lgsvl.Vector(target_transform.position.x,
            target_transform.position.y, target_transform.position.z))
        current_transform = sim.map_point_on_lane(lgsvl.Vector(current_transform.position.x,
            current_transform.position.y, current_transform.position.z))

        target_vector = np.array([
            target_transform.position.x - current_transform.position.x,
            target_transform.position.z - current_transform.position.z
        ])
        norm_target = np.linalg.norm(target_vector)
        if norm_target < 0.001:
            return True

        ego_yaw = current_transform.rotation.y
        ego_forward = np.array([math.cos(math.radians(ego_yaw)), math.sin(math.radians(ego_yaw))])

        npc_yaw = target_transform.rotation.y
        npc_forward = np.array([math.cos(math.radians(npc_yaw)), math.sin(math.radians(npc_yaw))])

        dot_product = np.dot(ego_forward, npc_forward)
        angle = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))

        if 80 <= angle <= 100:
            return "cross"
        else:
            return "not_cross"

    def initNpcVehicles(self, numOfNpc):
        # Set initial position of npc vehicle
        npcPosition = [
            [-458.5, 10.2, 342.3],
            [-456.0, 10.2, 339.2],
            [-474.5, 10.2, 326.0],
            [-601.9, 10.2, 296.1],
            [-596.0, 10.2, 295.6],
            # [-644.9, 10.2, 244.2],
            # [-634.8, 10.2, 245.8],
            [-493.6, 10.2, 203.5],
            [-491.2, 10.2, 206.5]
        ]

        self.npcList = []
        for position in npcPosition:
            self.addNpcVehicle(lgsvl.Vector(position[0], position[1], position[2]))

        return True


    def runGen(self, scenarioObj, weather):
        """
        parse the chromosome
        """

        # initialize simulation
        self.restartLGSVL()

        sim = self.sim
        ego = self.ego

        # Set signal loop
        controllables = self.sim.get_controllables("signal")
        for c in controllables:
            signal = self.sim.get_controllable(c.transform.position, "signal")
            control_policy = "trigger=200;green=50;yellow=5;red=5;loop"
            signal.control(control_policy)

        def on_collision(agent1, agent2, contact):
            """
            collision listener function
            """

            if agent2 is None or agent1 is None:
                self.isEgoFault = True
                util.print_debug(" --- Hit road obstacle --- ")
                return

            apollo = agent1
            npcVehicle = agent2
            if agent2.name == "8e776f67-63d6-4fa3-8587-ad00a0b41034":
                apollo = agent2
                npcVehicle = agent1
            else:
                self.isCollision += 1
                print("ego collision")

            npcVehicle.follow_closest_lane(True, npcVehicle.state.speed)

            util.print_debug(" --- On Collision, ego speed: " + str(apollo.state.speed) + ", NPC speed: " + str(
                npcVehicle.state.speed))
            if apollo.state.speed <= 0.005:
                self.isEgoFault = False
                return



        print("self is collison***", self.isCollision)

        ego.on_collision(on_collision)

        numOfTimeSlice = len(scenarioObj[0])
        numOfNpc = len(scenarioObj)

        # initialize npc vehicles
        checkInitNpc = self.initNpcVehicles(numOfNpc)

        deltaDList = [[self.maxint for i in range(numOfTimeSlice)] for j in
                      range(numOfNpc)]  # 1-D: NPC; 2-D: Time Slice
        dList = [[self.maxint for i in range(numOfTimeSlice)] for j in range(numOfNpc)]  # 1-D: NPC; 2-D:me Slice
        MinNpcSituations = [[self.maxint for i in range(numOfTimeSlice)] for j in range(numOfNpc)]
        egoSpeedList = []
        egoPathList = []

        sim.weather = lgsvl.WeatherState(rain=weather[0], fog=weather[1], wetness=weather[2], cloudiness=weather[3],
                                         damage=weather[4])
        npcList = self.npcList

        self.egoSpeed = []
        self.egoLocation = []
        self.npcSpeed = [[] for i in range(len(npcList))]
        self.npcLocation = [[] for i in range(len(npcList))]
        self.npcAction = [[] for i in range(len(self.npcList))]

        totalSimTime = self.totalSimTime
        actionChangeFreq = totalSimTime / numOfTimeSlice
        hitTime = numOfNpc
        resultDic = {}

        if checkInitNpc == False:
            resultDic['ttc'] = ''
            resultDic['fault'] = ''
            print('Can not initalize npc...')
            return resultDic

        print("numOfTimeSlice", numOfTimeSlice)

        # Execute scenarios based on genes
        for t in range(0, int(numOfTimeSlice)):
            print("t current", t)
            minNpcSituation = [0 for o in range(numOfNpc)]
            lane_stateList = []
            speedsList = []
            genes = []

            i = 0
            for npc in npcList:
                is_motif = None
                # Motif Gene
                if isinstance(scenarioObj[i][t][0], dict):
                    is_motif = True
                    lane_state = []
                    speeds = []
                    situation = self.is_within_distance_ahead(ego.state.transform, npc.state.transform)
                    direct = self.is_within_distance_right(ego.state.transform, npc.state.transform)
                    # Command = scenarioObj[i][t][1]
                    decelerate = scenarioObj[i][t][0]['decelerate']
                    accalare = scenarioObj[i][t][0]['accalare']
                    lanechangspeed = scenarioObj[i][t][0]['lanechangspeed'] # -100: left, -150: right
                    stop = scenarioObj[i][t][0]['stop']
                    turn_left = scenarioObj[i][t][0]['turn_left'] # -200
                    turn_right = scenarioObj[i][t][0]['turn_right'] # -300
                    u_turn = scenarioObj[i][t][0]['u_turn'] # -400

                    list_lane_state = []
                    list_speeds = []
                    # Get action based on situation
                    if self.is_cross_junction(ego.state.transform, npc.state.transform) == 'cross':
                        print("npc cross", npc)
                        if direct == 'left':
                            print("behind left", npc)
                            list_lane_state = [
                                copy.deepcopy([0, 0]),
                                copy.deepcopy([0, 0]),
                                copy.deepcopy([-100, -400, -150]),
                                copy.deepcopy([-100, -400, 0, -150]),
                                copy.deepcopy([-150, -300, 0]),
                                copy.deepcopy([-150, -300, 0]),
                                copy.deepcopy([-150, -300, -100]),
                                copy.deepcopy([-100, -200])
                            ]
                            list_speeds = [
                                copy.deepcopy([accalare, stop]),
                                copy.deepcopy([accalare, decelerate]),
                                copy.deepcopy([lanechangspeed, u_turn, lanechangspeed]),
                                copy.deepcopy([lanechangspeed, u_turn, accalare, lanechangspeed]),
                                copy.deepcopy([lanechangspeed, turn_right, stop]),
                                copy.deepcopy([lanechangspeed, turn_right, decelerate]),
                                copy.deepcopy([lanechangspeed, turn_right, lanechangspeed]),
                                copy.deepcopy([lanechangspeed, turn_left])
                            ]

                        if direct == 'right':
                            print("behind right", npc)
                            list_lane_state = [
                                copy.deepcopy([0, 0]),
                                copy.deepcopy([0, 0]),
                                copy.deepcopy([-100, -400, 0]),
                                copy.deepcopy([-100, -400, 0]),
                                copy.deepcopy([-150, -300, -100, 0]),
                                copy.deepcopy([-150, -300, -100, 0]),
                                copy.deepcopy([-100, -200, 0]),
                                copy.deepcopy([-100, -200, 0]),
                                copy.deepcopy([-100, -200, -150, 0])
                            ]
                            list_speeds = [
                                copy.deepcopy([accalare, stop]),
                                copy.deepcopy([accalare, decelerate]),
                                copy.deepcopy([lanechangspeed, u_turn, stop]),
                                copy.deepcopy([lanechangspeed, u_turn, decelerate]),
                                copy.deepcopy([lanechangspeed, turn_right, lanechangspeed, stop]),
                                copy.deepcopy([lanechangspeed, turn_right, lanechangspeed, decelerate]),
                                copy.deepcopy([lanechangspeed, turn_left, stop]),
                                copy.deepcopy([lanechangspeed, turn_left, decelerate]),
                                copy.deepcopy([lanechangspeed, turn_left, lanechangspeed, accalare])
                            ]
                    elif situation == "SameLaneAhead":
                        list_lane_state = [
                            copy.deepcopy([0]),
                            copy.deepcopy([0]),
                            copy.deepcopy([-100, -150]),
                            copy.deepcopy([-150, -100]),
                        ]
                        list_speeds = [
                            copy.deepcopy([stop]),
                            copy.deepcopy([decelerate]),
                            copy.deepcopy([lanechangspeed, lanechangspeed]),
                            copy.deepcopy([lanechangspeed, lanechangspeed])
                        ]

                    elif situation == "ahead":
                        if direct == 'left':
                            list_lane_state = [
                                copy.deepcopy([-150, 0]),
                                copy.deepcopy([-150, 0]),
                                copy.deepcopy([-150, -100, -150]),
                                copy.deepcopy([-150, -150, -100]),
                                copy.deepcopy([0, -150])
                            ]
                            list_speeds = [
                                copy.deepcopy([lanechangspeed, stop]),
                                copy.deepcopy([lanechangspeed, decelerate]),
                                copy.deepcopy([lanechangspeed, lanechangspeed, lanechangspeed]),
                                copy.deepcopy([lanechangspeed, lanechangspeed, lanechangspeed]),
                                copy.deepcopy([decelerate, lanechangspeed]),
                            ]
                        elif direct == 'right':
                            list_lane_state = [
                                copy.deepcopy([-100, 0]),
                                copy.deepcopy([-100, 0]),
                                copy.deepcopy([-100, -100, -150]),
                                copy.deepcopy([-100, -150, -100]),
                                copy.deepcopy([0, -100])
                            ]
                            list_speeds = [
                                copy.deepcopy([lanechangspeed, stop]),
                                copy.deepcopy([lanechangspeed, decelerate]),
                                copy.deepcopy([lanechangspeed, lanechangspeed, lanechangspeed]),
                                copy.deepcopy([lanechangspeed, lanechangspeed, lanechangspeed]),
                                copy.deepcopy([decelerate, lanechangspeed]),
                            ]
                    elif situation == "parall" or situation == "behind":
                        if direct == 'left':
                            list_lane_state = [
                                copy.deepcopy([0, -150]),
                                copy.deepcopy([0, -150, 0]),
                                copy.deepcopy([0, -150, 0]),
                            ]
                            list_speeds = [
                                copy.deepcopy([accalare, lanechangspeed]),
                                copy.deepcopy([accalare, lanechangspeed, stop]),
                                copy.deepcopy([accalare, lanechangspeed, decelerate]),
                            ]
                        elif direct == 'right':
                            list_lane_state = [
                                copy.deepcopy([0, -100]),
                                copy.deepcopy([0, -100, 0]),
                                copy.deepcopy([0, -100, 0]),
                            ]
                            list_speeds = [
                                copy.deepcopy([accalare, lanechangspeed]),
                                copy.deepcopy([accalare, lanechangspeed, stop]),
                                copy.deepcopy([accalare, lanechangspeed, decelerate]),
                            ]
                    elif situation == "SameLaneBehind":
                        list_lane_state = [
                            copy.deepcopy([0, -100, 0, -150]),
                            copy.deepcopy([0, -150, 0, -100]),
                            copy.deepcopy([-100, 0, -150]),
                            copy.deepcopy([-150, 0, -100]),
                        ]
                        list_speeds = [
                            copy.deepcopy([accalare, lanechangspeed, accalare, lanechangspeed]),
                            copy.deepcopy([accalare, lanechangspeed, accalare, lanechangspeed]),
                            copy.deepcopy([lanechangspeed, accalare, lanechangspeed]),
                            copy.deepcopy([lanechangspeed, accalare, lanechangspeed]),
                        ]

                    len_list = len(list_lane_state)
                    if len_list > 0:
                        index_list = random.randrange(0, len_list)
                        lane_state = list_lane_state[index_list]
                        speeds = list_speeds[index_list]

                    if len(lane_state) < 4:
                        tmpaction = [0] * (4 - len(lane_state))
                        tmpspeed = [1] * (4 - len(lane_state))
                        lane_state += tmpaction
                        speeds += tmpspeed

                # Atom Gene
                else:
                    is_motif = False
                    if isinstance(scenarioObj[i][t][0], list) and isinstance(scenarioObj[i][t][1], list):
                        lane_state = scenarioObj[i][t][1]
                        speeds = scenarioObj[i][t][0]
                    else:
                        tmpaction = [0, 1, 2]
                        tmpspeed = [30] * 3
                        if isinstance(scenarioObj[i][t][1], list):
                            lane_state = scenarioObj[i][t][1]
                            speeds = [scenarioObj[i][t][0]] + [random.randint(0, 2)] * 3
                        elif isinstance(scenarioObj[i][t][0], list):
                            speeds = scenarioObj[i][t][0]
                            lane_state = [scenarioObj[i][t][1]] + [30] * 3
                        else:
                            speeds = [scenarioObj[i][t][0]] + tmpspeed
                            lane_state = [scenarioObj[i][t][1]] + tmpaction

                lane_stateList.append(lane_state)
                speedsList.append(speeds)
                genes.append(is_motif)
                self.npcAction[i] += lane_state
                i += 1

            # Record the min delta D and d
            minDeltaD = float('inf')
            npcDeltaAtTList = [0 for i in range(numOfNpc)]
            egoSpeedList = [0 for i in range(numOfNpc)]
            minD = self.maxint
            npcDAtTList = [0 for i in range(numOfNpc)]

            for x in range(4):
                h = 0
                for npc in npcList:
                    # Execute the action of MotifGene
                    if genes[h] == True:
                        if lane_stateList[h][x] <= -200:
                            print("motif turn trigger!!. NPC number:", h, lane_stateList[h][x])
                            junction_id, near_junction = self.is_in_junction(npc.state.transform)
                            print("junction_id, near_junction", junction_id, near_junction)
                            all_current_lane = self.point_convert_to_lane(npc.transform)

                            if near_junction == 1 and all_current_lane != []:
                                forward = lgsvl.utils.transform_to_forward(npc.transform)
                                right = lgsvl.utils.transform_to_right(npc.transform)

                                all_overlap_lane = lanes_map[all_current_lane[0]]['successor']
                                target_lane_position = None
                                if lane_stateList[h][x] == -200:
                                    target_lane_position = npc.state.position + 20 * forward - 35 * right
                                elif lane_stateList[h][x] == -300:
                                    target_lane_position = npc.state.position + 10 * forward + 35 * right
                                elif lane_stateList[h][x] == -400:
                                    target_lane_position = npc.state.position - 35 * forward - 10 * right

                                target_lane = self.point_convert_to_lane(
                                    sim.map_point_on_lane(target_lane_position)
                                )

                                print(all_overlap_lane, target_lane)

                                min_lane_distane = 100000
                                cur_lane = None

                                for overlap_lane in all_overlap_lane:
                                    next_lane = lanes_map[overlap_lane]['successor']
                                    for each_lane in next_lane:
                                        for t_lane in target_lane:
                                            d_lane_to_lane = self.distane_to_lane(t_lane, each_lane)
                                            print("d_lane_to_lane, next_lane, target_lane, cur_lane", d_lane_to_lane, each_lane,
                                                t_lane, overlap_lane)
                                            if min_lane_distane > d_lane_to_lane:
                                                min_lane_distane = d_lane_to_lane
                                                cur_lane = overlap_lane

                                wp = self.get_way_point_on_lane(cur_lane, speedsList[h][x])
                                npc.follow(wp)
                                print(cur_lane, min_lane_distane)
                            else:
                                npc.follow_closest_lane(True, speedsList[h][x])
                        else:
                            ego_speed = 0
                            if ego.state.speed == 0:
                                ego_speed = 10
                            else:
                                ego_speed = ego.state.speed
                                situation = self.is_within_distance_ahead(npc.state.transform, ego.state.transform)
                                if situation == 'ahead':
                                    ego_speed *= 2.7

                            npc.follow_closest_lane(True, ego_speed * speedsList[h][x])

                            if lane_stateList[h][x] == -100:
                                self.setNpcChangeLane(npc, "LEFT")
                            elif lane_stateList[h][x] == -150:
                                self.setNpcChangeLane(npc, "RIGHT")

                    # Execute the action of AtomGene
                    else:
                        self.setNpcSpeed(npc, speedsList[h][x])
                        turnCommand = lane_stateList[h][x]
                        # <0:no turn 1:left 2:right>
                        if turnCommand == 1:
                            direction = "LEFT"
                            self.setNpcChangeLane(npc, direction)
                        elif turnCommand == 2:
                            direction = "RIGHT"
                            self.setNpcChangeLane(npc, direction)
                    h += 1

                # restart when npc lost
                for j in range(6):
                    k = 0  # k th npc
                    self.egoSpeed.append(ego.state.speed)
                    self.egoLocation.append(ego.state.transform)

                    for npc in npcList:
                        self.npcSpeed[k].append(npc.state.velocity)
                        self.npcLocation[k].append(npc.state.transform)
                        # Update ttc
                        curDeltaD = self.findttc(ego, npc)
                        if minDeltaD > curDeltaD:
                            minDeltaD = curDeltaD
                        npcDeltaAtTList[k] = minDeltaD
                        # Update distance
                        now_situation = self.is_within_distance_ahead(npc.state.transform, ego.state.transform)
                        curD = liability.findDistance(ego, npc)
                        min_situation = now_situation
                        npcth = k

                        if minD > curD:
                            minD = curD
                            min_situation = now_situation
                            npcth = k
                        npcDAtTList[k] = minD
                        minNpcSituation[k] = [minD, min_situation, npcth]
                        k += 1

                    # restart Apollo when ego offline
                    while not ego.bridge_connected:
                        print("+++++++++++++++++++++++=", ego.bridge_connected)
                        # print("----------------------line", line)
                        time.sleep(5)
                        resultDic['ttc'] = ''
                        resultDic['fault'] = ''
                        print(" ---- Bridge is cut off ----")
                        util.print_debug(" ---- Bridge is cut off ----")
                        now = datetime.now()
                        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
                        print(date_time)
                        return resultDic

                    egoSpeedList.append(self.get_speed(ego))
                    egoPathList.append(ego.state.position)

                    sim.run(0.5)
                    # 4 * 0.25 * 12 * 4
            ####################################
            # kth npc
            k = 0
            for npc in npcList:
                deltaDList[k][t] = npcDeltaAtTList[k]
                dList[k][t] = npcDAtTList[k]

                MinNpcSituations[k][t] = minNpcSituation[k]

                k += 1

        # Record scenario
        ttc = self.findFitness(deltaDList, dList, self.isEgoFault, self.isHit, hitTime)
        print("ttc", ttc)


        resultDic['ttc'] = -ttc
        resultDic['smoothness'] = self.jerk(egoSpeedList)
        # resultDic['pathSimilarity'] = self.findPathSimilarity(egoPathList, localPath)
        resultDic['MinNpcSituations'] = MinNpcSituations
        resultDic['egoSpeed'] = self.egoSpeed
        resultDic['egoLocation'] = self.egoLocation
        resultDic['npcSpeed'] = self.npcSpeed
        resultDic['npcLocation'] = self.npcLocation
        resultDic['npcAction'] = self.npcAction
        resultDic['isCollision'] = self.isCollision
        resultDic['fault'] = ''

        if self.isEgoFault:
            resultDic['fault'] = 'ego'
        util.print_debug(" === Finish simulation === ")
        util.print_debug(resultDic)

        return resultDic

    def runSimulation(self, GaData, isRstart):
        """
        run simulation of scenario
        """
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        util.print_debug("\n === Run simulation === [" + date_time + "]")

        ego = self.ego

        # Print ego position
        print("====== Ego Position ======")
        print(ego.state.position.x, ego.state.position.z, ego.state.rotation.y, ego.state.rotation.x)

        ge = MultiObjGenticAlgorithm(GaData.bounds, GaData.mutationProb, GaData.crossoverProb,
                                     GaData.popSize, GaData.numOfNpc, GaData.numOfTimeSlice, GaData.maxGen)

        if isRstart:
            print("is restart")
            ge.set_checkpoint('GaCheckpointsCrossroads/last_gen.obj')

        # check ck_path
        print("---- Check any ge checkpoint path ----")
        if ge.ck_path is not None:
            print(ge.ck_path)

        # load last iteration
        if ge.ck_path is not None:
            if os.path.exists(ge.ck_path):
                ck = open(ge.ck_path, 'rb')
                ge.pop = pickle.load(ck)
                ck.close()
                paths = [f for f in listdir('GaCheckpointsCrossroads')]
                alIterations = len(paths) - 2
                ge.max_gen -= alIterations
        # after restart
        else:
            i = 0
            while i < ge.pop_size:
                print("-------------------scenario: {i}th ---------".format(i=i))
                print("ge pop size current", len(ge.pop))
                chromsome = MutlChromosome(ge.bounds, ge.NPC_size, ge.time_size, None)
                chromsome.rand_init()

                # result1 = DotMap()
                util.print_debug("scenario::" + str(chromsome.scenario))
                result1 = self.runGen(chromsome.scenario, chromsome.weathers)
                print(result1['ttc'], result1['fault'])
                if result1 is None or result1['ttc'] == 0.0 or result1['ttc'] == '':
                    continue
                i += 1

                chromsome.ttc = result1['ttc']
                chromsome.MinNpcSituations = result1['MinNpcSituations']
                chromsome.smoothness = result1['smoothness']
                # chromsome.pathSimilarity = result1['pathSimilarity']
                chromsome.egoSpeed = result1['egoSpeed']
                chromsome.egoLocation = result1['egoLocation']
                chromsome.npcSpeed = result1['npcSpeed']
                chromsome.npcLocation = result1['npcLocation']
                chromsome.isCollision = result1['isCollision']
                chromsome.npcAction = result1['npcAction']
                ge.pop.append(chromsome)

                if self.isCollision:
                    self.isCollision = 0

                    print("============= Collision =============")
                    util.print_debug(" ***** Found an accident where ego is at fault ***** ")
                    # Dump the scenario where causes the accident
                    if os.path.exists('AccidentScenarioCrossroads') == False:
                        os.mkdir('AccidentScenarioCrossroads')
                    now = datetime.now()
                    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
                    ckName = 'AccidentScenarioCrossroads/accident-gen' + '-' + date_time
                    # if lisFlag == True:
                    #     ckName = ckName + "-LIS"
                    a_f = open(ckName, 'wb')
                    pickle.dump(chromsome, a_f)
                    a_f.truncate()
                    a_f.close()

            ge.take_checkpoint(ge.pop, 'last_gen.obj')

        # iteration of scenario
        print("ge.max_gen", ge.max_gen)
        for i in range(ge.max_gen):
            util.print_debug(" \n\n*** " + str(i) + "th generation ***")
            util.select(" \n\n*** " + str(i) + "th generation ***")
            ge.touched_chs = []
            ge.beforePop = []
            ge.pop_size = len(ge.pop)
            for t in range(len(ge.pop)):
                ge.beforePop.append(copy.deepcopy(ge.pop[t]))
            ge.cross()
            print("-----cross done----")
            ge.mutation()
            print("-----mutation done----")
            indexChs = 0
            for eachChs in ge.touched_chs:
                indexChs += 1
                print("eachChs processing...", indexChs, len(ge.touched_chs))
                res = self.runGen(eachChs.scenario, eachChs.weathers)
                if res is None or res['ttc'] == 0.0 or res['ttc'] == '':
                    continue

                eachChs.ttc = res['ttc']
                eachChs.MinNpcSituations = res['MinNpcSituations']
                eachChs.smoothness = res['smoothness']
                # eachChs.pathSimilarity = res['pathSimilarity']
                eachChs.egoSpeed = res['egoSpeed']
                eachChs.egoLocation = res['egoLocation']
                eachChs.npcSpeed = res['npcSpeed']
                eachChs.npcLocation = res['npcLocation']
                eachChs.isCollision = res['isCollision']
                eachChs.npcAction = res['npcAction']

                if self.isCollision:
                    self.isCollision = 0
                    print("=============Collision=============")
                    util.print_debug(" ***** Found an accident where ego is at fault ***** ")
                    # Dump the scenario where causes the accident
                    if os.path.exists('AccidentScenarioCrossroads') == False:
                        os.mkdir('AccidentScenarioCrossroads')
                    now = datetime.now()
                    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
                    ckName = 'AccidentScenarioCrossroads/accident-gen' + '-' + date_time
                    a_f = open(ckName, 'wb')
                    pickle.dump(eachChs, a_f)
                    a_f.truncate()
                    a_f.close()

            flag = []
            for m in range(len(ge.beforePop)):
                for h in range(len(ge.pop)):
                    if ge.beforePop[m].scenario == ge.pop[h].scenario:
                        if m not in flag:
                            flag.append(m)

            flag.reverse()

            for index in flag:
                ge.beforePop.pop(index)
            ge.pop += ge.beforePop
            ge.select_NDsort_roulette()

            N_generation = ge.pop
            util.select(str(N_generation))
            N_b = ge.g_best  # Record the scenario with the best score over all generations

            # Save last iteration
            ge.take_checkpoint(N_generation, 'last_gen.obj')

            now = datetime.now()
            date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
            ge.take_checkpoint(N_generation, 'generation-' + '-at-' + date_time)


##################################### MAIN ###################################

# Read scenario obj
objPath = sys.argv[1]
resPath = sys.argv[2]

objF = open(objPath, 'rb')
scenarioObj = pickle.load(objF)
objF.close()

resultDic = {}

try:
    sim = LgApSimulation()
    resultDic = sim.runSimulation(scenarioObj[0], scenarioObj[1])
except Exception as e:
    util.print_debug(e.message)
    resultDic['ttc'] = ''
    resultDic['fault'] = ''

# Send fitness score int object back to ge
if os.path.isfile(resPath) == True:
    os.system("rm " + resPath)
f_f = open(resPath, 'wb')
pickle.dump(resultDic, f_f)
f_f.truncate()
f_f.close()