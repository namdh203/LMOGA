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

from MutlChromosome import MutlChromosome

import json

# 70.31.197.180:41343 -> 22/tcp
# 70.31.197.180:41175 -> 5900/tcp
# 70.31.197.180:41196 -> 5901/tcp
# 70.31.197.180:41212 -> 6666/tcp
# 70.31.197.180:41352 -> 8888/tcp
# 70.31.197.180:41069 -> 8966/tcp
# 70.31.197.180:41052 -> 9090/tcp

APOLLO_HOST = "70.31.197.180"  # or 'localhost'
PORT = 8977
DREAMVIEW_PORT = 41352
BRIDGE_PORT = 41052
time_offset = 9

class LgApSimulation:
    def __init__(self):
        self.SIMULATOR_HOST = os.environ.get("SIMULATOR_HOST", "127.0.0.1")
        self.SIMULATOR_PORT = int(os.environ.get("SIMULATOR_PORT", PORT))
        self.BRIDGE_HOST = os.environ.get("BRIDGE_HOST", APOLLO_HOST)
        self.BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", BRIDGE_PORT))
        self.totalSimTime = 15
        # self.bridgeLogPath = "/home/kasm_user/apollo/data/log/cyber_bridge.INFO
        self.sim = None
        self.ego = None  # There is only one ego
        self.initEvPos = lgsvl.Vector(769, 10, -40)
        self.endEvPos = lgsvl.Vector(-847.312927246094, 10, 176.858657836914)
        # self.mapName = "12da60a7-2fc9-474d-a62a-5cc08cb97fe8"
        self.mapName = "bd77ac3b-fbc3-41c3-a806-25915c777022"
        self.roadNum = 2
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
        self.saveState = {}
        self.chooseStateJson = False

    def get_speed(self, vehicle):
        vel = vehicle.state.velocity
        return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def initSimulator(self):
        sim = lgsvl.Simulator(self.SIMULATOR_HOST, self.SIMULATOR_PORT)
        self.sim = sim

    def loadMap(self):
        sim = self.sim
        if sim.current_scene == self.mapName:
            sim.reset()
        else:
            sim.load(self.mapName)

    def initEV(self):
        sim = self.sim
        egoState = lgsvl.AgentState()
        spawn = sim.get_spawn()
        if self.roadNum == 1:
            if self.mapName == "bd77ac3b-fbc3-41c3-a806-25915c777022":      #tartu
                self.initEvPos = lgsvl.Vector(184.9, 35.7, 93.4)
                self.endEvPos = lgsvl.Vector(312.4, 36.7, 312.3)
                egoState.transform.rotation.y = 49
                egoState.transform.rotation.x = 360
            elif self.mapName == "12da60a7-2fc9-474d-a62a-5cc08cb97fe8":    #sanfrancisco
                self.initEvPos = lgsvl.Vector(-328.1, 10.2, 45.5)
                self.endEvPos = lgsvl.Vector(-445.7, 10.2, -22.7)
                egoState.transform.rotation.y = 81
                egoState.transform.rotation.x = 0
            elif self.mapName == "aae03d2a-b7ca-4a88-9e41-9035287a12cc":    #BorregasAve
                self.initEvPos = lgsvl.Vector(352.4, -7.6, -22.6)
                self.endEvPos = lgsvl.Vector(-16.6, -1.9, -49.3)
                egoState.transform.rotation.y = 195
                egoState.transform.rotation.x = 360
        elif self.roadNum == 2:
            if self.mapName == "bd77ac3b-fbc3-41c3-a806-25915c777022":      #tartu
                self.initEvPos = lgsvl.Vector(-76.0, 34.3, 325.3)
                self.endEvPos = lgsvl.Vector(93.8, 34.5, 382.2)
                egoState.transform.rotation.y = 155
                egoState.transform.rotation.x = 359
            if self.mapName == "12da60a7-2fc9-474d-a62a-5cc08cb97fe8":      #sanfrancisco
                self.initEvPos = lgsvl.Vector(-62.7, 10.2, -110.2)
                self.endEvPos = lgsvl.Vector(-208.2, 10.2, -181.6)
                egoState.transform.rotation.y = 224
                egoState.transform.rotation.x = 0
            elif self.mapName == "aae03d2a-b7ca-4a88-9e41-9035287a12cc":    #BorregasAve
                self.initEvPos = lgsvl.Vector(-40.3, -1.4, -11.8)
                self.endEvPos = lgsvl.Vector(348.2, -7.5, -64.4)
                egoState.transform.rotation.y = 105
                egoState.transform.rotation.x = 1
        elif self.roadNum == 3:
            if self.mapName == "12da60a7-2fc9-474d-a62a-5cc08cb97fe8":
                self.initEvPos = lgsvl.Vector(-62.7, 10.2, -110.2)
                self.endEvPos = lgsvl.Vector(-208.2, 10.2, -181.6)
                egoState.transform.rotation.y = 224
                egoState.transform.rotation.x = 0

        egoState.transform = sim.map_point_on_lane(self.initEvPos)
        forward = lgsvl.utils.transform_to_forward(egoState.transform)
        egoState.velocity = 3 * forward
        ego = sim.add_agent("8e776f67-63d6-4fa3-8587-ad00a0b41034", lgsvl.AgentType.EGO,
                            egoState)
        self.ego = ego
        sim.set_time_of_day((10 + time_offset) % 24, fixed=True)

    def connectEvToApollo(self):
        ego = self.ego
        ego.connect_bridge(self.BRIDGE_HOST, self.BRIDGE_PORT)
        while not ego.bridge_connected:
            time.sleep(1)
        print("Bridge connected")
        print(self.BRIDGE_HOST)
        # Dreamview setup
        dv = lgsvl.dreamview.Connection(self.sim, ego, APOLLO_HOST, str(DREAMVIEW_PORT))
        spawns = self.sim.get_spawn()

        modules = [
            'Localization',
            'Perception',
            'Transform',
            'Routing',
            'Prediction',
            'Planning',
            # 'Camera',
            # 'Traffic Light',
            'Control'
        ]

        dv.set_destination(self.endEvPos.x, self.endEvPos.z, 0, CoordType.Unity)
        time.sleep(5)

    def restartLGSVL(self):
        self.initSimulator()
        self.loadMap()
        self.initEV()
        self.connectEvToApollo()

    def addNpcVehicle(self, posVector, vehicleType="SUV"):
        sim = self.sim
        npcList = self.npcList
        npcState = lgsvl.AgentState()
        npcState.transform = sim.map_point_on_lane(posVector)
        npc = sim.add_agent(vehicleType, lgsvl.AgentType.NPC, npcState)
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
        x2 = npc.state.transform.position.x - ego.state.transform.position.x
        z2 = npc.state.transform.position.z - ego.state.transform.position.z
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

        ttc = math.sqrt(pre_x ** 2 + pre_z ** 2) / ego_speed
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
            return "OneLaneBefore"
        elif d_angle < 37.0:
            return 'before'
        elif 37.0 <= d_angle <= 143:
            return 'parall'
        elif d_angle == 180:
            return "OneLaneAfter"
        else:
            return 'after'

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

    def initNpcVehicles(self, numOfNpc):
        # Set initial position of npc vehicle
        ego = self.ego
        npcPosition = []
        npcDetail = []
        for npcs in range(numOfNpc):
            row = random.randint(-1, 1)
            col = random.randint(-1, 1)
            npcDetail.append([row, col])

        for i in range(len(npcDetail)):
            npc_x = ego.state.position.x
            npc_y = ego.state.position.y
            npc_z = ego.state.position.z
            if npcDetail[i][0] == -1:
                npc_x -= 6

                if npcDetail[i][1] == -1:
                    npc_z -= random.uniform(7.5, 15)
                elif npcDetail[i][1] == 1:
                    npc_z += random.uniform(7.5, 15)
                # else:
                #     npc_z += random.uniform(7.5, 15)
            elif npcDetail[i][0] == 0:
                if npcDetail[i][1] == -1:
                    npc_z -= random.uniform(7.5, 15)
                elif npcDetail[i][1] == 1:
                    npc_z += random.uniform(7.5, 15)
                else:
                    npc_z += random.uniform(7.5, 15)
            elif npcDetail[i][0] == 1:
                npc_x += 6
                if npcDetail[i][1] == -1:
                    npc_z -= random.uniform(7.5, 15)
                elif npcDetail[i][1] == 1:
                    npc_z += random.uniform(7.5, 15)
                # else:
                #     npc_z += random.uniform(7.5, 15)

            for npcLocate in npcPosition:
                if abs(npc_x - ego.state.position.x) <= 1.5:
                    if npcDetail[i][0] == -1:
                        npc_x -= 6
                    elif npcDetail[i][0] == 1 or npcDetail[i][0] == 0:
                        npc_x += 6
                    if abs(npc_z - ego.state.position.z) <= 7.5:
                        if npcDetail[i][1] == -1:
                            npc_z -= 7.5
                        elif npcDetail[i][1] == 1 or npcDetail[i][1] == 0:
                            npc_z += 7.5

                if abs(npc_x - npcLocate[0]) <= 6:
                    if npcDetail[i][0] == -1:
                        npc_x -= 8
                    elif npcDetail[i][0] == 1 or npcDetail[i][0] == 0:
                        npc_x += 8

                    if abs(npc_z - npcLocate[2]) <= 5:
                        if npcDetail[i][1] == -1:
                            npc_z -= 7.5
                        elif npcDetail[i][1] == 1 or npcDetail[i][1] == 0:
                            npc_z += 7.5
            npcPosition.append([npc_x, npc_y, npc_z])

        for position in npcPosition:
            self.addNpcVehicle(lgsvl.Vector(position[0], position[1], position[2]))


    def runGen(self, scenarioObj, weather):
        """
        parse the chromosome
        """

        # initialize simulation
        self.initSimulator()

        sim = self.sim
        ego = self.ego

        numOfTimeSlice = len(scenarioObj[0])
        numOfNpc = len(scenarioObj)

        # initialize npc vehicles
        self.initNpcVehicles(numOfNpc)

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

        print("numOfTimeSlice", numOfTimeSlice)

        # Execute scenarios based on genes
        for t in range(0, int(numOfTimeSlice)):
            print("t current", t)
            minNpcSituation = [0 for o in range(numOfNpc)]
            lane_stateList = []
            speedsList = []

            i = 0
            for npc in npcList:
                print("npclist", npcList)
                # print("scenario", scenarioObj)
                # print("scenarioObj[{i}][{t}][0]".format(i=i, t=t), scenarioObj[i][t][0])

                # Motif Gene
                if isinstance(scenarioObj[i][t][0], dict):
                    lane_state = []
                    speeds = []
                    situation = self.is_within_distance_ahead(npc.state.transform, ego.state.transform)
                    Command = scenarioObj[i][t][1]
                    decelerate = scenarioObj[i][t][0]['decelerate']
                    accalare = scenarioObj[i][t][0]['accalare']
                    lanechangspeed = scenarioObj[i][t][0]['lanechangspeed']
                    stop = scenarioObj[i][t][0]['stop']

                    # Get action based on situation
                    if Command == 0:
                        if situation == "OneLaneBefore":
                            lane_state = copy.deepcopy([0])
                            speeds = copy.deepcopy([decelerate])
                        elif situation == "before":
                            lane_state = copy.deepcopy([0, -100])
                            speeds = copy.deepcopy([decelerate, lanechangspeed])
                        elif situation == "parall":
                            lane_state = copy.deepcopy([-100])
                            speeds = copy.deepcopy([lanechangspeed])
                        elif situation == "after":
                            lane_state = copy.deepcopy([0, -100])
                            speeds = copy.deepcopy([accalare, lanechangspeed])
                        elif situation == "OneLaneAfter":
                            lane_state = copy.deepcopy([-100, 0, -100])
                            speeds = copy.deepcopy([lanechangspeed, accalare, lanechangspeed])
                    elif Command == 1:
                        if situation == "OneLaneBefore":
                            lane_state = copy.deepcopy([0, 0])
                            speeds = copy.deepcopy([decelerate, stop])
                        elif situation == "before":
                            lane_state = copy.deepcopy([0, -100, 0])
                            speeds = copy.deepcopy([decelerate, lanechangspeed, stop])
                        elif situation == "parall":
                            lane_state = copy.deepcopy([-100, 0])
                            speeds = copy.deepcopy([lanechangspeed, stop])
                        elif situation == "after":
                            lane_state = copy.deepcopy([0, -100, 0])
                            speeds = copy.deepcopy([accalare, lanechangspeed, stop])
                        elif situation == "OneLaneAfter":
                            lane_state = copy.deepcopy([-100, 0, -100, 0])
                            speeds = copy.deepcopy([lanechangspeed, accalare, lanechangspeed, stop])
                    elif Command == 2:
                        if situation == "OneLaneBefore":
                            lane_state = copy.deepcopy([0, 0])
                            speeds = copy.deepcopy([decelerate, decelerate])
                        elif situation == "before":
                            lane_state = copy.deepcopy([0, -100, 0])
                            speeds = copy.deepcopy([decelerate, lanechangspeed, decelerate])
                        elif situation == "parall":
                            lane_state = copy.deepcopy([-100, 0])
                            speeds = copy.deepcopy([lanechangspeed, decelerate])
                        elif situation == "after":
                            lane_state = copy.deepcopy([0, -100, 0])
                            speeds = copy.deepcopy([accalare, lanechangspeed, decelerate])
                        elif situation == "OneLaneAfter":
                            lane_state = copy.deepcopy([-100, 0, -100, 0])
                            speeds = copy.deepcopy([lanechangspeed, accalare, lanechangspeed, decelerate])
                    elif Command == 3:
                        if situation == "OneLaneBefore":
                            lane_state = copy.deepcopy([0, -100])
                            speeds = copy.deepcopy([accalare, lanechangspeed])
                        else:
                            lane_state = copy.deepcopy([0])
                            speeds = copy.deepcopy([accalare])
                    elif Command == 4:
                        if situation == "parall" or situation == "before":
                            lane_state = copy.deepcopy([-100, -100])
                            speeds = copy.deepcopy([lanechangspeed, lanechangspeed])
                        elif situation == "after":
                            lane_state = copy.deepcopy([0, -100, -100])
                            speeds = copy.deepcopy([accalare, lanechangspeed, lanechangspeed])
                        else:
                            lane_state = copy.deepcopy([0])
                            speeds = copy.deepcopy([accalare])

                    if len(lane_state) < 4:
                        tmpaction = [0] * (4 - len(lane_state))
                        tmpspeed = [1] * (4 - len(lane_state))
                        lane_state += tmpaction
                        speeds += tmpspeed

                # Atom Gene
                else:
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
                    if speedsList[h][0] + speedsList[h][1] + speedsList[h][2] + speedsList[h][3] < 8:
                        if ego.state.speed == 0:
                            ego_speed = 10
                        else:
                            ego_speed = ego.state.speed
                            situation = self.is_within_distance_ahead(npc.state.transform, ego.state.transform)
                            if situation == 'after':
                                ego_speed *= 1.5 * 1.8

                        self.setNpcSpeed(npc, ego_speed * speedsList[h][x])

                        lane_change = 0
                        if lane_stateList[h][x] == -100:
                            if self.is_within_distance_right(ego.state.transform, npc.state.transform) == 'left':
                                lane_change += -1
                            else:
                                lane_change += 1

                        if lane_change == -1:
                            direction = "LEFT"
                            self.setNpcChangeLane(npc, direction)
                        elif lane_change == 1:
                            direction = "RIGHT"
                            self.setNpcChangeLane(npc, direction)
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
                    totalDistances = 0
                    # forward = lgsvl.utils.transform_to_forward(ego.state.transform)
                    minDistances = 999
                    for npc in npcList:
                        minDistances = min(minDistances, math.sqrt(
                            (npc.state.transform.position.x - ego.state.transform.position.x) ** 2 +
                            (npc.state.transform.position.z - ego.state.transform.position.z) ** 2))
                        # print("npc postion", npc.state.transform.position.x, npc.state.transform.position.z)
                    print("(j, minDistances)", j, minDistances)
                    # print("ego position", ego.state.transform.position.x, ego.state.transform.position.z)
                    if minDistances > 130:
                        resultDic['ttc'] = ''
                        resultDic['fault'] = 'npcTooLong'
                        print("npc too long detected")
                        print("ego position", ego.state.transform.position.x, ego.state.transform.position.z)
                        return resultDic
                    k = 0  # k th npc
                    self.egoSpeed.append(ego.state.speed)
                    # print("egoSpeed list", self.egoSpeed[-1])
                    self.egoLocation.append(ego.state.transform)

                    if len(self.egoLocation) >= 36:
                        if math.sqrt((self.egoLocation[-1].position.x - self.egoLocation[0].position.x) ** 2 +
                                    (self.egoLocation[-1].position.z - self.egoLocation[0].position.z) ** 2) <= 20:
                            print("ego position", ego.state.transform.position.x, ego.state.transform.position.z)
                            print("ego stop too long...")
                            resultDic['ttc'] = ''
                            resultDic['fault'] = ''
                            return resultDic

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
                    # if math.sqrt((ego.state.position.x - self.endEvPos.x) ** 2 + (
                    #         ego.state.position.z - self.endEvPos.z) ** 2) <= 7:
                    #     print("Reach Destinaiton!!!")
                    #     print("ego position", ego.state.transform.position.x, ego.state.transform.position.z)
                    #     resultDic['ttc'] = ''
                    #     resultDic['fault'] = ''
                    #     return resultDic

                    sim.run(0.5)
                    time.sleep(0.1)
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
        # Set signal loop
        controllables = self.sim.get_controllables("signal")
        for c in controllables:
            signal = self.sim.get_controllable(c.transform.position, "signal")
            control_policy = "trigger=200;green=50;yellow=0;red=0;loop"
            signal.control(control_policy)

        # Print ego position
        print("====== Ego Position ======")
        print(ego.state.position.x, ego.state.position.z, ego.state.rotation.y, ego.state.rotation.x)


        def on_collision(agent1, agent2, contact):
            """
            collision listener function
            """
            print("ego collision")
            self.isCollision += 1

            if agent2 is None or agent1 is None:
                self.isEgoFault = True
                util.print_debug(" --- Hit road obstacle --- ")
                return

            apollo = agent1
            npcVehicle = agent2
            if agent2.name == "8e776f67-63d6-4fa3-8587-ad00a0b41034":
                apollo = agent2
                npcVehicle = agent1

            util.print_debug(" --- On Collision, ego speed: " + str(apollo.state.speed) + ", NPC speed: " + str(
                npcVehicle.state.speed))
            if apollo.state.speed <= 0.005:
                self.isEgoFault = False
                return

        print("self is collison***", self.isCollision)

        ego.on_collision(on_collision)

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