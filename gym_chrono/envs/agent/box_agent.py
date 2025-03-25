# =======================================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2021 projectchrono.org
# All right reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# ========================================================================================================
# Authors: Huzaifa Unjhawala (refactored from original code by Simone Benatti, Aaron Young, Asher Elmquist)
# ========================================================================================================
import gymnasium as gym
import numpy as np
import math
import os
from gym_chrono.envs.utils.terrain_utils import SCMParameters
from gym_chrono.envs.utils.perlin_bitmap_generator import generate_random_bitmap
from gym_chrono.envs.utils.asset_utils import *
from gym_chrono.envs.utils.utils import CalcInitialPose, chVector_to_npArray, npArray_to_chVector, SetChronoDataDirectories
from gym_chrono.envs.ChronoBase import ChronoBaseEnv
import pychrono.vehicle as veh
import pychrono as chrono
from typing import Any

try:
    from pychrono import irrlicht as chronoirr
except:
    print('Could not import ChronoIrrlicht')
try:
    import pychrono.sensor as sens
except:
    print('Could not import Chrono Sensor')

try:
    from pychrono import irrlicht as chronoirr
except:
    print('Could not import ChronoIrrlicht')


# Bunch of utilities required for the environment
# Standard Python imports

# Gymnasium imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))

class box_agent(ChronoBaseEnv):

    # Supported render modes
    # Human - Render birds eye vier of the vehicle
    metadata = {'additional_render.modes': ['agent_pov', 'None']}

    def __init__(self, additional_render_mode='None'):
        # Check if render mode is suppoerted
        if additional_render_mode not in box_agent.metadata['additional_render.modes']:
            raise Exception(
                f'Render mode: {additional_render_mode} not supported')
        ChronoBaseEnv.__init__(self, additional_render_mode)

        # Ser the Chrono data directories for all the assest information
        SetChronoDataDirectories()

        # -------------------------------
        # Action and Observation Space
        # -------------------------------

        # Set camera frame as this is the observation
        self.image_width = 640
        self.image_height = 480
        self.update_rate = 30
        self.fov = 1.408

        # Observation space has 2 components
        # 1. Camera image (RGB) of size (m_camera_width, m_camera_height)
        # 2. Vehicle state relative to the goal of size (5,)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(
                3, self.image_height, self.image_width), dtype=np.uint8),
            "data": gym.spaces.Box(low=-100, high=100, shape=(5,), dtype=np.float32)})

        # Action space is the steering, throttle and braking where
        # Steering is between -1 and 1
        # Throttle is between -1 and 1, negative is braking
        # This is done to aide training - part of recommended rl tips to have symmetric action space
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32)
        # -------------------------------
        # Simulation specific class variables
        # -------------------------------
        self.m_assets = None  # List of assets in the simulation
        self.m_system = None  # Chrono system
        self.virtual_robot = None  # Vehicle set in reset method
        self.m_vehicle_pos = None  # Vehicle position
        self.m_driver = None  # Driver set in reset method
        self.m_driver_input = None  # Driver input set in reset method
        self.m_chassis = None  # Chassis body of the vehicle
        self.m_chassis_body = None  # Chassis body of the vehicle
        self.m_chassis_collision_box = None  # Chassis collision box of the vehicle
        self.m_proper_collision = True
        # Initial location and rotation of the vehicle
        self.m_initLoc = None
        self.m_initRot = None
        self.m_contact_force = None  # Contact force on the vehicle

        # Control and dynamics frequency
        self.m_control_frequency = 10  # Control frequency of the simulation
        self.m_step_size = 1e-3  # Step size of the simulation
        self.m_steps_per_control = round(
            1 / (self.m_step_size * self.m_control_frequency))

        self.m_steeringDelta = 0.05  # At max the steering can change by 0.05 in 0.1 seconds
        self.m_throttleDelta = 0.1
        self.m_brakingDelta = 0.1

        # Terrrain
        self.m_terrain = None  # Actual deformable terrain
        self.m_min_terrain_height = -5  # min terrain height
        self.m_max_terrain_height = 5  # max terrain height
        self.m_terrain_length = 80.0  # size in X direction
        self.m_terrain_width = 80.0  # size in Y direction
        self.m_assets = []
        self.m_positions = []
        # Sensor manager
        self.m_sens_manager = None  # Sensor manager for the simulation
        self.m_have_camera = False  # Flag to check if camera is present
        self.m_camera = None  # Camera sensor
        self.m_have_gps = False
        self.m_gps = None  # GPS sensor
        self.m_gps_origin = None  # GPS origin
        self.m_have_imu = False
        self.m_imu = None  # IMU sensor
        self.m_imu_origin = None  # IMU origin
        self.m_camera_frequency = 20
        self.m_gps_frequency = 10
        self.m_imu_frequency = 100

        # -------------------------------
        # Gym Env specific parameters
        # -------------------------------
        self.m_max_time = 20  # Max time for each episode
        self.m_reward = 0  # Reward for the episode
        self.m_debug_reward = 0  # Debug reward for the episode
        # Reward helpers
        self.m_action = None  # Action taken by the agent
        self.m_old_action = None  # Action taken by the agent at previous time step
        # Position of goal as numpy array
        self.m_goal = None
        # Distance to goal at previos time step -> To gauge "progress"
        self.m_vector_to_goal = None
        self.m_vector_to_goal_noNoise = None
        self.m_old_distance = None
        # Observation of the environment
        self.m_observation = None
        # Flag to determine if the environment has terminated -> In the event of timeOut or reach goal
        self.m_terminated = False
        # Flag to determine if the environment has truncated -> In the event of a crash
        self.m_truncated = False
        # Flag to check if the render setup has been done -> Some problem if rendering is setup in reset
        self.m_render_setup = False
        # Flag to count success while testing
        self.m_success = False
        # Flag to check if there is a plan to render or not
        self.m_play_mode = False
        self.m_additional_render_mode = additional_render_mode

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state -> Set up for standard gym API
        :param seed: Seed for the random number generator
        :param options: Options for the simulation (dictionary)
        """
        # -------------------------------
        # Reset Chrono system
        # -------------------------------
        self.m_system = chrono.ChSystemNSC()
        self.m_system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        self.m_system.SetCollisionSystemType(
            chrono.ChCollisionSystem.Type_BULLET)

        # -------------------------------
        # Reset the terrain
        # -------------------------------
        self.m_isFlat = True
        terrain_delta = 0.05
        self.m_isRigid = True

        mmesh = chrono.ChTriangleMeshConnected()
        mmesh.LoadWavefrontMesh(
            project_root + '/envs/data/environment/new_flat_3/new_flat_3.obj', False, True)
        # scale to a different size
        # mmesh.Transform(chrono.ChVector3d(0, 0, 0), chrono.ChMatrix33d(2))

        trimesh_shape = chrono.ChVisualShapeTriangleMesh()
        trimesh_shape.SetMesh(mmesh)
        trimesh_shape.SetName("ENV MESH")
        trimesh_shape.SetMutable(False)

        mesh_body = chrono.ChBody()
        mesh_body.SetPos(chrono.ChVector3d(0, 0, 0))
        mesh_body.SetRot(chrono.Q_ROTATE_Y_TO_Z)
        mesh_body.AddVisualShape(trimesh_shape)
        mesh_body.SetFixed(True)
        self.m_system.Add(mesh_body)


        print("added environment")
        # -------------------------------
        # Reset the vehicle
        # -------------------------------

        patch_mat = chrono.ChContactMaterialNSC()
        self.virtual_robot = chrono.ChBodyEasyBox(
            0.25, 0.25, 0.5, 100, True, True, patch_mat)
        self.virtual_robot.SetPos(chrono.ChVector3d(-1.25, -1.25, 0.25))
        robot_theta = self.initialize_agent_pos(seed)

        print("added robot")

        # -------------------------------
        # Set the goal point
        # -------------------------------
        self.set_goal(seed)
        print("set goal")

        # -------------------------------
        # Reset the obstacles
        # -------------------------------
        # self.add_obstacles(proper_collision=False)

        # -------------------------------
        # Initialize the sensors
        # -------------------------------
        del self.m_sens_manager
        self.m_sens_manager = sens.ChSensorManager(self.m_system)
        # Set the lighting scene
        self.m_sens_manager.scene.AddPointLight(chrono.ChVector3f(
            100, 100, 100), chrono.ChColor(1, 1, 1), 5000.0)

        print("light added")

        # Add all the sensors -> For now orientation is ground truth
        # self.add_sensors(camera=True, gps=True, imu=False)
        

        offset_pose = chrono.ChFramed(chrono.ChVector3d(0.3, 0, 0.25), chrono.QUNIT)
        print("line 248")
        print(self.virtual_robot)
        print(self.update_rate)
        print(offset_pose)
        print(self.image_width)
        print(self.image_height)
        print(self.fov)
    
        self.m_camera = sens.ChCameraSensor(
            self.virtual_robot,  # body camera is attached to
            self.update_rate,  # update rate in Hz
            offset_pose,  # offset pose
            self.image_width,  # image width
            self.image_height,  # image height
            self.fov,
            6
        )

        print("sensors added")
        # -------------------------------
        # Get the initial observation
        # -------------------------------
        self.m_observation = self.get_observation()
        self.m_old_distance = self.m_vector_to_goal.Length()
        self.m_old_action = np.zeros((2,))
        self.m_contact_force = 0
        self.m_debug_reward = 0
        self.m_reward = 0
        self.m_render_setup = False

        self.m_terminated = False
        self.m_truncated = False
        return self.m_observation, {}

    def step(self, action):
        """
        Box Agent takes a step in the environment - Frequency by default is 10 Hz
        """
        # steering = action[0]
        
        # Move robot forward in the direction it is facing
        if (action == 1): # move forward
            self.virtual_robot.SetPos(self.virtual_robot.GetPos() + chrono.ChVectorD(0, 0, 0.1))
        elif (action == 2): # turn left
            self.virtual_robot.SetRot(self.virtual_robot.GetRot() * chrono.Q_from_AngZ(0.1))
        elif (action == 3): # turn right
            self.virtual_robot.SetRot(self.virtual_robot.GetRot() * chrono.Q_from_AngZ(-0.1))
        elif (action == 4): # reached goal
            pass

        # This is used in the reward function
        self.m_action = action

        # Update the sensor manager
        self.m_sens_manager.Update()

        # Get the observation
        self.m_observation = self.get_observation()
        self.m_reward = self.get_reward()
        self.m_debug_reward += self.m_reward

        # Check if we hit something or reached the goal
        self._is_terminated()
        self._is_truncated()

        return self.m_observation, self.m_reward, self.m_terminated, self.m_truncated, {}

    def render(self, mode='human'):
        """
        Render the environment
        """

        # ------------------------------------------------------
        # Add visualization - only if we want to see "human" POV
        # ------------------------------------------------------
        if mode == 'human':
            self.render_mode = 'human'

            if self.m_render_setup == False:
                self.vis = chronoirr.ChVisualSystemIrrlicht()
                self.vis.AttachSystem(self.m_system)
                self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
                self.vis.SetWindowSize(1280, 720)
                self.vis.SetWindowTitle('Box Agent')
                self.vis.Initialize()
                self.vis.AddSkyBox()
                self.vis.AddCamera(chrono.ChVector3d(
                    0, 0, 80), chrono.ChVector3d(0, 0, 1))
                self.vis.AddTypicalLights()
                self.vis.AddLightWithShadow(chrono.ChVector3d(
                    1.5, -2.5, 5.5), chrono.ChVector3d(0, 0, 0.5), 3, 4, 10, 40, 512)
                self.m_render_setup = True

            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()
        elif mode == 'follow':
            self.render_mode = 'follow'
            if self.m_render_setup == False:
                self.vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
                self.vis.SetWindowTitle('Box Agent')
                self.vis.SetWindowSize(1280, 1024)
                trackPoint = chrono.ChVector3d(0.0, 0.0, 1.75)
                self.vis.SetChaseCamera(trackPoint, 6.0, 0.5)
                self.vis.Initialize()
                self.vis.AddLightDirectional()
                self.vis.AddSkyBox()
                self.vis.AttachVehicle(self.m_vehicle.GetVehicle())
                self.m_render_setup = True

            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()
        # else:
            # raise NotImplementedError

    def get_observation(self):
        """
        Get the observation of the environment
            1. Camera image (RGB) of size (m_camera_width, m_camera_height)
            2. Delta x of the goal in local frame of the vehicle
            3. Delta y of the goal in local frame of the vehicle
            4. Vehicle heading
            5. Heading needed to reach the goal
            6. Velocity of the vehicle     
        :return: Observation of the environment
        """
        camera_buffer = self.m_camera.GetMostRecentRGBA8Buffer()
        if camera_buffer.HasData():
            camera_data = camera_buffer.GetRGBA8Data()
            camera_data = torch.tensor(camera_data, dtype=torch.uint8)
            # Remove the 4th column which is transparency
            camera_data = camera_data[:, :, :3]
            camera_data = torch.flip(camera_data, dims=[0])  # Flip vertically
        else:
            camera_data = torch.zeros(
                self.image_height, self.image_width, 3, dtype=torch.uint8)


        robot_x = torch.tensor(
            self.virtual_robot.GetPos().x, dtype=torch.float32)
        robot_y = torch.tensor(
            self.virtual_robot.GetPos().y, dtype=torch.float32)
        quat_list = [self.virtual_robot.GetRot().e0, self.virtual_robot.GetRot().e1,
                     self.virtual_robot.GetRot().e2, self.virtual_robot.GetRot().e3]
        yaw = self.quaternion_to_yaw(quat_list)
        robot_yaw = torch.tensor(yaw, dtype=torch.float32)

        # Goal position (assuming self.m_goal is a numpy array [goal_x, goal_y])
        goal_x, goal_y = self.m_goal

        # Vector to goal in global coordinates
        vector_to_goal_global = np.array([goal_x - robot_x.item(), goal_y - robot_y.item()])

        # Rotate the vector to goal into the robot's local frame
        # Since robot_yaw is the yaw angle, we can use a 2D rotation matrix
        cos_yaw = np.cos(robot_yaw.item())
        sin_yaw = np.sin(robot_yaw.item())
        rotation_matrix = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
        vector_to_goal_local = rotation_matrix @ vector_to_goal_global

        # Vehicle heading (already computed as robot_yaw)
        vehicle_heading = robot_yaw.item()

        # Target heading to goal (angle from the robot's position to the goal)
        target_heading_to_goal = np.arctan2(vector_to_goal_global[1], vector_to_goal_global[0])

        obs_dict = {
            "rgb": camera_data,
            # "depth": depth_data,
            "gps": torch.stack((robot_x, robot_y)),
            "compass": robot_yaw,
            "vector_to_goal_local": torch.tensor(vector_to_goal_local, dtype=torch.float32),
            "vehicle_heading": torch.tensor(vehicle_heading, dtype=torch.float32),
            "target_heading_to_goal": torch.tensor(target_heading_to_goal, dtype=torch.float32),
        }

        return obs_dict

    def get_reward(self):
        """
        Not using delta action for now
        """
        # Compute the progress made
        progress_scale = 20.  # coefficient for scaling progress reward
        distance = self.m_vector_to_goal_noNoise.Length()
        # The progress made with the last action
        progress = self.m_old_distance - distance

        reward = progress_scale * progress

        # If we have not moved even by 1 cm in 0.1 seconds give a penalty
        if np.abs(progress) < 0.01:
            reward -= 10

        self.m_old_distance = distance

        return reward

    def _is_terminated(self):
        """
        Check if the environment is terminated
        """
        # If we are within a certain distance of the goal -> Terminate and give big reward
        # if np.linalg.norm(self.observation[:3] - self.goal) < 0.4:
        if np.linalg.norm(self.m_vector_to_goal_noNoise.Length()) < 10:
            print('--------------------------------------------------------------')
            print('Goal Reached')
            print('Initial position: ', self.m_initLoc)
            print('Goal position: ', self.m_goal)
            print('--------------------------------------------------------------')
            self.m_reward += 2500
            self.m_debug_reward += self.m_reward
            self.m_terminated = True
            self.m_success = True

        # If we have exceeded the max time -> Terminate and give penalty for how far we are from the goal
        if self.m_system.GetChTime() > self.m_max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Initial position: ', self.m_initLoc)
            # dist = np.linalg.norm(self.observation[:3] - self.goal)
            dist = self.m_vector_to_goal_noNoise.Length()
            print('Final position of Gator: ',
                  self.m_chassis_body.GetPos())
            print('Goal position: ', self.m_goal)
            print('Distance to goal: ', dist)
            # Give it a reward based on how close it reached the goal
            # self.m_reward -= 400
            self.m_reward -= 10 * dist

            self.m_debug_reward += self.m_reward
            print('Reward: ', self.m_reward)
            print('Accumulated Reward: ', self.m_debug_reward)
            print('--------------------------------------------------------------')
            self.m_terminated = True

    def _is_truncated(self):
        """
        Check if we have crashed or fallen off terrain
        """
        collision = self.m_assets.CheckContact(
            self.m_chassis_body, proper_collision=self.m_proper_collision)
        if collision:
            self.m_reward -= 600
            print('--------------------------------------------------------------')
            print(f'Crashed')
            print('--------------------------------------------------------------')
            self.m_debug_reward += self.m_reward
            self.m_truncated = True

    def initialize_agent_pos(self, seed):
        """
        Initialize the robot position
        :param seed: Seed for the random number generator
        :return: Random angle between 0 and 2pi along which agent is oriented
         """
        # Random angle between 0 and 2pi
        theta = random.random() * 2 * np.pi
        # # x, y = self.m_terrain_length * 0.5 * \
        # #     np.cos(theta), self.m_terrain_width * 0.5 * np.sin(theta)
        # # z = 0.25
        # # ang = np.pi + theta
        # # self.m_initLoc = chrono.ChVector3d(x, y, z)
        # # self.m_initRot = chrono.QuatFromAngleZ(ang)
        self.virtual_robot.SetPos(chrono.ChVector3d(-1.25, -1.25, 0.25))
        return theta

    def set_goal(self, seed):
        """
        Set the goal point
        :param seed: Seed for the random number generator
        """
        # Random angle between -pi/2 and pi/2
        delta_theta = (random.random() - 0.5) * 1.0 * np.pi
        # ensure that the goal is always an angle between -pi/2 and pi/2 from the gator

        # TODO: fix the theta stuff
        theta = random.random() * 2 * np.pi

        gx, gy = self.m_terrain_length * 0.5 * np.cos(theta + np.pi + delta_theta), self.m_terrain_width * 0.5 * np.sin(
            theta + np.pi + delta_theta)
        self.m_goal = chrono.ChVector3d(
            gx, gy, 1.0)

        # Modify the goal point to be minimum 15 m away from gator
        # i = 0
        # while (self.m_goal - self.m_initLoc).Length() < 15:
        #     gx = random.random() * self.m_terrain_length - self.m_terrain_length / 2
        #     gy = random.random() * self.m_terrain_width - self.m_terrain_width / 2
        #     self.m_goal = chrono.ChVector3d(
        #         gx, gy, self.m_max_terrain_height + 1)
        #     if i > 100:
        #         print('Failed setting goal randomly, using default')
        #         gx = self.m_terrain_length * 0.625 * \
        #             np.cos(gator_theta + np.pi + delta_theta)
        #         gy = self.m_terrain_width * 0.625 * \
        #             np.sin(gator_theta + np.pi + delta_theta)
        #         break
        #     i += 1

        # Set the goal visualization
        goal_contact_material = chrono.ChContactMaterialNSC()
        goal_mat = chrono.ChVisualMaterial()
        goal_mat.SetAmbientColor(chrono.ChColor(1., 0., 0.))
        goal_mat.SetDiffuseColor(chrono.ChColor(1., 0., 0.))

        goal_body = chrono.ChBodyEasySphere(
            0.55, 1000, True, False, goal_contact_material)

        goal_body.SetPos(self.m_goal)
        goal_body.SetFixed(True)
        goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)

        self.m_system.Add(goal_body)

    # def add_obstacles(self, proper_collision=False):
    #     """Add obstacles to the terrain using asset utilities"""
    #     self.m_proper_collision = proper_collision

    #     if (self.m_proper_collision):
    #         # Create baseline type of rock assets
    #         rock1 = Asset(visual_shape_path="sensor/offroad/rock1.obj",
    #                       scale=1, bounding_box=chrono.ChVector3d(3.18344, 3.62827, 0))
    #         rock2 = Asset(visual_shape_path="sensor/offroad/rock2.obj",
    #                       scale=1, bounding_box=chrono.ChVector3d(4.01152, 2.64947, 0))
    #         rock3 = Asset(visual_shape_path="sensor/offroad/rock3.obj",
    #                       scale=1, bounding_box=chrono.ChVector3d(2.53149, 2.48862, 0))
    #         rock4 = Asset(visual_shape_path="sensor/offroad/rock4.obj",
    #                       scale=1, bounding_box=chrono.ChVector3d(2.4181, 4.47276, 0))
    #         rock5 = Asset(visual_shape_path="sensor/offroad/rock5.obj",
    #                       scale=1, bounding_box=chrono.ChVector3d(3.80205, 2.56996, 0))
    #     else:  # If there is no proper collision then collision just based on distance
    #         # Create baseline type of rock assets
    #         rock1 = Asset(visual_shape_path="sensor/offroad/rock1.obj",
    #                       scale=1)
    #         rock2 = Asset(visual_shape_path="sensor/offroad/rock2.obj",
    #                       scale=1)
    #         rock3 = Asset(visual_shape_path="sensor/offroad/rock3.obj",
    #                       scale=1)
    #         rock4 = Asset(visual_shape_path="sensor/offroad/rock4.obj",
    #                       scale=1)
    #         rock5 = Asset(visual_shape_path="sensor/offroad/rock5.obj",
    #                       scale=1)

    #     # Add these Assets to the simulationAssets
    #     self.m_assets = SimulationAssets(
    #         self.m_system, self.m_terrain, self.m_terrain_length, self.m_terrain_width)

    #     rock1_random = random.randint(0, 10)
    #     rock2_random = random.randint(0, 10)
    #     rock3_random = random.randint(0, 10)

    #     self.m_assets.AddAsset(rock1, number=rock1_random)
    #     self.m_assets.AddAsset(rock2, number=rock2_random)
    #     self.m_assets.AddAsset(rock3, number=rock3_random)
    #     # self.m_assets.AddAsset(rock4, number=2)
    #     # self.m_assets.AddAsset(rock5, number=2)

    #     # Randomly position these assets and add them to the simulation
    #     self.m_assets.RandomlyPositionAssets(self.m_goal, self.m_chassis_body)

    def add_sensors(self, camera=True, gps=True, imu=True):
        """
        Add sensors to the simulation
        :param camera: Flag to add camera sensor
        :param gps: Flag to add gps sensor
        :param imu: Flag to add imu sensor
        """
        # -------------------------------
        # Add camera sensor
        # -------------------------------
        print("line 601")
        if camera:
            self.m_have_camera = True
            offset_pose = chrono.ChFramed(chrono.ChVector3d(0.3, 0, 0.25), chrono.QUNIT)
            print("line 608")
            # print(self.virtual_robot)
            self.m_camera = sens.ChCameraSensor(
                self.virtual_robot,  # body camera is attached to
                self.update_rate,  # update rate in Hz
                offset_pose,  # offset pose
                self.image_width,  # image width
                self.image_height,  # image height
                self.fov,
                6
            )
            print("line 615")
            self.m_camera.SetName("Camera Sensor")
            self.m_camera.PushFilter(sens.ChFilterRGBA8Access())
            if (self.m_additional_render_mode == 'agent_pov'):
                self.m_camera.PushFilter(sens.ChFilterVisualize(
                    self.image_width, self.image_height, "Agent POV"))
            self.m_sens_manager.AddSensor(self.m_camera)
        # if gps:
        #     self.m_have_gps = True
        #     std = 0.01  # GPS noise standard deviation - Good RTK GPS
        #     gps_noise = sens.ChNoiseNormal(chrono.ChVector3d(
        #         0, 0, 0), chrono.ChVector3d(std, std, std))
        #     gps_loc = chrono.ChVector3d(0, 0, 0)
        #     gps_rot = chrono.QuatFromAngleAxis(0, chrono.ChVector3d(0, 1, 0))
        #     gps_frame = chrono.ChFramed(gps_loc, gps_rot)
        #     self.m_gps_origin = chrono.ChVector3d(43.073268, -89.400636, 260.0)

        #     self.m_gps = sens.ChGPSSensor(
        #         self.m_chassis_body,
        #         self.m_gps_frequency,
        #         gps_frame,
        #         self.m_gps_origin,
        #         gps_noise
        #     )
        #     self.m_gps.SetName("GPS Sensor")
        #     self.m_gps.PushFilter(sens.ChFilterGPSAccess())
        #     self.m_sens_manager.AddSensor(self.m_gps)
        # if imu:
        #     self.m_have_imu = True
        #     std = 0.01
        #     imu_noise = sens.ChNoiseNormal(chrono.ChVector3d(
        #         0, 0, 0), chrono.ChVector3d(std, std, std))
        #     imu_loc = chrono.ChVector3d(0, 0, 0)
        #     imu_rot = chrono.QuatFromAngleAxis(0, chrono.ChVector3d(0, 1, 0))
        #     imu_frame = chrono.ChFramed(imu_loc, imu_rot)
        #     self.m_imu_origin = chrono.ChVector3d(43.073268, -89.400636, 260.0)
        #     self.m_imu = sens.ChIMUSensor(
        #         self.m_chassis_body,
        #         self.m_imu_frequency,
        #         imu_frame,
        #         imu_noise,
        #         self.m_imu_origin
        #     )
        #     self.m_imu.SetName("IMU Sensor")
        #     self.m_imu.PushFilter(sens.ChFilterMagnetAccess())
        #     self.m_sens_manager.AddSensor(self.m_imu)

    # def set_nice_vehicle_mesh(self):
    #     self.m_play_mode = True

    def close(self):
        del self.virtual_robot
        del self.m_sens_manager
        del self.m_system
        # del self.m_assets.system
        # del self.m_assets
        del self

    def __del__(self):
        del self.m_sens_manager
        del self.m_system
        # del self.m_assets.system
        # del self.m_assets
        pass
