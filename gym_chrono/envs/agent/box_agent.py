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
# Authors: Sriram Ashokkumar
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
import torch

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
        # 1. Camera image (RGB) of size (cam_width, cam_height)
        # 2. Vehicle state relative to the goal of size (5,)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(
                3, self.image_height, self.image_width), dtype=np.uint8),
            "depth": gym.spaces.Box(low=0, high=1, shape=(self.image_height, self.image_width), dtype=np.float32),
            "data": gym.spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float32)})

        # Action space
        self.action_space = gym.spaces.Discrete(5)

        # -------------------------------
        # Simulation specific class variables
        # -------------------------------
        self.system = None  # Chrono system
        self.virtual_robot = None
        self.assets = None  # List of assets in the simulation
        self.initLoc = None

        # Control and dynamics frequency
        self._control_frequency = 10  # Control frequency of the simulation
        self._step_size = 1e-3  # Step size of the simulation
        self._steps_per_control = round(
            1 / (self._step_size * self._control_frequency))

        # Terrain
        self.m_terrain_length = 100  # size in X direction
        self.m_terrain_width = 100  # size in Y direction
        self.assets = []
        # Sensor manager
        self.m_sens_manager = None  # Sensor manager for the simulation
        self.cam = None  # Camera sensor

        # -------------------------------
        # Gym Env specific parameters
        # -------------------------------
        self.m_max_time = 10  # Max time for each episode
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
        self.observation = None
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
        self.system = chrono.ChSystemSMC()
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        self.system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

        # -------------------------------
        # Reset the terrain
        # -------------------------------
        self.m_isFlat = True
        self.m_isRigid = True

        ground_mat = chrono.ChContactMaterialSMC()
        ground_mat.SetFriction(0.9)
        ground_mat.SetYoungModulus(1e7)
        ground = chrono.ChBodyEasyBox(100, 100, 0.1, 1000, True, True, ground_mat)
        ground.SetPos(chrono.ChVector3d(0, 0, 0))
        ground.SetFixed(True)
        ground.EnableCollision(True)
        ground.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/concrete.jpg"))
        self.system.Add(ground)


        # -------------------------------
        # Reset the vehicle
        # -------------------------------

        patch_mat = chrono.ChContactMaterialSMC()
        self.virtual_robot = chrono.ChBodyEasyBox(
            0.25, 0.25, 0.5, 100, True, True, patch_mat)
        self.virtual_robot.SetPos(chrono.ChVector3d(-1.25, -1.25, 0.25))
        self.virtual_robot.SetFixed(True)
        self.system.Add(self.virtual_robot)
        robot_theta = self.initialize_agent_pos(seed)  # set the orientation of the agent ##TODO


        # -------------------------------
        # Set the goal point
        # -------------------------------
        self.set_goal(seed)

        # -------------------------------
        # Reset the obstacles
        # -------------------------------
        self.add_obstacles(proper_collision=False)

        # -------------------------------
        # Initialize the sensors
        # -------------------------------
        del self.m_sens_manager
        self.m_sens_manager = sens.ChSensorManager(self.system)
        # Set the lighting scene
        self.m_sens_manager.scene.AddPointLight(chrono.ChVector3f(
            100, 100, 100), chrono.ChColor(1, 1, 1), 5000.0)
        
        offset_pose = chrono.ChFramed(chrono.ChVector3d(0.3, 0, 0.25), chrono.QUNIT)
        self.cam = sens.ChCameraSensor(
            self.virtual_robot,  # body camera is attached to
            100, #self.update_rate,  # update rate in Hz
            offset_pose,  # offset pose
            self.image_width,  # image width
            self.image_height,  # image height
            self.fov,
            6
        )
        self.cam.SetName("Camera Sensor")
        self.cam.PushFilter(sens.ChFilterVisualize(
            self.image_width, self.image_height, "agent pov"))
        self.cam.PushFilter(sens.ChFilterRGBA8Access())
        self.m_sens_manager.AddSensor(self.cam)

        self.lidar = sens.ChLidarSensor(
            self.virtual_robot,             # body lidar is attached to
            100,                     # scanning rate in Hz
            offset_pose,            # offset pose
            self.image_width,                   # number of horizontal samples
            self.image_height,                    # number of vertical channels
            self.fov,                    # horizontal field of view
            chrono.CH_PI/6,         # vertical field of view
            -chrono.CH_PI/6,
            3.66,                  # max lidar range
            sens.LidarBeamShape_RECTANGULAR,
            1,          # sample radius
            0,       # divergence angle
            0,       # divergence angle
            sens.LidarReturnMode_STRONGEST_RETURN)
        self.lidar.SetName("Lidar Sensor")
        self.lidar.SetLag(0)
        self.lidar.SetCollectionWindow(1/20)
        self.lidar.PushFilter(sens.ChFilterVisualize(
            self.image_width, self.image_height, "depth camera"))
        self.lidar.PushFilter(sens.ChFilterDIAccess())
        self.m_sens_manager.AddSensor(self.lidar)


        # -------------------------------
        # Get the initial observation
        # -------------------------------
        self.observation = self.get_observation()
        self.m_old_distance = self.m_vector_to_goal
        self.m_old_action = np.zeros((2,))
        # self.m_contact_force = 0
        self.m_debug_reward = 0
        self.m_reward = 0
        self.m_render_setup = False

        self.m_terminated = False
        self.m_truncated = False
        return self.observation, {}

    def step(self, action):
        """
        Box Agent takes a step in the environment - Frequency by default is 10 Hz
        """
        # steering = action[0]
        
        # Move robot forward in the direction it is facing
        if (action == 1): # move forward
            self.virtual_robot.SetPos(self.virtual_robot.GetPos() + chrono.ChVector3d(0.1, 0, 0))
        elif (action == 2): # turn left
            self.virtual_robot.SetRot(self.virtual_robot.GetRot() * chrono.QuatFromAngleZ(0.1))
        elif (action == 3): # turn right
            self.virtual_robot.SetRot(self.virtual_robot.GetRot() * chrono.QuatFromAngleZ(-0.1))
        elif (action == 4): # reached goal
            pass

        # This is used in the reward function
        self.m_action = action

        # Update the sensor manager
        self.system.DoStepDynamics(self._step_size)
        self.m_sens_manager.Update()
        
        # Get the observation
        self.observation = self.get_observation()
        self.m_reward = self.get_reward()
        self.m_debug_reward += self.m_reward

        # Check if we hit something or reached the goal
        self._is_terminated()
        self._is_truncated()

        return self.observation, self.m_reward, self.m_terminated, self.m_truncated, {}

    def render(self, mode='follow'):
        """
        Render the environment
        """

        # ------------------------------------------------------
        # Add visualization - only if we want to see "human" POV
        # ------------------------------------------------------
        if mode == 'human':
            self.render_mode = 'human'

            # if self.m_render_setup == False:
            #     self.vis = chronoirr.ChVisualSystemIrrlicht()
            #     self.vis.AttachSystem(self.system)
            #     self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
            #     self.vis.SetWindowSize(1280, 720)
            #     self.vis.SetWindowTitle('Box Agent')
            #     self.vis.Initialize()
            #     self.vis.AddSkyBox()
            #     self.vis.AddCamera(chrono.ChVector3d(
            #         0, 0, 80), chrono.ChVector3d(0, 0, 1))
            #     self.vis.AddTypicalLights()
            #     self.vis.AddLightWithShadow(chrono.ChVector3d(
            #         1.5, -2.5, 5.5), chrono.ChVector3d(0, 0, 0.5), 3, 4, 10, 40, 512)
            #     self.m_render_setup = True

            # self.vis.BeginScene()
            # self.vis.Render()
            # self.vis.EndScene()
        elif mode == 'follow':
            self.render_mode = 'follow'
            if self.m_render_setup == False:
                self.vis = chronoirr.ChVisualSystemIrrlicht(self.system)
                self.vis.SetWindowTitle('Agent Exploration')
                self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
                self.vis.AddLightWithShadow(chrono.ChVector3d(2, 2, 2),  # point
                                            chrono.ChVector3d(0, 0, 0),  # aimpoint
                                            5,                       # radius (power)
                                            1, 11,                     # near, far
                                            55)                       # angle of FOV

                # vis.EnableShadows()
                self.vis.EnableAbsCoordsysDrawing(True)
                self.vis.Initialize()
                self.vis.AddSkyBox()
                self.vis.AddCamera(chrono.ChVector3d(-7/3, 0, 4.5/3),
                                chrono.ChVector3d(0, 0, 0))
                self.m_render_setup = True

            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()
        # else:
            # raise NotImplementedError

    def get_observation(self):
        """
        Get the observation of the environment
            1. Camera image (RGB) of size (cam_width, cam_height)
            2. Delta x of the goal in local frame of the vehicle
            3. Delta y of the goal in local frame of the vehicle
            4. Vehicle heading
            5. Heading needed to reach the goal
            6. Velocity of the vehicle     
        :return: Observation of the environment
        """
        camera_buffer = self.cam.GetMostRecentRGBA8Buffer()
        if camera_buffer.HasData():
            camera_data = camera_buffer.GetRGBA8Data()
            camera_data = torch.tensor(camera_data, dtype=torch.uint8)
            # Remove the 4th column which is transparency
            camera_data = camera_data[:, :, :3]
            camera_data = torch.flip(camera_data, dims=[0])  # Flip vertically
        else:
            camera_data = torch.zeros(
                self.image_height, self.image_width, 3, dtype=torch.uint8)
            
        depth_buffer = self.lidar.GetMostRecentDIBuffer()
        if depth_buffer.HasData():
            depth_data = depth_buffer.GetDIData()
            # Removes the 2nd column which is intensity
            depth_data = torch.tensor(
                depth_data[:, :, 0], dtype=torch.float32)

            # Flip vertically and horizontally
            depth_data = torch.flip(depth_data, dims=[0, 1])

            MIN_DEPTH = 0
            MAX_DEPTH = 5.5
            depth_data = np.clip(
                (depth_data - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0, 1)

            # Set pixels to white for depth values greater than MAX_DEPTH
            depth_data[depth_data == 0] = 1  # Set all zero values to 1
        else:
            depth_data = torch.zeros(
                self.image_height, self.image_width, dtype=torch.float32)


        robot_x = torch.tensor(
            self.virtual_robot.GetPos().x, dtype=torch.float32)
        robot_y = torch.tensor(
            self.virtual_robot.GetPos().y, dtype=torch.float32)
        
        quat_list = [self.virtual_robot.GetRot().e0, self.virtual_robot.GetRot().e1,
                     self.virtual_robot.GetRot().e2, self.virtual_robot.GetRot().e3]
        yaw = self.quaternion_to_yaw(quat_list)
        robot_yaw = torch.tensor(yaw, dtype=torch.float32)

        # Goal position (assuming self.m_goal is a numpy array [goal_x, goal_y])
        goal_x, goal_y = self.m_goal.x, self.m_goal.y

        # Vector to goal in global coordinates
        vector_to_goal_global = np.array([goal_x - robot_x.item(), goal_y - robot_y.item()])
        self.m_vector_to_goal = np.linalg.norm(vector_to_goal_global)

        # Rotate the vector to goal into the robot's local frame
        # Since robot_yaw is the yaw angle, we can use a 2D rotation matrix
        cos_yaw = np.cos(robot_yaw.item())
        sin_yaw = np.sin(robot_yaw.item())
        rotation_matrix = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
        vector_to_goal_local = rotation_matrix @ vector_to_goal_global

        # Target heading to goal (angle from the robot's position to the goal)
        target_heading_to_goal = np.arctan2(vector_to_goal_global[1], vector_to_goal_global[0])

        observation_array = np.array(
            [vector_to_goal_local[0], vector_to_goal_local[1], robot_yaw.item(), target_heading_to_goal])
        camera_data = np.transpose(camera_data, (2, 0, 1))
        obs_dict = {"image": camera_data, "depth": depth_data, "data": observation_array}
        return obs_dict

    def get_reward(self):
        """
        Not using delta action for now
        """
        # Compute the progress made
        progress_scale = 20.  # coefficient for scaling progress reward
        distance = self.m_vector_to_goal
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

        print("Distance to goal:", self.m_vector_to_goal)
        # If we are within a certain distance of the goal -> Terminate and give big reward
        # if np.linalg.norm(self.observation[:3] - self.goal) < 0.4:
        if np.linalg.norm(self.m_vector_to_goal) < 1:
            print('--------------------------------------------------------------')
            print('Goal Reached')
            print('Initial position: ', self.initLoc)
            print('Goal position: ', self.m_goal)
            print('--------------------------------------------------------------')
            self.m_reward += 2500
            self.m_debug_reward += self.m_reward
            self.m_terminated = True
            self.m_success = True

        # If we have exceeded the max time -> Terminate and give penalty for how far we are from the goal
        print("Time: ", self.system.GetChTime())
        if self.system.GetChTime() > self.m_max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Initial position: ', self.initLoc)
            # dist = np.linalg.norm(self.observation[:3] - self.goal)
            dist = self.m_vector_to_goal
            # print('Final position of Gator: ',
            #       self.m_chassis_body.GetPos())
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
        Check if the robot has crashed (touched an obstacle) or fallen off terrain.
        """
        # Approximate robot dimensions (from its creation: 0.25, 0.25, 0.5)
        robot_width = 0.25
        robot_depth = 0.25
        robot_radius = np.sqrt((robot_width / 2)**2 + (robot_depth / 2)**2)
        
        # Get robot position
        robot_pos = self.virtual_robot.GetPos()
        
        # Loop over each obstacle in the assets list (stored as (obstacle, radius))
        for obstacle, obs_radius in self.assets:
            obs_pos = obstacle.GetPos()
            dx = robot_pos.x - obs_pos.x
            dy = robot_pos.y - obs_pos.y
            distance = np.sqrt(dx**2 + dy**2)
            if distance <= (robot_radius + obs_radius):
                self.m_reward -= 600
                print('--------------------------------------------------------------')
                print('Crashed into obstacle')
                print('--------------------------------------------------------------')
                self.m_debug_reward += self.m_reward
                self.m_truncated = True
                return  # Exit as we have detected a collision
        
        # Optionally, also check if the robot has fallen off the terrain.
        if self._fallen_off_terrain():
            self.m_reward -= 600
            print('--------------------------------------------------------------')
            print('Fallen off terrain')
            print('--------------------------------------------------------------')
            self.m_debug_reward += self.m_reward
            self.m_truncated = True
    
    def _fallen_off_terrain(self):
        """
        Check if we have fallen off the terrain
        For now just checks if the CG of the vehicle is within the rectangle bounds with some tolerance
        """
        terrain_length_tolerance = 100
        terrain_width_tolerance = 100

        vehicle_is_outside_terrain = abs(self.virtual_robot.GetPos().x) > terrain_length_tolerance or abs(
            self.virtual_robot.GetPos().y) > terrain_width_tolerance
        if (vehicle_is_outside_terrain):
            return True
        else:
            return False


    def initialize_agent_pos(self, seed):
        """
        Initialize the robot position    -- reference off_road_gator.py for random init pos
        :param seed: Seed for the random number generator
        :return: Random angle between 0 and 2pi along which agent is oriented
         """
        # Random angle between 0 and 2pi
        theta = random.random() * 2 * np.pi
        self.initLoc = chrono.ChVector3d(-1.25, -1.25, 0.25)
        self.virtual_robot.SetPos(self.initLoc)
        return theta

    def set_goal(self, seed):
        """
        Set the goal point randomly within a 20-meter range from the robot's initial position.
        The goal will be at least 2 meters away.
        """
        # Use the robot's initial position as reference
        robot_pos = self.initLoc  # This is set in initialize_agent_pos
        
        # Choose a random distance between 2 and 20 meters and a random angle between 0 and 2π
        r = np.random.uniform(2, 20)
        theta = random.random() * 2 * np.pi
        gx = robot_pos.x + r * np.cos(theta)
        gy = robot_pos.y + r * np.sin(theta)
        self.m_goal = chrono.ChVector3d(gx, gy, 0.5)

        # (Optional) Ensure the goal is not too close to the robot; repeat if necessary.
        while (self.m_goal - robot_pos).Length() < 2:
            r = np.random.uniform(2, 20)
            theta = random.random() * 2 * np.pi
            gx = robot_pos.x + r * np.cos(theta)
            gy = robot_pos.y + r * np.sin(theta)
            self.m_goal = chrono.ChVector3d(gx, gy, 0.5)

        # Set the goal visualization
        goal_contact_material = chrono.ChContactMaterialSMC()
        goal_mat = chrono.ChVisualMaterial()
        goal_mat.SetAmbientColor(chrono.ChColor(1., 0., 0.))
        goal_mat.SetDiffuseColor(chrono.ChColor(1., 0., 0.))

        goal_body = chrono.ChBodyEasySphere(0.2, 1000, True, False, goal_contact_material)
        goal_body.SetPos(self.m_goal)
        goal_body.SetFixed(True)
        goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)

        self.system.Add(goal_body)


    def add_obstacles(self, proper_collision=False):
        """Add obstacles to the terrain as fixed ChBodyEasyBox instances with collision disabled.
        Each obstacle is placed within a 2 to 15 m radius of the robot. Their size is used to compute
        an approximate radius (half-diagonal) for collision checking."""
        num_obstacles = 3  # Change as needed

        # Get the current robot position
        robot_pos = self.virtual_robot.GetPos()

        # Reset assets list to store tuples: (obstacle, obstacle_radius)
        self.assets = []

        for i in range(num_obstacles):
            # Random dimensions for the obstacle (width, depth, height)
            width = np.random.uniform(0.5, 2.0)
            depth = np.random.uniform(0.5, 2.0)
            height = np.random.uniform(0.5, 2.0)

            # Create a contact material for the obstacle
            obstacle_mat = chrono.ChContactMaterialSMC()
            obstacle_mat.SetFriction(0.9)
            obstacle_mat.SetYoungModulus(1e7)

            # Create the obstacle box
            obstacle = chrono.ChBodyEasyBox(width, depth, height, 100, True, True, obstacle_mat)
            
            # Generate a random position within 2 to 15 meters from the robot (using polar coordinates)
            r = np.random.uniform(2, 15)
            theta = np.random.uniform(0, 2 * np.pi)
            x = robot_pos.x + r * np.cos(theta)
            y = robot_pos.y + r * np.sin(theta)
            
            # Place the obstacle so it sits on the ground (z = height/2)
            obstacle.SetPos(chrono.ChVector3d(x, y, height / 2))
            
            # Set the obstacle as fixed and disable collision
            obstacle.SetFixed(True)
            obstacle.EnableCollision(False)
            
            # Optionally, set a distinct color (e.g., red) for visualization
            obstacle.GetVisualShape(0).SetColor(chrono.ChColor(1, 0, 0))
            
            # Compute an approximate radius (half-diagonal of its base rectangle)
            obs_radius = np.sqrt((width / 2)**2 + (depth / 2)**2)
            
            # Add the obstacle and its radius to the assets list
            self.system.Add(obstacle)
            self.assets.append((obstacle, obs_radius))

        

    def add_sensors(self, camera=True, gps=True, imu=True):
        """
        Add sensors to the simulation
        :param camera: Flag to add camera sensor
        :param gps: Flag to add gps sensor
        :param imu: Flag to add imu sensor
        """
        pass

    def quaternion_to_yaw(self, quaternion):
        # Unpack quaternion
        w, x, y, z = quaternion

        # Calculate yaw (angle with respect to the x-axis)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return yaw

    def close(self):
        del self.virtual_robot
        del self.m_sens_manager
        del self.system
        # del self.assets.system
        del self.assets
        del self

    def __del__(self):
        del self.m_sens_manager
        del self.system
        # del self.assets.system
        del self.assets
        pass
