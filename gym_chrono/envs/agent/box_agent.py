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
# =======================================================================================
# Authors: Sriram Ashokkumar
# =======================================================================================

import gymnasium as gym
import numpy as np
import math
import os
from gym_chrono.envs.utils.terrain_utils import SCMParameters
from gym_chrono.envs.utils.perlin_bitmap_generator import generate_random_bitmap
from gym_chrono.envs.utils.asset_utils import *
from gym_chrono.envs.utils.utils import (
    CalcInitialPose,
    chVector_to_npArray,
    npArray_to_chVector,
    SetChronoDataDirectories,
)
from gym_chrono.envs.ChronoBase import ChronoBaseEnv
import pychrono.vehicle as veh
import pychrono as chrono
from typing import Any
import torch

# Import ChronoIrrlicht and Chrono Sensor once
try:
    from pychrono import irrlicht as chronoirr
except ImportError:
    print("Could not import ChronoIrrlicht")
try:
    import pychrono.sensor as sens
except ImportError:
    print("Could not import Chrono Sensor")

# Determine project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))


class box_agent(ChronoBaseEnv):
    """Box agent environment for Project Chrono with obstacles and goal navigation."""

    metadata = {"additional_render.modes": ["agent_pov", "None"]}

    def __init__(self, additional_render_mode="None"):
        if additional_render_mode not in box_agent.metadata["additional_render.modes"]:
            raise Exception(f"Render mode: {additional_render_mode} not supported")
        super().__init__(additional_render_mode)
        SetChronoDataDirectories()

        # Simulation parameters
        self.image_width = 640
        self.image_height = 480
        self.update_rate = 30
        self.fov = 1.408

        # Observation and action spaces
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(3, self.image_height, self.image_width), dtype=np.uint8),
            "depth": gym.spaces.Box(low=0, high=1, shape=(self.image_height, self.image_width), dtype=np.float32),
            "data": gym.spaces.Box(low=-100, high=100, shape=(4,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(5)

        # Internal simulation variables
        self.system = None
        self.virtual_robot = None
        self.assets = []
        self.initLoc = None

        self._control_frequency = 10
        self._step_size = 1e-3
        self._steps_per_control = round(1 / (self._step_size * self._control_frequency))

        self.m_terrain_length = 100
        self.m_terrain_width = 100

        self.m_sens_manager = None
        self.cam = None

        # Environment-specific parameters
        self.m_max_time = 10
        self.m_reward = 0
        self.m_debug_reward = 0
        self.m_action = None
        self.m_old_action = None
        self.m_goal = None
        self.m_vector_to_goal = None
        self.m_vector_to_goal_noNoise = None
        self.m_old_distance = None
        self.observation = None
        self.m_terminated = False
        self.m_truncated = False
        self.m_render_setup = False
        self.m_success = False
        self.m_play_mode = False
        self.m_additional_render_mode = additional_render_mode

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        # Initialize Chrono system
        self.system = chrono.ChSystemSMC()
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        self.system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

        # Create terrain
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

        # Create vehicle
        patch_mat = chrono.ChContactMaterialSMC()
        self.virtual_robot = chrono.ChBodyEasyBox(0.25, 0.25, 0.5, 100, True, True, patch_mat)
        self.virtual_robot.SetPos(chrono.ChVector3d(-1.25, -1.25, 0.25))
        self.virtual_robot.SetFixed(True)
        self.system.Add(self.virtual_robot)
        self.initialize_agent_pos(seed)

        # Set goal and obstacles
        self.set_goal(seed)
        self.add_obstacles(proper_collision=False)

        # Initialize sensors
        self.m_sens_manager = sens.ChSensorManager(self.system)
        self.m_sens_manager.scene.AddPointLight(chrono.ChVector3f(100, 100, 100), chrono.ChColor(1, 1, 1), 5000.0)
        offset_pose = chrono.ChFramed(chrono.ChVector3d(0.3, 0, 0.25), chrono.QUNIT)
        self.cam = sens.ChCameraSensor(self.virtual_robot, 100, offset_pose, self.image_width, self.image_height, self.fov, 6)
        self.cam.SetName("Camera Sensor")
        self.cam.PushFilter(sens.ChFilterVisualize(self.image_width, self.image_height, "agent pov"))
        self.cam.PushFilter(sens.ChFilterRGBA8Access())
        self.m_sens_manager.AddSensor(self.cam)

        self.lidar = sens.ChLidarSensor(
            self.virtual_robot, 100, offset_pose, self.image_width, self.image_height, self.fov,
            chrono.CH_PI/6, -chrono.CH_PI/6, 3.66, sens.LidarBeamShape_RECTANGULAR,
            1, 0, 0, sens.LidarReturnMode_STRONGEST_RETURN)
        self.lidar.SetName("Lidar Sensor")
        self.lidar.SetLag(0)
        self.lidar.SetCollectionWindow(1/20)
        self.lidar.PushFilter(sens.ChFilterVisualize(self.image_width, self.image_height, "depth camera"))
        self.lidar.PushFilter(sens.ChFilterDIAccess())
        self.m_sens_manager.AddSensor(self.lidar)

        self.observation = self.get_observation()
        self.m_old_distance = self.m_vector_to_goal
        self.m_old_action = np.zeros((2,))
        self.m_debug_reward = 0
        self.m_reward = 0
        self.m_render_setup = False
        self.m_terminated = False
        self.m_truncated = False

        current_yaw = self.quaternion_to_yaw([
            self.virtual_robot.GetRot().e0,
            self.virtual_robot.GetRot().e1,
            self.virtual_robot.GetRot().e2,
            self.virtual_robot.GetRot().e3,
        ])
        self.prev_yaw = current_yaw
        self.cumulative_rotation = 0.0

        return self.observation, {}

    def step(self, action):
        """Apply the given action and update the environment state."""
        if action == 1:  # move forward
            self.virtual_robot.SetPos(self.virtual_robot.GetPos() + chrono.ChVector3d(0.1, 0, 0))
        elif action == 2:  # turn left
            self.virtual_robot.SetRot(self.virtual_robot.GetRot() * chrono.QuatFromAngleZ(0.1))
        elif action == 3:  # turn right
            self.virtual_robot.SetRot(self.virtual_robot.GetRot() * chrono.QuatFromAngleZ(-0.1))
        elif action == 4:  # reached goal (no action specified)
            pass

        self.m_action = action
        self.system.DoStepDynamics(self._step_size)
        self.m_sens_manager.Update()

        # Update cumulative rotation tracking
        current_yaw = self.quaternion_to_yaw([
            self.virtual_robot.GetRot().e0,
            self.virtual_robot.GetRot().e1,
            self.virtual_robot.GetRot().e2,
            self.virtual_robot.GetRot().e3,
        ])
        delta_yaw = abs(current_yaw - self.prev_yaw)
        if delta_yaw > np.pi:
            delta_yaw = 2 * np.pi - delta_yaw  # wrap difference to [0, π]
        
        if action in [2, 3]:
            self.cumulative_rotation += delta_yaw
        else:
            # If the action is not turning, reset the accumulated rotation
            self.cumulative_rotation = 0.0
        
        self.prev_yaw = current_yaw

        self.observation = self.get_observation()
        self.m_reward = self.get_reward()
        self.m_debug_reward += self.m_reward

        self._is_terminated()
        self._is_truncated()
        return self.observation, self.m_reward, self.m_terminated, self.m_truncated, {}

    def render(self, mode="follow"):
        """Render the environment using Chrono Irrlicht."""
        if mode == "follow":
            self.render_mode = "follow"
            if not self.m_render_setup:
                self.vis = chronoirr.ChVisualSystemIrrlicht(self.system)
                self.vis.SetWindowTitle("Agent Exploration")
                self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
                self.vis.AddLightWithShadow(chrono.ChVector3d(2, 2, 2), chrono.ChVector3d(0, 0, 0), 5, 1, 11, 55)
                self.vis.EnableAbsCoordsysDrawing(True)
                self.vis.Initialize()
                self.vis.AddSkyBox()
                self.vis.AddCamera(chrono.ChVector3d(-7/3, 0, 4.5/3), chrono.ChVector3d(0, 0, 0))
                self.m_render_setup = True
            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()

    def get_observation(self):
        """Collect the current observation from camera, lidar, and robot state."""
        # Process camera image
        camera_buffer = self.cam.GetMostRecentRGBA8Buffer()
        if camera_buffer.HasData():
            camera_data = camera_buffer.GetRGBA8Data()
            camera_data = torch.tensor(camera_data, dtype=torch.uint8)
            camera_data = camera_data[:, :, :3]
            camera_data = torch.flip(camera_data, dims=[0])
        else:
            camera_data = torch.zeros(self.image_height, self.image_width, 3, dtype=torch.uint8)

        # Process depth from lidar
        depth_buffer = self.lidar.GetMostRecentDIBuffer()
        if depth_buffer.HasData():
            depth_data = depth_buffer.GetDIData()
            depth_data = torch.tensor(depth_data[:, :, 0], dtype=torch.float32)
            depth_data = torch.flip(depth_data, dims=[0, 1])
            MIN_DEPTH, MAX_DEPTH = 0, 5.5
            depth_data = np.clip((depth_data - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0, 1)
            depth_data[depth_data == 0] = 1
        else:
            depth_data = torch.zeros(self.image_height, self.image_width, dtype=torch.float32)

        # Get robot state
        robot_pos = self.virtual_robot.GetPos()
        robot_x = torch.tensor(robot_pos.x, dtype=torch.float32)
        robot_y = torch.tensor(robot_pos.y, dtype=torch.float32)
        quat_list = [
            self.virtual_robot.GetRot().e0,
            self.virtual_robot.GetRot().e1,
            self.virtual_robot.GetRot().e2,
            self.virtual_robot.GetRot().e3,
        ]
        yaw = self.quaternion_to_yaw(quat_list)
        robot_yaw = torch.tensor(yaw, dtype=torch.float32)

        # Compute goal vector in global and local coordinates
        goal_x, goal_y = self.m_goal.x, self.m_goal.y
        vector_to_goal_global = np.array([goal_x - robot_x.item(), goal_y - robot_y.item()])
        self.m_vector_to_goal = np.linalg.norm(vector_to_goal_global)
        cos_yaw = np.cos(robot_yaw.item())
        sin_yaw = np.sin(robot_yaw.item())
        rotation_matrix = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
        vector_to_goal_local = rotation_matrix @ vector_to_goal_global
        target_heading_to_goal = np.arctan2(vector_to_goal_global[1], vector_to_goal_global[0])
        observation_array = np.array([vector_to_goal_local[0], vector_to_goal_local[1], robot_yaw.item(), target_heading_to_goal])

        # Transpose camera data to (channels, height, width)
        camera_data = np.transpose(camera_data, (2, 0, 1))
        return {"image": camera_data, "depth": depth_data, "data": observation_array}

    def get_reward(self):
        """Compute reward based on progress towards the goal."""
        progress_scale = 40
        distance = self.m_vector_to_goal
        progress = self.m_old_distance - distance
        reward = progress_scale * progress
        # if np.abs(progress) < 0.01:
        #     reward -= 2
        if np.abs(progress) < 0.01 and self.m_action not in [2, 3]:
            reward -= 10

        # Penalize if excessive spinning is detected
        if self.cumulative_rotation > 4 * np.pi:
            reward -= 100  # adjust penalty value as needed
            # Optionally, reset cumulative rotation after penalty
            self.cumulative_rotation = 0.0

        self.m_old_distance = distance
        return reward

    def _is_terminated(self):
        """Check if the episode should terminate (goal reached or time out)."""
        print("Distance to goal:", self.m_vector_to_goal)
        if np.linalg.norm(self.m_vector_to_goal) < 1.5:
            print("--------------------------------------------------------------")
            print("Goal Reached")
            print("Initial position:", self.initLoc)
            print("Goal position:", self.m_goal)
            print("--------------------------------------------------------------")
            self.m_reward += 2500
            self.m_debug_reward += self.m_reward
            self.m_terminated = True
            self.m_success = True

        print("Time:", self.system.GetChTime())
        if self.system.GetChTime() > self.m_max_time:
            print("--------------------------------------------------------------")
            print("Time out")
            print("Initial position:", self.initLoc)
            dist = self.m_vector_to_goal
            print("Goal position:", self.m_goal)
            print("Distance to goal:", dist)
            self.m_reward -= 10 * dist
            self.m_debug_reward += self.m_reward
            print("Reward:", self.m_reward)
            print("Accumulated Reward:", self.m_debug_reward)
            print("--------------------------------------------------------------")
            self.m_terminated = True

    def _is_truncated(self):
        """Check if the robot has crashed into an obstacle or fallen off the terrain."""
        robot_width = 0.25
        robot_depth = 0.25
        robot_radius = np.sqrt((robot_width / 2)**2 + (robot_depth / 2)**2)
        robot_pos = self.virtual_robot.GetPos()
        for obstacle, obs_radius in self.assets:
            obs_pos = obstacle.GetPos()
            dx = robot_pos.x - obs_pos.x
            dy = robot_pos.y - obs_pos.y
            distance = np.sqrt(dx**2 + dy**2)
            if distance <= (robot_radius + obs_radius):
                self.m_reward -= 600
                print("--------------------------------------------------------------")
                print("Crashed into obstacle")
                print("--------------------------------------------------------------")
                self.m_debug_reward += self.m_reward
                self.m_truncated = True
                return
        if self._fallen_off_terrain():
            self.m_reward -= 600
            print("--------------------------------------------------------------")
            print("Fallen off terrain")
            print("--------------------------------------------------------------")
            self.m_debug_reward += self.m_reward
            self.m_truncated = True

    def _fallen_off_terrain(self):
        """Return True if the robot's position is outside terrain bounds."""
        terrain_length_tolerance = 100
        terrain_width_tolerance = 100
        robot_pos = self.virtual_robot.GetPos()
        return abs(robot_pos.x) > terrain_length_tolerance or abs(robot_pos.y) > terrain_width_tolerance

    def initialize_agent_pos(self, seed):
        """Initialize the robot's starting position and return a random orientation."""
        theta = random.random() * 2 * np.pi
        self.initLoc = chrono.ChVector3d(-1.25, -1.25, 0.25)
        self.virtual_robot.SetPos(self.initLoc)
        return theta

    def set_goal(self, seed):
        """Set a random goal within 20 meters of the robot, ensuring it's at least 2 meters away."""
        robot_pos = self.initLoc
        r = np.random.uniform(2, 20)
        theta = random.random() * 2 * np.pi
        gx = robot_pos.x + r * np.cos(theta)
        gy = robot_pos.y + r * np.sin(theta)
        self.m_goal = chrono.ChVector3d(gx, gy, 0.5)
        while (self.m_goal - robot_pos).Length() < 2:
            r = np.random.uniform(2, 20)
            theta = random.random() * 2 * np.pi
            gx = robot_pos.x + r * np.cos(theta)
            gy = robot_pos.y + r * np.sin(theta)
            self.m_goal = chrono.ChVector3d(gx, gy, 0.5)

        goal_contact_material = chrono.ChContactMaterialSMC()
        goal_mat = chrono.ChVisualMaterial()
        goal_mat.SetAmbientColor(chrono.ChColor(1.0, 0.0, 0.0))
        goal_mat.SetDiffuseColor(chrono.ChColor(1.0, 0.0, 0.0))
        goal_body = chrono.ChBodyEasySphere(0.2, 1000, True, False, goal_contact_material)
        goal_body.SetPos(self.m_goal)
        goal_body.SetFixed(True)
        goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)
        self.system.Add(goal_body)

    def add_obstacles(self, proper_collision=False):
        """Add obstacles randomly within 2 to 15 meters of the robot."""
        num_obstacles = 3
        robot_pos = self.virtual_robot.GetPos()
        self.assets = []
        for i in range(num_obstacles):
            width = np.random.uniform(0.5, 2.0)
            depth = np.random.uniform(0.5, 2.0)
            height = np.random.uniform(0.5, 2.0)
            obstacle_mat = chrono.ChContactMaterialSMC()
            obstacle_mat.SetFriction(0.9)
            obstacle_mat.SetYoungModulus(1e7)
            obstacle = chrono.ChBodyEasyBox(width, depth, height, 100, True, True, obstacle_mat)
            r = np.random.uniform(2, 15)
            theta = np.random.uniform(0, 2 * np.pi)
            x = robot_pos.x + r * np.cos(theta)
            y = robot_pos.y + r * np.sin(theta)
            obstacle.SetPos(chrono.ChVector3d(x, y, height / 2))
            obstacle.SetFixed(True)
            obstacle.EnableCollision(False)
            obstacle.GetVisualShape(0).SetColor(chrono.ChColor(1, 0, 0))
            obs_radius = np.sqrt((width / 2)**2 + (depth / 2)**2)
            self.system.Add(obstacle)
            self.assets.append((obstacle, obs_radius))

    def add_sensors(self, camera=True, gps=True, imu=True):
        pass

    def quaternion_to_yaw(self, quaternion):
        """Convert quaternion to yaw angle."""
        w, x, y, z = quaternion
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    def close(self):
        del self.virtual_robot
        del self.m_sens_manager
        del self.system
        del self.assets
        del self

    def __del__(self):
        del self.m_sens_manager
        del self.system
        del self.assets
