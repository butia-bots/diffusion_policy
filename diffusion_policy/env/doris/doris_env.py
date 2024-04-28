import gym
from gym import spaces
import moveit_commander
import rospy
from sensor_msgs.msg import Image
import numpy as np
import math
from tf.transformations import euler_from_quaternion
import ros_numpy
import cv2
import collections
import pymunk
from pymunk.vec2d import Vec2d
import pygame


class DorisEnv(gym.Env):
    def __init__(self, teleop) -> None:
        self.teleop = teleop
        self.arm = moveit_commander.MoveGroupCommander.MoveGroupCommander('arm', robot_description='doris_arm/robot_description', ns='/doris_arm')
        self.arm.set_pose_reference_frame('doris_arm/base_link')
        self.gripper = moveit_commander.MoveGroupCommander.MoveGroupCommander('gripper', robot_description='doris_arm/robot_description', ns='/doris_arm')
        self.rgb_sub = rospy.Subscriber('/butia_vision/bvb/image_rgb', Image, self._update_rgb)
        self.action_space = spaces.Box(low=np.array([-1.0,-1.0,-1.0,-2*np.pi,-2*np.pi,-2*np.pi,0.0], high=np.array([1.0,1.0,1.0,2*np.pi,2*np.pi,2*np.pi,1.0])), shape=(7,))
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,480,640),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=np.array([-1.0,-1.0,-1.0,-2*np.pi,-2*np.pi,-2*np.pi,0.0]),
                high=np.array([1.0,1.0,1.0,2*np.pi,2*np.pi,2*np.pi,1.0]),
                shape=(7,),
                dtype=np.float32
            )
        })

    def set_arm_pose(self, pose: np.ndarray):
        pose[5] = math.atan2(pose[1], pose[0])
        self.arm.set_pose_target(pose.tolist())
        self.arm.go(wait=True)

    def get_arm_pose(self)->np.ndarray:
        ps = self.arm.get_current_pose()
        quat = [
            ps.pose.orientation.x,
            ps.pose.orientation.y,
            ps.pose.orientation.z,
            ps.pose.orientation.w
        ]
        rot = euler_from_quaternion(quat)
        return np.array([
            ps.pose.position.x,
            ps.pose.position.y,
            ps.pose.position.z,
            *rot
        ])
    
    def set_gripper_opening(self, opening: float):
        self.gripper.set_joint_value_target([opening/2.0,-opening/2.0])
        self.gripper.go(wait=True)

    def get_gripper_opening(self)->float:
        return sum([abs(v) for v in self.gripper.get_current_joint_values()])

    def _update_rgb(self, msg: Image):
        self.img_rgb = msg

    def _get_info(self):
        return {}

    def _get_obs(self):
        return {
            'image': np.reshape(cv2.resize(cv2.cvtColor(ros_numpy.numpify(self.img_rgb), cv2.COLOR_BGR2RGB), (640,480)).astype(self.observation_space['image'].dtype)/255.0, self.observation_space['image'].shape),
            'agent_pos': np.array([*self.get_arm_pose(), self.get_gripper_opening()], dtype=self.observation_space['agent_pos'].dtype)
        }

    def reset(self):
        self.set_arm_pose(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.set_gripper_opening(0.0)
        return self._get_obs()
    
    def step(self, action):
        self.set_arm_pose(action[:6])
        self.set_gripper_opening(action[6])
        reward = 0.0
        done = False
        info = {}
        return self._get_obs(), reward, done, info
    
    def render(self, mode="human"):
        return cv2.cvtColor(ros_numpy.numpify(self.img_rgb), cv2.COLOR_BGR2RGB)
    
    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            if self.teleop:
                act = np.array([*self.get_arm_pose(), self.get_gripper_opening()])
                act[0] += pygame.joystick.Joystick(0).get_axis(0)*0.05
                act[1] += pygame.joystick.Joystick(0).get_axis(1)*0.05
                act[2] += pygame.joystick.Joystick(0).get_axis(2)*0.05
                act[3] += pygame.joystick.Joystick(0).get_axis(3)*0.05
                act[4] += pygame.joystick.Joystick(0).get_axis(4)*0.05
                act[5] = math.atan2(act[1], act[0])
            return act
        return TeleopAgent(act)