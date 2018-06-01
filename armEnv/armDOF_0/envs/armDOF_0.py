import numpy as np
import math
import os
from gym import utils
from gym.envs.mujoco import mujoco_env

class ArmDOF_0Env(mujoco_env.MujocoEnv, utils.EzPickle):
    #Initialize environment
    def __init__(self):
        utils.EzPickle.__init__(self)
        FILE_PATH = os.path.dirname(os.path.realpath(__file__)) + '/assets/armDOF_0.xml'
        mujoco_env.MujocoEnv.__init__(self, FILE_PATH, 1)

    #Define step function for the environment to allow a prescribed action to alter the observations.
    #Calculates distance between grip and object coordinates and uses this to calculate reward.
    #The normalized action that has been executed is also part of the reward function to prevent shakey movements
    # in the arm when it has reached its goal.
    # (1) was used to calculate what action roughly leads to 1 degree of rotation in a joint.
    def _step(self, a):
        vector = self.get_body_com("grip")-self.get_body_com("object")
        self.do_simulation(a, self.frame_skip)
        ob=self._get_obs()
        ans = np.linalg.norm(vector)
        #(1) angle = math.degrees(math.atan(self.get_body_com("DOF2")[2] - self.get_body_com("DOF1")[2] / self.get_body_com("DOF2")[0] - self.get_body_com("DOF1")[0]))
        #print (angle)
        #print (a)
        #string = str(90 -angle) + '__' + str(self.model.data.qpos.ravel())
        #print (string)

        #reflex_reward = 0
        #if (self.get_body_com("DOF2"))[2] < (self.get_body_com("DOF1"))[2]:
        #    reflex_reward = (self.get_body_com("DOF1"))[2] - (self.get_body_com("DOF2"))[2]
        reward = - (100 * (ans) ** 2) - np.square(a).sum()# - 100 * abs(reflex_reward)
        return ob, reward, False, {}

    # Used to reset the model after each episode.
    # Every episode the position of the object is randomized within a range.
    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.goal = np.append(self.np_random.uniform(low=-0.05, high=0.05, size=1),self.np_random.uniform(low=0, high=0.1, size=1))
        self.goal = np.append(self.np_random.uniform(low=0.084, high=0.16, size=1),self.goal)
        qpos[-3:] = self.goal
        qvel[-3:] = 0 
        self.set_state(qpos, qvel)
        return self._get_obs()

    # Defines observation space.
    def _get_obs(self):
        dof1_coord = self.get_body_com("DOF1")
        dof2_coord = self.get_body_com("DOF2")
        wrist_coord = self.get_body_com("wrist")
        grip_coord = self.get_body_com("grip")
        obj_coord = self.get_body_com("object")

        #print(dof1_coord)
        #print(dof2_coord)
        #print(wrist_coord)
        #print(grip_coord)
        #print(obj_coord)
        return np.concatenate([dof1_coord.ravel(), dof2_coord.ravel(), wrist_coord.ravel(), grip_coord.ravel(), obj_coord.ravel()]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent + 0.5

