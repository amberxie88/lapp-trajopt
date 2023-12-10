from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.arms.xarm7 import XArm7
from pyrep.robots.end_effectors.xarm_gripper import XArmGripper
from pyrep.robots.arms.ur5 import UR5
from pyrep.robots.end_effectors.robotiq85_gripper import Robotiq85Gripper
# from rlbench.backend.robot import Robot
from pyrep.objects.object import Object
from pyrep.const import ObjectType
from pyrep.objects.vision_sensor import VisionSensor

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape, RenderMode
from pyrep.errors import ConfigurationPathError
from pyrep.backend import sim

import numpy as np
import math
import random
import transforms3d
import csv
import matplotlib.pyplot as plt
from utils import read_shapenet
from pathlib import Path
import io
import yaml

SCENE_FILE = join(dirname(abspath(__file__)),
                  'assets/original_scene_xarm.ttt')
RLB_PATH = "/PATH/TO/RLBench"
PYREP_PATH = "/PATH/TO/PyRep"

from pyrep.const import ObjectType, TextureMappingMode
TEX_KWARGS = {
    'mapping_mode': TextureMappingMode.PLANE,
    'repeat_along_u': False,
    'repeat_along_v': False,
    'uv_scaling': [10., 10.]
}

class ShapenetEnv(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=True)
        self.pr.start()
        arm = XArm7()
        self.agent = arm
        for arm_obj in arm.get_objects_in_tree():
            if 'visual' in arm_obj.get_name():
                arm_obj.set_renderable(True)
            else: 
                arm_obj.set_renderable(False)

        self.agent.set_motor_locked_at_zero_velocity(True)
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()

        self.camera = VisionSensor('my_vision_sensor')
        if cfg.camera[0] == "zoom":
            self.camera.set_position([1.5, 0, 1.5])
            self.camera.set_orientation([-math.pi, -50 / 180 * math.pi, math.pi/2])
            self.camera.set_render_mode(RenderMode.OPENGL) 
        elif cfg.camera[0] == "standard":
            self.camera.set_render_mode(RenderMode.OPENGL) 
        else:
            raise NotImplementedError

        # pointcloud stuff
        self.bounds = np.array([[0, -0.75, 0.6], [1, 0.75, 1.5]])
        self.add_offset = np.array([[-0.5, 0, -0.4]])
        
        self.extra_cameras, self.extra_seg_cameras = [], []
        for cam_name in cfg.camera[1:]:
            if cam_name == "zoom":
                new_camera = VisionSensor.create(self.camera.resolution, 
                                                position=[1.5, 0, 1.5], 
                                                orientation=[-math.pi, -50 / 180 * math.pi, math.pi/2])
                new_camera.set_render_mode(RenderMode.OPENGL) 
                self.extra_cameras.append(new_camera)
            elif cam_name == "zoom_left":
                new_camera = VisionSensor.create(self.camera.resolution, 
                                                position=[1.5, -0.5, 1.5], 
                                                orientation=[-140 / 180 * math.pi, -40 / 180 * math.pi, 140 / 180 * math.pi])
                new_camera.set_render_mode(RenderMode.OPENGL) 
                self.extra_cameras.append(new_camera)
            elif cam_name == "zoom_right":
                new_camera = VisionSensor.create(self.camera.resolution, 
                                                position=[1.5, 0.5, 1.5], 
                                                orientation=[140 / 180 * math.pi, -40 / 180 * math.pi, 40 / 180 * math.pi])
                new_camera.set_render_mode(RenderMode.OPENGL) 
                self.extra_cameras.append(new_camera)
            elif cam_name == "standard":
                raise NotImplementedError 
                self.extra_cameras.append(new_camera)
            elif cam_name == "zoom_seg":
                seg_camera = VisionSensor.create(self.camera.resolution, 
                                                position=[1.5, 0, 1.5], 
                                                orientation=[-math.pi, -50 / 180 * math.pi, math.pi/2],
                                                render_mode=RenderMode.OPENGL_COLOR_CODED)
                self.extra_seg_cameras.append(seg_camera)
            else:
                raise NotImplementedError

        # load cfg.env_cfg yaml file    
        with open(cfg.env_cfg, 'r') as stream:  
            self.env_cfg = yaml.safe_load(stream)   
        shapenet_to_generate = set([shape['type'] for shape in self.env_cfg['shapes']])
        self.table_coll_cuboid = Shape('TableCollCuboid')

        self.shapes = []
        self.shapenet_objs = read_shapenet(shapenet_to_generate)

        self.reset_cfg()
        self.render_resolution = self.camera.get_resolution()
        self.grid_states = None

        self.radius = cfg.radius

    def get_seg_meta(self):
        handles = []
        for shape in self.shapes:
            handles.append(shape.get_handle())
        return handles

    def get_pyrep_goal(self):
        desired_joints = []
        temp = []
        all_states, all_colls, all_obs, all_extra_obs = [], [], [], [[] for _ in range(len(self.cfg.camera)-1)]
       
        self.shape_idx = self.env_cfg['motionplanning']['shape_idx']
        cpos = self.shapes[self.shape_idx].get_position()
        radius = self.radius
       
        pos1 = cpos + np.array(self.env_cfg['motionplanning']['start_offset'])
        pos2 = cpos + np.array(self.env_cfg['motionplanning']['end_offset'])
        pos_lst = [pos1, pos2]

        for pos in pos_lst:
            states, colls, obs, extra_obs = self.get_path_collision_data(pos)
            all_states.extend(states)
            all_colls.extend(colls)
            all_obs.extend(obs)
            for i in range(len(extra_obs)):
                all_extra_obs[i].extend(extra_obs[i])
            desired_joints.append(states[-1])
            temp.append(all_obs[-1])

        meta = self.get_meta()
        extra_obs_dict = dict()
        for idx, cam_name in enumerate(self.cfg.camera[1:]):
            extra_obs_dict[cam_name] = all_extra_obs[idx]
        use_seg = any(["seg" in cam for cam in self.cfg.camera])
        if use_seg:
            # add handles
            meta['handles'] = self.get_seg_meta()
        meta.update(dict(all_obs=all_obs, all_states=all_states, all_colls=all_colls))
        meta.update(extra_obs_dict)
        return desired_joints, meta

    def reset_cfg(self):
        self.agent.set_joint_positions(self.initial_joint_positions)

        for shape in self.shapes:
            shape.remove()
        self.shapes = []

        n_objects = len(self.env_cfg['shapes'])
        for idx in range(n_objects):
            curr_obj = self.env_cfg['shapes'][idx]
            shapenet_class = self.shapenet_objs[curr_obj['type']]
            shapenet_obj = shapenet_class[curr_obj['id']]

            pid_x, pid_y, pid_z = curr_obj['position']
            
            shape = Shape.import_shape(shapenet_obj.path, scaling_factor=0.35)
            shape.set_position([pid_x, pid_y, pid_z])
            shape.set_collidable(True)
            shape.set_dynamic(False)
            self.rotate_shape(shape)
            shape.name = shapenet_obj.name 
            shape.parent_name = shapenet_obj.parent_name
            self.shapes.append(shape)
        return self._get_state()

    def rotate_shape(self, shape):
        euler = self.euler_world_to_shape(shape, [math.pi/2, 0, 0])
        shape.rotate(euler)

    def euler_world_to_shape(self, shape, euler):
        m = sim.simGetObjectMatrix(shape._handle, -1)
        x_axis = np.array([m[0], m[4], m[8]])
        y_axis = np.array([m[1], m[5], m[9]])
        z_axis = np.array([m[2], m[6], m[10]])
        euler = np.array([math.pi/2, 0, 0])
        R = transforms3d.euler.euler2mat(*euler, axes='rxyz')
        T = np.array([x_axis, y_axis, z_axis]).T
        new_R = np.linalg.inv(T)@R@T
        new_euler = transforms3d.euler.mat2euler(new_R, axes='rxyz')
        return new_euler

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return np.concatenate([self.agent.get_joint_positions()])

    def _get_joint_positions(self):
        return self.agent.get_joint_positions()

    def get_path_collision_data(self, pos, euler=[0, math.radians(180), 0]):
        try:
            path = self.agent.get_path(position=pos, euler=euler, ignore_collisions=True)
        except ConfigurationPathError as e:
            return None

        states = []
        colls = []
        obs = []
        extra_obs = [[] for _ in range(len(self.extra_cameras) + len(self.extra_seg_cameras))]

        done_path = False 
        while not done_path:
            done_path = path.step()
            self.pr.step()
            states.append(self._get_state())
            coll = []
            for shape in self.shapes:
                if self.agent.check_arm_collision(shape):
                    coll.append(1)
                else:
                    coll.append(0)
            colls.append(coll)
            obs.append(self.render())
            for cam_id in range(len(self.extra_cameras)):
                extra_obs[cam_id].append(self.render_extra(cam_id))
            for cam_id in range(len(self.extra_seg_cameras)):
                extra_obs[cam_id + len(self.extra_cameras)].append(self.get_masks(cam_id))

        return states, colls, obs, extra_obs

    def render(self, null=False):
        img_arr = self.camera.capture_rgb() 
        img = (img_arr * 255).astype(np.uint8)
        return img_arr

    def render_extra(self, cam_id):
        img_arr = self.extra_cameras[cam_id].capture_rgb() 
        img = (img_arr * 255).astype(np.uint8)
        return img_arr

    def get_masks(self, cam_id):
        img_arr = self.extra_seg_cameras[cam_id].capture_rgb()
        ids = img_arr[:,:,0]+256*img_arr[:,:,1]+256*256*img_arr[:,:,2]
        ids = 255 * ids
        return ids

    def get_meta(self):
        positions, names, parent_names = [], [], []
        for shape in self.shapes:
            positions.append(shape.get_position())
            names.append(shape.name)
            parent_names.append(shape.parent_name)
        positions = np.array(positions)
        names = np.array(names)
        meta = dict(positions=positions, names=names, parent_names=parent_names)
        return meta

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()