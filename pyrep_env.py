import os
import numpy as np
import math
from problem.problem import Problem

import torch

class PyrepWrapper:
    def __init__(self, env, cfg):
        self.env = env 
        self.cfg = cfg
        self.DOF = 7
        self.laco_model = None

    def close(self):
        self.env.shutdown()

    def init_problem(self):
        desired_joints, meta = self.env.get_pyrep_goal()
        img = self.env.render().transpose(2,0,1)
        try:
            vit_obs = self.image_processor.preprocess(img)['pixel_values'][0]
            self.vit_obs = torch.from_numpy(vit_obs).unsqueeze(0).to(self.cfg.device)
        except:
            pass
        problem = Problem("problem", desired_joints[0], desired_joints[1], self.cfg.problem.num_steps,
                len(desired_joints[0]), self.cfg.problem.min_velocity, self.cfg.problem.max_velocity,
                self.cfg.problem.min_dist, -3.5, 3.5)
        problem.lang = self.env.env_cfg['lang']['lang_str']
        return problem, meta

    def set_state(self, action):
        self.env.agent.set_joint_positions(action, disable_dynamics=True)
        self.env.pr.step()

    def add_laco_model(self, model, image_processor):
        self.laco_model = model
        self.image_processor = image_processor

    def render(self):
        return (self.env.render()*254).astype(np.uint8).transpose(2,0,1)

    def get_collision_info(self, traj_lst, d_check, lang, get_gt=False):
        traj_lst = traj_lst.reshape(-1, self.DOF)
        gradient_lst = []
        dist_lst = []
        gt_lst = []
        for state in traj_lst:
            if self.laco_model is None:
                self.env.agent.set_joint_positions(state, disable_dynamics=True)
                self.env.pr.step()
                if self.env.agent.check_arm_collision(self.env.shapes[self.env.shape_idx]):
                    dist_lst.append(1)
                else:
                    dist_lst.append(0)
                gradient_lst.append(np.random.rand(self.DOF))
            else:
                state_th = torch.from_numpy(state).unsqueeze(0)
                state_th.requires_grad = True
                data = dict(vit_obs=self.vit_obs, language=[lang], states=state_th)
                pred = self.laco_model.forward(data)
                dist_lst.append((1-pred[0][0].item()))

                if get_gt: 
                    current_state = self.env.agent.get_joint_positions()
                    self.env.agent.set_joint_positions(state, disable_dynamics=True)
                    self.env.pr.step()
                    col = 0
                    for idx in range(len(self.env.shapes)):
                        if idx in self.env.env_cfg['lang']['indices_mask']:
                            continue
                        if self.env.agent.check_arm_collision(self.env.shapes[idx]):
                            col = 1
                            break
                    gt_lst.append(col)
                    self.env.agent.set_joint_positions(current_state, disable_dynamics=True)
                    self.env.pr.step()
                    
                grad = torch.autograd.grad(-pred[:, 0], state_th)
                gradient_lst.append(grad[0][0].cpu().numpy())
        gradients = np.array(gradient_lst)
        distances = np.array(dist_lst)
        gt_list = np.array(gt_lst)
        return dict(raw_gradients=gradients, raw_distances=distances, gt_collision=gt_list)


def load_environment(cfg):
    name = cfg.env_name
    if name == "pyrep_shapenet":
        from envs.shapenet_env import ShapenetEnv
        env = ShapenetEnv(cfg) 
    else:
        raise NotImplementedError
    return env

def make(cfg):
    env = load_environment(cfg)
    env = PyrepWrapper(env, cfg)
    env.height = 256
    return env