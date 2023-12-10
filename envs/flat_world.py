import matplotlib.pyplot as plt
import numpy as np
import yaml

from enum import Enum 

class RelativePos(Enum):
    OVERLAP = 0
    TOP_LEFT = 1
    BOTTOM_LEFT = 2
    TOP_RIGHT = 3 
    BOTTOM_RIGHT = 4 
    LEFT = 5
    RIGHT = 6 
    TOP = 7
    BOTTOM = 8

class FlatWorld:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.obstacle_list = self.cfg['obstacles']
        self.agent = self.cfg['agent']
        self.agent['actual_pos'] = self.agent['pos']
        self.height, self.width = self.cfg['height'], self.cfg['width']

    def set_state(self, action):
        """
        Action is just the x, y position of the agent. 
        """
        self.agent['actual_pos'] = action.copy()
        self.agent['pos'][0] = int(action[0])
        self.agent['pos'][1] = int(action[1])

    def get_collision_info(self, traj_lst, d_check):
        traj_lst = traj_lst.reshape(-1, 2)
        gradient_lst = []
        dist_lst = []
        for agent_pos in traj_lst:
            self.step(agent_pos)
            gradient_item = np.zeros(2)
            min_dist = np.inf
            for obs_id in range(len(self.obstacle_list)):
                dist, gradients, relative_pos = self.sdf(obs_id)
                
                if dist < min_dist:
                    gradient_item = gradients
                    min_dist = dist
            gradient_lst.append(gradient_item)
            dist_lst.append(min_dist)
        gradients = np.array(gradient_lst)
        distances = np.array(dist_lst)
        return dict(raw_gradients=gradients, raw_distances=distances)


    def sdf(self, obs_id):
        agent_x, agent_y = self.agent['actual_pos']
        agent_width, agent_height = self.agent['size']
        obs_x, obs_y = self.obstacle_list[obs_id]['pos']
        obs_width, obs_height = self.obstacle_list[obs_id]['size']

        agent_x1, agent_y1, agent_x2, agent_y2 = self.get_rectangle_points(agent_x, agent_y, agent_width, agent_height)

        obs_x1, obs_y1, obs_x2, obs_y2 = self.get_rectangle_points(obs_x, obs_y, obs_width, obs_height)

        relative_pos, dist = self.get_relative_pos_distance(agent_x1, agent_y1, agent_x2, agent_y2, obs_x1, obs_y1, obs_x2, obs_y2)
        fx, fy = self.get_gradients(relative_pos, agent_x1, agent_y1, agent_x2, agent_y2, obs_x1, obs_y1, obs_x2, obs_y2)
        gradients = np.array([fx, fy])

        return dist, gradients, relative_pos

    def get_relative_pos_distance(self, agent_x1, agent_y1, agent_x2, agent_y2, obs_x1, obs_y1, obs_x2, obs_y2):
        assert agent_x2 > agent_x1 and obs_x2 > obs_x1
        assert agent_y2 < agent_y1 and obs_y2 < obs_y1

        # the obstacle is ____ relative to the agent
        left = obs_x2 < agent_x1
        right = agent_x2 < obs_x1 
        top = agent_y1 < obs_y2
        bottom = obs_y1 < agent_y2
        if top and left:
            distance = self.distance((agent_x1, agent_y1), (obs_x2, obs_y2))
            return RelativePos.TOP_LEFT, distance
        elif top and right:
            distance = self.distance((agent_x2, agent_y1), (obs_x1, obs_y2))
            return RelativePos.TOP_RIGHT, distance
        elif bottom and left:
            distance = self.distance((agent_x1, agent_y2), (obs_x2, obs_y1))
            return RelativePos.BOTTOM_LEFT, distance
        elif bottom and right:
            distance = self.distance((agent_x2, agent_y2), (obs_x1, obs_y1))
            return RelativePos.BOTTOM_RIGHT, distance
        elif left:
            distance = agent_x1 - obs_x2
            return RelativePos.LEFT, distance
        elif right:
            distance = obs_x1 - agent_x2
            return RelativePos.RIGHT, distance
        elif top:
            distance = obs_y2 - agent_y1
            return RelativePos.TOP, distance
        elif bottom:
            distance = agent_y2 - obs_y1
            return RelativePos.BOTTOM, distance
        else:
            # technically not SDF. Accurate SDF calculation better for gradients though...
            distance = 0
            return RelativePos.OVERLAP, distance

    def get_gradients(self, rel_pos, agent_x1, agent_y1, agent_x2, agent_y2, obs_x1, obs_y1, obs_x2, obs_y2):
        if rel_pos == RelativePos.OVERLAP:
            return 1, 0
        elif rel_pos == RelativePos.LEFT:
            return 1, 0
        elif rel_pos == RelativePos.RIGHT:
            return -1, 0 
        elif rel_pos == RelativePos.TOP:
            return 0, -1
        elif rel_pos == RelativePos.BOTTOM:
            return 0, 1
        elif rel_pos == RelativePos.TOP_LEFT:
            p1, p2 = (agent_x1, agent_y1), (obs_x2, obs_y2)
        elif rel_pos == RelativePos.BOTTOM_LEFT:
            p1, p2 = (agent_x1, agent_y2), (obs_x2, obs_y1)
        elif rel_pos == RelativePos.TOP_RIGHT:
            p1, p2 = (agent_x2, agent_y1), (obs_x1, obs_y2)
        elif rel_pos == RelativePos.BOTTOM_RIGHT:
            p1, p2 = (agent_x2, agent_y2), (obs_x1, obs_y1)
        else:
            raise NotImplementedError

        common = 1 / (np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1]) **2) + 1e-6)
        fx = common * (p1[0] - p2[0])
        fy = common * (p1[1] - p2[1])
        return fx, fy

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_rectangle_points(self, x, y, w, h):
        x1, y1 = x, y 
        x2, y2 = x+w, y-h
        return x1, y1, x2, y2

    def render(self):
        image = np.zeros((3, self.cfg['height'], self.cfg['width']), dtype=np.uint8)
        for obstacle in self.obstacle_list:
            x, y = obstacle['pos']
            width, height = obstacle['size']
            image[0, self.height-y:self.height-y+height, x:x+width] = 255
        agent_x, agent_y = self.agent['pos']
        agent_width, agent_height = self.agent['size']
        image[1, self.height-agent_y:self.height-agent_y+agent_height, agent_x:agent_x+agent_width] = 255
        return image

