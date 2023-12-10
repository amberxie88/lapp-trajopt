import random
import re
import time
import math
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from typing import Any, NamedTuple
from dm_env import StepType, specs
from PIL import Image

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def chain(*iterables):
    for it in iterables:
        yield from it


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update_params(net, target_net):
    for (n, param), (tn, target_param) in zip(net.named_parameters(), target_net.named_parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def grad_norm(params, norm_type=2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


def param_norm(params, norm_type=2.0):
    total_norm = torch.norm(
        torch.stack([torch.norm(p.detach(), norm_type) for p in params]),
        norm_type)
    return total_norm.item()


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None or self._every == -1:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

class ShapeNetObj:
    def __init__(self, path, name, parent_name):
        self.path = path
        self.name = name
        self.parent_name = parent_name

    def __repr__(self):
        return f"{self.name}: {self.path}"

class ShapeNetClass:
    def __init__(self, base_names, synset_id, num_instances):
        self.base_names = base_names
        self.synset_id = synset_id
        self.num_instances = num_instances
        self.paths, self.names = get_shapenet_data(synset_id)

    def __getitem__(self, idx):
        if idx >= self.num_instances:
            raise IndexError()
        path = f"/PATH/TO/shapenetcore_v2/{self.synset_id}/{self.paths[idx]}/models/model_normalized.obj"
        name = self.names[idx][np.random.randint(len(self.names[idx]))]
        return ShapeNetObj(path, name, self.base_names[0])

    def get_random_shape(self):
        idx = np.random.randint(self.num_instances)
        return self[idx]

def get_shapenet_data(synset_id):
    csv_base_dir = "/PATH/TO/shapenet/metadata"
    csv_file = f"{csv_base_dir}/{synset_id}.csv"
    paths, names = [], []
    with open(csv_file, newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # first row of csv
            if row[0] == 'fullId':
                continue
            paths.append(row[0].split(".")[-1])
            names.append(row[2].split(','))
    return paths, names

def read_shapenet(shapenet_to_generate):
    json_file = "/PATH/TO/shapenet/shapenetcore_v2/taxonomy.json"

    with open(json_file) as json_data:
        data = json.load(json_data)

        all_objs = dict()
        all_children = []
        # keep track of children to process them last
        for obj in data:
            for child in obj['children']:
                all_children.append(child)

        # this object doesn't exist in the shapenet folder, not sure why
        all_children.append("02834778") # bicycle

        # process objects recursively
        for obj in data:
            if obj['synsetId'] not in all_children:
                name_lst = obj['name'].split(',')
                if len(shapenet_to_generate) > 0 and name_lst[0] not in shapenet_to_generate:
                    continue
                parent_obj = ShapeNetClass(name_lst, obj['synsetId'], obj['numInstances'])
                all_objs[name_lst[0]] = parent_obj

                # assert path exists
                # path, _ = parent_obj[np.random.randint(parent_obj.num_instances)]
                # assert os.path.exists(path), path
        return all_objs

def freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False

def get_var_from_kwargs(cls, key, optional=False, default=None, **kwargs):
    if key in kwargs:
        return kwargs[key]
    elif optional and default is not None:
        return default
    elif not optional:
        method_name = sys._getframe().f_back.f_code.co_name
        raise Exception("Method: \"" + method_name + "\" expects a parameter with key: " + str(key))
    else:
        return None