#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# the original version comes from:
# https://github.com/ShangtongZhang/DeepRL

# below code has been modified to my suit my needs

import numpy as np
from baselines.common.running_mean_std import RunningMeanStd
import torch
from config import Config

class MeanStdNormalizer:
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):

        self.read_only = read_only
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1, ) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)


class RescaleNormalizer:
    def __init__(self, coef=1.0):

        self.coef = coef

    def __call__(self, x):
        x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        RescaleNormalizer.__init__(self, 1.0 / 255)

class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            assert k in self.keys
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)

def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, device=Config.DEVICE, dtype=torch.float32)
    return x