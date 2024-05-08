import torch
import numpy as np

from abc import abstractmethod

from utils.utils import get_one_hot


class Convertor:
    def __init__(self):
        pass

    @abstractmethod
    def from_feature_space(self, value):
        pass

    @abstractmethod
    def to_feature_space(self, value):
        pass


class RewardConvertor(Convertor):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def from_feature_space(self, value):
        return value * self.scale

    def to_feature_space(self, value):
        return value / self.scale


class ActionConvertor(Convertor):
    def __init__(self, action_space):
        super().__init__()
        self.n = action_space.n

    def to_feature_space(self, value):
        assert(0 <= value < self.n)
        return get_one_hot(value, self.n)

    def from_feature_space(self, value):
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        return np.argmax(value, axis=-1)


# todo this is only for discrete spaces
class StateConvertor(Convertor):
    def __init__(self, state_space, state_mean, state_std):
        super().__init__()
        self.n = state_space.n
        self.state_mean = state_mean
        self.state_std = state_std

    def to_feature_space(self, value):
        assert (0 <= value < self.n)
        return (get_one_hot(value, self.n) - self.state_mean) / self.state_std

    def from_feature_space(self, value):
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        value = value * self.state_std + self.state_mean
        return np.argmax(value, axis=-1)
