from abc import abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryData:
    # todo make sure you're passing the feature vectors of each of these
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def normalize(self, state_mean, state_std, reward_scale):
        self.observations = (self.observations - state_mean) / state_std
        self.rewards /= reward_scale


class TrajectoryDataset:
    def __init__(self, normalize_reward=True, normalize_state=True):
        self.called_item = False
        self.normalize_reward = normalize_reward
        self.normalize_state = normalize_state
        self.storage = []

        self._max_episode_length = 0
        self._reward_scale = 0
        self._state_mean = None
        self._state_std = None

    @abstractmethod
    def dataset_size(self) -> int:
        pass

    def __len__(self):
        return self.dataset_size()

    @abstractmethod
    def get_item(self, idx: int) -> TrajectoryData:
        pass

    def cache_if_still_havent(self):
        if self.called_item:
            return
        self.called_item = True
        all_states = []

        for i in range(self.dataset_size()):
            self.storage.append(self.get_item(i))
            all_states.extend(self.storage[-1].observations)

        self._state_mean, self._state_std = np.mean(all_states), np.std(all_states)
        self._reward_scale = self.get_reward_scale()

        for traj in self.storage:
            traj.normalize(self._state_mean, self._state_std, self._reward_scale)

        self._max_episode_length = max(len(traj.observations) for traj in self.storage)

    def __getitem__(self, item):
        self.cache_if_still_havent()
        return self.storage[item]

    def get_max_episode_length(self):
        self.cache_if_still_havent()
        return max(len(self.__getitem__(i).observations) for i in range(self.dataset_size()))

    @abstractmethod
    def get_reward_scale(self):
        pass

    def state_dim(self):
        self.cache_if_still_havent()
        return self.storage[0].observations[0].shape[-1]

    def action_dim(self):
        self.cache_if_still_havent()
        return self.storage[0].action[0].shape[-1]
