from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from utils.utils import discount_cumsum
from data.convertor import Convertor


@dataclass
class TrajectoryData:
    """
        feature space
    """
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns: np.ndarray
    dones: np.ndarray


class TrajectoryDataset:
    def __init__(self, gamma=1):
        self.called_item = False
        self.storage = []

        self._max_episode_length = 0
        self.gamma = gamma

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
        for i in range(self.dataset_size()):
            self.storage.append(self.get_item(i))
        self._max_episode_length = max(len(traj.observations) for traj in self.storage)

    def __getitem__(self, item):
        self.cache_if_still_havent()
        return self.storage[item]

    def get_max_episode_length(self):
        self.cache_if_still_havent()
        return self._max_episode_length

    @abstractmethod
    def get_reward_scale(self):
        pass

    @abstractmethod
    def state_dim(self):
        pass

    @abstractmethod
    def action_dim(self):
        pass

    @property
    @abstractmethod
    def state_convertor(self) -> Convertor:
        pass

    @property
    @abstractmethod
    def action_convertor(self) -> Convertor:
        pass

    @property
    @abstractmethod
    def reward_convertor(self) -> Convertor:
        pass

    def collect_trajectory(self, env, policy, step_limit=1000):
        obs, _ = env.reset()
        observation, info = env.reset()
        observations, actions, rewards, dones = [], [], [], []
        for _ in range(step_limit):
            action, info = policy.predict(observation)
            action = action.item()
            observation, reward, terminated, truncated, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            if terminated or truncated:
                break
        rewards = np.array(rewards)
        returns = discount_cumsum(rewards, self.gamma)
        return observations, actions, rewards, returns, dones

    def update_return_to_go(self, rtg, reward):
        return (rtg - reward) / self.gamma
