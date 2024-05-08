import random
import numpy as np
import torch

from abc import abstractmethod
from dataclasses import dataclass
from utils.utils import discount_cumsum
from data.convertor import Convertor
from torch.utils.data import Dataset


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

    def __len__(self):
        return len(self.observations)


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
        traj: TrajectoryData = self.storage[item]
        return traj

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


class LimitedContextWrapper(Dataset):
    def __init__(self, trajectory_dataset: TrajectoryDataset, context_len):

        self.context_len = context_len
        self.trajectory_dataset = trajectory_dataset

    def __len__(self):
        return len(self.trajectory_dataset)

    def __getitem__(self, idx):
        traj = self.trajectory_dataset[idx]
        traj_len = len(traj)

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj.observations[si: si + self.context_len])
            actions = torch.from_numpy(traj.actions[si: si + self.context_len])
            returns_to_go = torch.from_numpy(traj.returns[si: si + self.context_len])
            timesteps = torch.arange(start=si, end=si + self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj.observations)
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                            dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj.actions)
            actions = torch.cat([actions,
                                 torch.zeros(([padding_len] + list(actions.shape[1:])),
                                             dtype=actions.dtype)],
                                dim=0)

            returns_to_go = torch.from_numpy(traj.returns)
            returns_to_go = torch.cat([returns_to_go,
                                       torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                                   dtype=returns_to_go.dtype)],
                                      dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        states = states.float()
        actions = actions.float()
        returns_to_go = returns_to_go.float()
        return timesteps, states, actions, returns_to_go, traj_mask
