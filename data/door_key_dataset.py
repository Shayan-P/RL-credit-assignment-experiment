import torch
import numpy as np

from utils.utils import get_one_hot
from envs.door_key import DoorKeyEnv, DoorKeyEnvSmall, MiniGridEnv
from algorithms.random_policy import RandomPolicy
from data.trajectory import TrajectoryDataset, TrajectoryData
from utils.utils import discount_cumsum
from data.convertor import StateConvertor, RewardConvertor, ActionConvertor


class DoorKeyDataset(TrajectoryDataset):
    def __init__(self, env: MiniGridEnv, n_trajectories=10000, reward_scale=None):
        super().__init__(gamma=1)

        self.env = env
        policy = RandomPolicy(self.env)
        self.trajectories = []
        self.trajectories += self.collect_trajectories(self.env, policy, n_trajectories=n_trajectories)
        # todo other policies maybe?

        raise Exception("not implemented yet")

        print("Dataset Info:")
        print('episode_max_length:', self.get_max_episode_length())
        print('reward_scale:', self.reward_scale)
        print(f'return min={all_returns.min()}, max={all_returns.max()} mean={all_returns.mean()}')
        print('state_mean:', self.state_mean)
        print('state_std:', self.state_std)
        print('gamma:', self.gamma)

    @property
    def state_convertor(self):
        return self._state_convertor

    @property
    def reward_convertor(self):
        return self._reward_convertor

    @property
    def action_convertor(self):
        return self._action_convertor

    def dataset_size(self):
        return len(self.trajectories)

    def get_item(self, item):
        observations, actions, rewards, returns, dones = self.trajectories[item]
        observations_features, actions_features, rewards_features, returns_features, dones_features = [], [], [], [], []
        for o, a, r, rt, d in zip(observations, actions, rewards, returns, dones):
            observations_features.append(self.state_convertor.to_feature_space(o))
            actions_features.append(self.action_convertor.to_feature_space(a))
            rewards_features.append(self.reward_convertor.to_feature_space(r))
            returns_features.append(self.reward_convertor.to_feature_space(rt))
            dones_features.append(d)

        return TrajectoryData(
            observations=np.array(observations_features),
            actions=np.array(actions_features),
            rewards=np.array(rewards_features),
            returns=np.array(returns_features),
            dones=np.array(dones_features)
        )

    def get_reward_scale(self):
        return self.reward_scale

    def state_dim(self):
        return self.env.observation_space.n

    def action_dim(self):
        return self.env.action_space.n
