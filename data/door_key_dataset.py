import torch
import numpy as np

from envs.door_key import MiniGridEnv
from algorithms.random_policy import RandomPolicy
from data.trajectory import TrajectoryDataset, TrajectoryData
from data.convertor import DiscreteStateConvertor, RewardConvertor, DiscreteActionConvertor, \
    Convertor
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper


class DoorKeyDataset(TrajectoryDataset):
    def __init__(self, env: MiniGridEnv, n_trajectories=10000, reward_scale=None):
        super().__init__(gamma=1)

        # todo this is not clean. later register the env with it's wrapper
        if not isinstance(env, ImgObsWrapper):
            env = ImgObsWrapper(env)

        self.env = env
        policy = RandomPolicy(self.env)
        self.trajectories = []
        self.trajectories += self.collect_trajectories(self.env, policy, n_trajectories=n_trajectories)
        # todo other policies maybe?

        self.original_obs_shape = env.observation_space.shape
        self.flattened_obs_dim = np.prod(self.original_obs_shape)

        all_state_features = []
        all_returns = []
        for observations, actions, rewards, returns, dones in self.trajectories:
            all_state_features.extend(list(observations))
            all_returns.append(returns[0])
        all_returns = np.array(all_returns)
        all_state_features = np.array(all_state_features)
        self.state_mean = np.mean(all_state_features, axis=0)
        self.state_std = np.std(all_state_features, axis=0)

        if reward_scale is None:
            self.reward_scale = np.abs(np.mean(all_returns)) + np.std(all_returns)  #  todo or maybe a consider the interval (including min)
        else:
            self.reward_scale = reward_scale

        self._state_convertor = DoorKeyCustomStateConvertor(self.original_obs_shape, self.state_mean, self.state_std)
        self._reward_convertor = RewardConvertor(self.reward_scale)
        self._action_convertor = DiscreteActionConvertor(self.env.action_space)

        print("Dataset Info:")
        print('episode_max_length:', self.get_max_episode_length())
        print('reward_scale:', self.reward_scale)
        print(f'return min={all_returns.min()}, max={all_returns.max()} mean={all_returns.mean()}')
        # print('state_mean:', self.state_mean)
        # print('state_std:', self.state_std)
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

    def get_item(self, item) -> TrajectoryData:
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
        return self.flattened_obs_dim

    def action_dim(self):
        return self.env.action_space.n


class DoorKeyCustomStateConvertor(Convertor):
    def __init__(self, shape, state_mean, state_std):
        super().__init__()
        self.shape = shape
        self.state_mean = state_mean
        self.state_std = state_std
        self.state_std = np.where(self.state_std == 0, 1e-7, self.state_std)  # prevent overflow

    def to_feature_space(self, value):
        value = (value - self.state_mean) / self.state_std
        d = len(self.shape)
        value = value.reshape(list(value.shape[:-d]) + [-1])
        return value

    def from_feature_space(self, value):
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        value = value.reshape(list(value.shape[:-1]) + list(self.shape))
        value = value * self.state_std + self.state_mean
        return value
