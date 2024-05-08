import torch
import numpy as np

from utils.utils import get_one_hot
from envs.random_walk import RandomWalkEnv
from algorithms.random_policy import RandomPolicy
from data.trajectory import TrajectoryDataset, TrajectoryData
from utils.utils import discount_cumsum
from data.convertor import StateConvertor, RewardConvertor, ActionConvertor


class RandomWalkDataset(TrajectoryDataset):
    def __init__(self, n_trajectories=10000, reward_scale=None):
        super().__init__(gamma=1)

        self.env = RandomWalkEnv()
        policy = RandomPolicy(self.env)
        self.trajectories = []
        self.trajectories += self.collect_trajectories(self.env, policy, n_trajectories=n_trajectories)
        # todo other policies maybe?

        # todo perhaps this should be based on the best path in the graph?
        all_state_features = []
        all_returns = []
        for observations, actions, rewards, returns, dones in self.trajectories:
            all_state_features.extend([get_one_hot(obs, self.state_dim()) for obs in observations])
            all_returns.extend(returns)
        all_returns = np.array(all_returns)
        all_state_features = np.array(all_state_features)
        self.state_mean = np.mean(all_state_features, axis=0)
        self.state_std = np.std(all_state_features, axis=0)

        if reward_scale is None:
            self.reward_scale = np.max(all_returns)  # todo or maybe a consider the interval (including min)
        else:
            self.reward_scale = reward_scale

        self._state_convertor = StateConvertor(self.env.observation_space, self.state_mean, self.state_std)
        self._reward_convertor = RewardConvertor(self.reward_scale)
        self._action_convertor = ActionConvertor(self.env.action_space)

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
            o, a, r, rt, d = self.transform_to_feature_space(o, a, r, rt, d)
            observations_features.append(o)
            actions_features.append(a)
            rewards_features.append(r)
            returns_features.append(rt)
            dones_features.append(d)

        return TrajectoryData(
            observations=np.array(observations_features),
            actions=np.array(actions_features),
            rewards=np.array(rewards_features),
            returns=np.array(returns_features),
            dones=np.array(dones_features)
        )

    def transform_to_feature_space(self, observation, action, reward, rt, done):
        observations_feature, actions_feature, rewards_feature, return_features, dones_feature = (
            (get_one_hot(observation, self.state_dim()) - self.state_mean) / self.state_std,
            get_one_hot(action, self.action_dim()),
            reward / self.reward_scale,
            rt / self.reward_scale,
            done
        )
        return observations_feature, actions_feature, rewards_feature, return_features, dones_feature

    def transform_from_feature_space(self, observations_feature, actions_feature, rewards_feature, return_features, dones_feature):
        if isinstance(observations_feature, torch.Tensor):
            observations_feature = observations_feature.cpu().numpy()
        if isinstance(actions_feature, torch.Tensor):
            actions_feature = actions_feature.cpu().numpy()
        if isinstance(rewards_feature, torch.Tensor):
            rewards_feature = rewards_feature.cpu().numpy()
        if isinstance(return_features, torch.Tensor):
            return_features = return_features.cpu().numpy()
        if isinstance(dones_feature, torch.Tensor):
            dones_feature = dones_feature.cpu().numpy()

        observation, action, reward, rt, done = (
            np.argmax(observations_feature, axis=-1),
            np.argmax(actions_feature, axis=-1),
            rewards_feature * self.reward_scale,
            return_features * self.reward_scale,
            dones_feature
        )
        return observation, action, reward, rt, done

    def collect_trajectories(self, env, policy, n_trajectories, step_limit=1000):
        return [
            self.collect_trajectory(env=env, policy=policy, step_limit=step_limit) for _ in range(n_trajectories)
        ]

    def get_reward_scale(self):
        return self.reward_scale

    def state_dim(self):
        return self.env.observation_space.n

    def action_dim(self):
        return self.env.action_space.n

    # def one_hot_action(self, acts):
    #     # todo change .n if we change the env
    #     ret = np.zeros((len(acts), self.env.action_space.n))
    #     ret[np.arange(len(acts)), acts] = 1
    #     return ret
    #
    # def one_hot_state(self, obs):
    #     # todo change .n if we change the env
    #     ret = np.zeros((len(obs), self.env.observation_space.n))
    #     ret[np.arange(len(obs)), obs] = 1
    #     return ret
