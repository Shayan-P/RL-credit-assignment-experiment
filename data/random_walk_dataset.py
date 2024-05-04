import torch
import numpy as np

from envs.random_walk import RandomWalkEnv
from algorithms.random_policy import RandomPolicy
from data.trajectory import TrajectoryDataset, TrajectoryData


class RandomWalkDataset(TrajectoryDataset):
    def __init__(self, n_trajectories=10000):
        super().__init__()

        self.env = RandomWalkEnv()
        policy = RandomPolicy(self.env)
        self.trajectories = []
        self.trajectories += self.collect_trajectories(self.env, policy, n_trajectories=n_trajectories)
        # todo other policies maybe?

    def dataset_size(self):
        return len(self.trajectories)

    def one_hot_action(self, acts):
        # todo change .n if we change the env
        ret = np.zeros((len(acts), self.env.action_space.n))
        ret[np.arange(len(acts)), acts] = 1
        return ret

    def one_hot_state(self, obs):
        # todo change .n if we change the env
        ret = np.zeros((len(obs), self.env.observation_space.n))
        ret[np.arange(len(obs)), obs] = 1
        return ret

    def get_item(self, item):
        observations, actions, rewards, dones = self.trajectories[item]
        # extract features here
        return TrajectoryData(
            observations=self.one_hot_state(observations),
            actions=self.one_hot_action(actions),
            rewards=np.array(rewards),
            dones=np.array(dones)
        )

    @staticmethod
    def collect_trajectories(env, policy, n_trajectories, step_limit=1000):
        return [
            RandomWalkDataset.collect_trajectory(env, policy, step_limit=step_limit) for _ in range(n_trajectories)
        ]

    @staticmethod
    def collect_trajectory(env, policy, step_limit=1000):
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
        return observations, actions, rewards, dones

    def get_reward_scale(self):
        # todo perhaps this should be based on the best path in the graph?
        return 10

    # todo maybe we should do reward normalization in the dataset itself?

