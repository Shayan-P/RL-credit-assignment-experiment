from abc import abstractmethod

import gym
import numpy as np
import torch

# class RandomPolicy(BasePolicy):
#     def __init__(self, env: gym.Env):
#         super(BasePolicy, self).__init__(action_space=env.action_space, observation_space=env.observation_space)
#         self.ob_space = env.observation_space
#         self.ac_space = env.action_space

#     def _predict(self, obs, deterministic=False):
#         batch = obs.shape[0]
#         return torch.tensor([self.ac_space.sample() for _ in range(batch)])

class BasePolicy:
    @abstractmethod
    def predict(self, obs):
        pass

    def reset(self):
        """
        should be called at the start of each episode
        """
        pass

    def add_to_history(self, obs, action, reward, done):
        """
            this is necessary because some policies need to see the reward of their previous actions
        """
        pass


class RandomPolicy(BasePolicy):
    def __init__(self, env: gym.Env):
        self.ob_space = env.observation_space
        self.ac_space = env.action_space


    def predict(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        if not isinstance(obs, torch.Tensor): # should be primitives right?
            obs = torch.tensor(obs)
        if len(obs.shape) == len(self.ob_space.shape): # no vectorization:
            return self.ac_space.sample()
        else:
            l = len(self.ob_space.shape)
            assert len(obs.shape) == l+1
            assert obs.shape[-l:] == self.ob_space.shape
            rest = tuple(obs.shape[:-l])
            return torch.tensor([self.ac_space.sample()
                                 for _ in range(np.prod(rest))]).reshape(rest + self.ac_space.shape)

    def reset(self):
        pass

        # batch = obs.shape[0] if isinstance(obs, torch.Tensor) else 1
        # return torch.tensor([self.ac_space.sample() for _ in range(batch)]), {}

# def sweep(engine_class, agents, probs, labels, n_runs=2000, max_steps=500):
#     logs = dict()
#     pbar = tqdm(agents)
#     for idx, agent in enumerate(pbar):
#         pbar.set_description(f'Alg:{labels[idx]}')
#         # maybe we can get engine from the algorithm itself?
#         engine = engine_class(probs=probs, max_steps=max_steps, agent=agent)
#         ep_log = engine.run(n_runs)
#         ep_log = pd.concat(ep_log, ignore_index=True)
#         ep_log['Alg'] = labels[idx]
#         logs[f'{labels[idx]}'] = ep_log
#     logs = pd.concat(logs, ignore_index=True)
#     return logs
