import gym
import torch

from gymnasium.spaces import Space
from stable_baselines3.common.policies import BasePolicy

# class RandomPolicy(BasePolicy):
#     def __init__(self, env: gym.Env):
#         super(BasePolicy, self).__init__(action_space=env.action_space, observation_space=env.observation_space)
#         self.ob_space = env.observation_space
#         self.ac_space = env.action_space

#     def _predict(self, obs, deterministic=False):
#         batch = obs.shape[0]
#         return torch.tensor([self.ac_space.sample() for _ in range(batch)])

class RandomPolicy:
    def __init__(self, env: gym.Env):
        self.ob_space = env.observation_space
        self.ac_space = env.action_space

    def predict(self, obs):
        batch = obs.shape[0] if isinstance(obs, torch.Tensor) else 1
        return torch.tensor([self.ac_space.sample() for _ in range(batch)]), {}

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
