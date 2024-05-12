import torch

from algorithms.random_policy import BasePolicy
from data.trajectory import TrajectoryDataset


def evaluate_policy(policy: BasePolicy, env, num_eval_ep, max_test_ep_len=1000, render=False):
    policy.reset()

    total_reward = 0
    total_timesteps = 0

    with torch.no_grad():
        for _ in range(num_eval_ep):
            obs, _ = env.reset()
            policy.reset()

            for t in range(max_test_ep_len):
                action = policy.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                policy.add_to_history(obs, action, reward, done)
                total_reward += reward
                total_timesteps += 1
                if done:
                    break

    results = {
        'eval/avg_reward': total_reward / num_eval_ep,
        'eval/avg_ep_len': total_timesteps / num_eval_ep
    }
    return results
