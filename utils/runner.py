def run_for_episode(env, policy, num_episodes=1, step_limit=1000):
    observation, info = env.reset()
    for _ in range(step_limit):
        action = policy.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def run_for_steps(env, policy, step_limit=1000):
    observation, _ = env.reset()
    for _ in range(step_limit):
        action = policy.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
