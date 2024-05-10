# from data.random_walk_dataset import RandomWalkDataset
# from envs.random_walk import RandomWalkEnv

# env = RandomWalkEnv(num_nodes=5, weight_max=10, reach_the_goal_reward=100, max_episode_length=1024)
# dataset = RandomWalkDataset(env= env, n_trajectories=1000)
# print(dataset.dataset_size())
# print(dataset.get_item(0))

from data.door_key_dataset import DoorKeyDataset
from envs.door_key import DoorKeyEnv

env = DoorKeyEnv()
dataset = DoorKeyDataset(env=env)
print(dataset.dataset_size())
print(dataset.get_item(0))