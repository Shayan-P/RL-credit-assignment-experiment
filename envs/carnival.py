import stable_baselines3  # registers atari games
import gymnasium as gym


def get_carnival_env():
    return gym.make('Carnival-v4')

# ['Carnival-v0',
#  'CarnivalDeterministic-v0',
#  'CarnivalNoFrameskip-v0',
#  'Carnival-v4',
#  'CarnivalDeterministic-v4',
#  'CarnivalNoFrameskip-v4',
#  'Carnival-ram-v0',
#  'Carnival-ramDeterministic-v0',
#  'Carnival-ramNoFrameskip-v0',
#  'Carnival-ram-v4',
#  'Carnival-ramDeterministic-v4',
#  'Carnival-ramNoFrameskip-v4',
#  'ALE/Carnival-v5',
#  'ALE/Carnival-ram-v5']
