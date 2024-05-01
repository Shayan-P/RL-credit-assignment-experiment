import gymnasium as gym
from bsuite import bsuite
from bsuite.utils import gym_wrapper
from bsuite.utils.gym_wrapper import GymFromDMEnv


def get_umbrella_length_env() -> gym.Env:
    # change later. what are other versions of umbrella length
    # + can we visualize it?
    bs_env = bsuite.load_from_id('umbrella_length/0')
    return gym_wrapper.GymFromDMEnv(bs_env)
