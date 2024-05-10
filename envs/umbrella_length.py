import gymnasium as gym
from bsuite import bsuite
from bsuite.utils import gym_wrapper
from bsuite.utils.gym_wrapper import GymFromDMEnv


def get_umbrella_length_env(chain_length) -> gym.Env:
    # change later. what are other versions of umbrella length
    # + can we visualize it?
    return gym.make("bsuite/umbrella_length-v0", chain_length=chain_length, n_distractor=20)
