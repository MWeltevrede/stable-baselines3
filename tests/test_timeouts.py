import gym
import numpy as np
import pytest
from gym import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


class DummyEnv(gym.Env):
    """
    Custom gym environment for testing the handling of timeouts
    """

    def __init__(self):
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(0, 1, (1,))
        self._observations = [0, 1]

    def reset(self):
        obs = self._observations[0]
        return obs

    def step(self, action):
        obs = self._observations[1]
        reward = 0
        done = True
        info = {"TimeLimit.truncated": True}
        return obs, reward, done, info


@pytest.mark.parametrize("optimize_memory_usage", [False, True])
def test_terminal_observation(optimize_memory_usage):
    env = DummyEnv
    env = make_vec_env(env)
    env = VecNormalize(env, norm_obs=False)

    buffer = ReplayBuffer(
        3,
        env.observation_space,
        env.action_space,
        optimize_memory_usage=optimize_memory_usage,
        handle_timeout_termination=True,
    )

    # Interract and store transitions
    obs = env.reset()
    for _ in range(2):
        action = np.array([env.action_space.sample()])
        next_obs, reward, done, info = env.step(action)

        if done[0]:
            next_obs[0] = info[0]["terminal_observation"]
        buffer.add(obs, next_obs, action, reward, done, info)

        if done:
            obs = env.reset()
        else:
            obs = next_obs

    sample = buffer._get_samples(np.array([0]), env=env)
    next_obs = env.unnormalize_obs(sample.next_observations[0])
    assert next_obs == 1
