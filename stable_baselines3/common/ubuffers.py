import numpy as np
import torch as th

from stable_baselines3.common.buffers import ReplayBuffer


class UncertaintyReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        uncertainty="egreedy",
        env=None,
        device="cpu",
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        state_action_bonus=False,
        uncertainty_of_sampling=False,  # If false, we calculate epistemic uncertainty in environment collection instead of buffer sampling
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination
        )

        self.uncertainty = uncertainty
        self.env = env
        self.device = device
        self.recently_added_transitions = set()
        self.state_action_bonus = state_action_bonus
        self.uncertainty_of_sampling = uncertainty_of_sampling

    def add(self, obs, next_obs, action, reward, done, infos):
        if not self.uncertainty_of_sampling and not self.uncertainty == "egreedy":
            if self.state_action_bonus:
                self.uncertainty.observe(obs, action)
            else:
                self.uncertainty.observe(next_obs)
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size, env=None):
        if not self.optimize_memory_usage:
            sampled_batch = super().sample(batch_size=batch_size, env=env)
        else:
            # Do not sample the element with index `self.pos` as the transitions is invalid
            # (we use only one array to store `obs` and `next_obs`)
            if self.full:
                batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
            else:
                batch_inds = np.random.randint(0, self.pos, size=batch_size)
            sampled_batch = super()._get_samples(batch_inds, env=env)

        if not self.uncertainty == "egreedy":
            with th.no_grad():
                if self.state_action_bonus:
                    intrinsic_rewards = self.uncertainty(sampled_batch.observations, sampled_batch.actions).unsqueeze(dim=-1)
                else:
                    intrinsic_rewards = self.uncertainty(sampled_batch.next_observations).unsqueeze(dim=-1)

            real_rewards = sampled_batch.rewards
            sampled_batch = sampled_batch._replace(rewards=th.stack([real_rewards, intrinsic_rewards], dim=0))

        if self.uncertainty_of_sampling and not self.uncertainty == "egreedy":
            # our uncertainty measures epistemic uncertainty with respect to what we have sampled for training
            if self.state_action_bonus:
                self.uncertainty.observe(sampled_batch.observations, sampled_batch.actions)
            else:
                self.uncertainty.observe(sampled_batch.next_observations)

        return sampled_batch