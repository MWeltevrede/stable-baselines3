import numpy as np
import torch as th
import queue

from typing import Any, Dict, List

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
        episodic_discount=False,
        split_uncertainty=False,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination
        )

        self.step_count = 0
        self.uncertainty = uncertainty
        self.env = env
        self.device = device
        self.recently_added_transitions = set()
        self.state_action_bonus = state_action_bonus
        self.uncertainty_of_sampling = uncertainty_of_sampling
        self.episodic_discount = episodic_discount
        self.split_uncertainty = split_uncertainty
        assert not (self.episodic_discount and self.uncertainty_of_sampling), "Episodic sampling and uncertainty of buffer sampling is not supported."
        assert not (self.split_uncertainty and not self.state_action_bonus), "Split uncertainty can only be done with a state-action bonus."

        if self.episodic_discount:
            if self.split_uncertainty:
                self.rewards = np.zeros((self.buffer_size, self.n_envs, 1 + self.action_space.n), dtype=np.float32)
            else:
                self.rewards = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)



    def add(self, obs, next_obs, action, reward, done, infos):
        if self.step_count < 500_000:
            normalise = True
        else:
            normalise = False

        if not self.uncertainty == "egreedy":
            if not self.uncertainty_of_sampling:
                if self.state_action_bonus:
                    if self.episodic_discount:
                        if self.split_uncertainty:
                            self.uncertainty.observe(obs, action, done, update_rms=normalise)

                            actions = th.as_tensor(range(self.action_space.n), device=self.device).repeat(obs.shape[0]).unsqueeze(1)
                            obs_repeated = th.repeat_interleave(th.as_tensor(next_obs, device=self.device), self.action_space.n, dim=0)
                            intrinsic_reward = self.uncertainty(obs_repeated, actions).reshape(obs.shape[0], -1).detach().cpu().numpy()
                            reward = np.concatenate([np.expand_dims(reward, axis=-1), intrinsic_reward], axis=1)
                        else:
                            intrinsic_reward = self.uncertainty.observe(obs, action, done, update_rms=normalise)
                            reward = np.stack([reward, intrinsic_reward], axis=1)
                    else:
                        self.uncertainty.observe(obs, action, update_rms=normalise)
                else:
                    if self.episodic_discount:
                        intrinsic_reward = self.uncertainty.observe(next_obs, done, update_rms=normalise)
                        reward = np.stack([reward, intrinsic_reward], axis=1)
                    else:
                        self.uncertainty.observe(next_obs, update_rms=normalise)
            else:
                if normalise:
                    if self.state_action_bonus:
                        self.uncertainty.update_rms(obs, action)
                    else:
                        self.uncertainty.update_rms(next_obs)
        
        super().add(obs, next_obs, action, reward, done, infos)
        self.step_count += self.n_envs

    def skip_add(self, obs, next_obs, action, reward, done, infos):
        super().add(obs, next_obs, action, reward, done, infos)
        self.step_count += self.n_envs

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

        real_rewards = sampled_batch.rewards
        if not self.uncertainty == "egreedy":
            with th.no_grad():
                if self.state_action_bonus:
                    if self.episodic_discount:
                        if self.split_uncertainty:
                            intrinsic_rewards = []
                            for i in range(self.action_space.n):
                                intrinsic_rewards.append(sampled_batch.rewards.reshape(-1, 1 + self.action_space.n)[:, 1+i].unsqueeze(-1))
                            real_rewards = sampled_batch.rewards.reshape(-1, 1 + self.action_space.n)[:, 0].unsqueeze(-1)
                        else:
                            intrinsic_rewards = sampled_batch.rewards.reshape(-1, 2)[:, 1].unsqueeze(-1)
                            intrinsic_rewards = intrinsic_rewards * self.uncertainty(sampled_batch.observations, sampled_batch.actions, global_only=True).unsqueeze(-1)
                            real_rewards = sampled_batch.rewards.reshape(-1, 2)[:, 0].unsqueeze(-1)
                    else:
                        intrinsic_rewards = self.uncertainty(sampled_batch.observations, sampled_batch.actions).unsqueeze(dim=-1)
                else:
                    if self.episodic_discount:
                        intrinsic_rewards = sampled_batch.rewards.reshape(-1, 2)[:, 1].unsqueeze(-1)
                        intrinsic_rewards = intrinsic_rewards * self.uncertainty(sampled_batch.next_observations).unsqueeze(-1)
                        real_rewards = sampled_batch.rewards.reshape(-1, 2)[:, 0].unsqueeze(-1)
                    else:
                        intrinsic_rewards = self.uncertainty(sampled_batch.next_observations).unsqueeze(dim=-1)
        else:
            intrinsic_rewards = real_rewards
        if self.split_uncertainty:
            sampled_batch = sampled_batch._replace(rewards=th.stack([real_rewards, *intrinsic_rewards], dim=0))
        else:
            sampled_batch = sampled_batch._replace(rewards=th.stack([real_rewards, intrinsic_rewards], dim=0))

        if self.uncertainty_of_sampling and not self.uncertainty == "egreedy":
            # our uncertainty measures epistemic uncertainty with respect to what we have sampled for training
            if self.state_action_bonus:
                self.uncertainty.observe(sampled_batch.observations, sampled_batch.actions, update_rms=False)
            else:
                self.uncertainty.observe(sampled_batch.next_observations, update_rms=False)

        return sampled_batch

class ExploreGoUncertaintyReplayBuffer(UncertaintyReplayBuffer):
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
        episodic_discount=False,
        split_uncertainty=False,
        include_pure_experience=False,
    ):
        assert optimize_memory_usage == False, "Optimize_memory_usage has to be False."
        super().__init__(
            buffer_size, 
            observation_space, 
            action_space, 
            uncertainty=uncertainty,
            env=env,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
            state_action_bonus=True,
            uncertainty_of_sampling=False,  # If false, we calculate epistemic uncertainty in environment collection instead of buffer sampling
            episodic_discount=True,
            split_uncertainty=True,)
        
        self.experience_queue = queue.Queue()
        self.include_pure_experience = include_pure_experience

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        normal_inds: np.ndarray,
    ) -> None:
        if self.step_count < 500_000:
            normalise = True
        else:
            normalise = False
        if not self.uncertainty == "egreedy":
            self.uncertainty.observe(obs, action, done, update_rms=normalise)

            actions = th.as_tensor(range(self.action_space.n), device=self.device).repeat(obs.shape[0]).unsqueeze(1)
            obs_repeated = th.repeat_interleave(th.as_tensor(next_obs, device=self.device), self.action_space.n, dim=0)
            intrinsic_reward = self.uncertainty(obs_repeated, actions).reshape(obs.shape[0], -1).detach().cpu().numpy()
            reward = np.concatenate([np.expand_dims(reward, axis=-1), intrinsic_reward], axis=1)
            
        if self.include_pure_experience:
              super().add(obs, next_obs, action, reward, done, infos)
        else:
            for i in range(obs.shape[0]):
                # First add normal (non-pure) experience to the experience queue
                if normal_inds[i] == True:
                    experience_tuple = (obs[i], next_obs[i], action[i], reward[i], done[i], infos[i])
                    self.experience_queue.put(experience_tuple)

            # Add experience to the buffer once enough has been collected
            if self.experience_queue.qsize() >= self.n_envs:
                obs_list = []
                next_obs_list = []
                action_list = []
                reward_list = []
                done_list = []
                infos_list = []
                for _ in range(self.n_envs):
                    experience_tuple = self.experience_queue.get()
                    obs_list.append(experience_tuple[0])
                    next_obs_list.append(experience_tuple[1])
                    action_list.append(experience_tuple[2])
                    reward_list.append(experience_tuple[3])
                    done_list.append(experience_tuple[4])
                    infos_list.append(experience_tuple[5])
                obs_list = np.stack(obs_list, axis=0)
                next_obs_list = np.stack(next_obs_list, axis=0)
                action_list = np.stack(action_list, axis=0)
                reward_list = np.stack(reward_list, axis=0)
                done_list = np.stack(done_list, axis=0)

                super().skip_add(obs_list, next_obs_list, action_list, reward_list, done_list, infos_list)

