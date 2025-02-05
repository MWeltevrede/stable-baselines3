from typing import Any, Dict, Optional, Tuple, Type, Union, List

import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from copy import deepcopy

from stable_baselines3.common.ubuffers import ExploreGoUncertaintyReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.upolicies import UncertaintyMlpPolicy


class UncertaintyDQN(DQN):
    """
    Deep Q-Network (DQN) using uncertainty
    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param beta: The scaling factor of the intrinsic rewards.
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param double_q: whether to use double dqn
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[UncertaintyMlpPolicy]],
        env: Union[GymEnv, str],
        beta: float = 0,
        uncertainty = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        u_tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ExploreGoUncertaintyReplayBuffer]] = None, 
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        double_q: bool = False,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        lam: float = 1,
        alpha: float = 0,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        max_pure_expl_steps: int = 0,
    ):
        self.double_q = double_q
        self.beta = beta
        self.uncertainty = uncertainty
        self.u_tau = u_tau

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            # double_q=double_q,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        if self.n_envs > 1:
            self.betas = np.array([beta * lam ** (1 + (k / (self.n_envs-1))*alpha) for k in range(self.n_envs)])
        else:
            self.betas = np.array(beta)
        
        self.max_pure_expl_steps = max_pure_expl_steps
        self.num_pure_expl_steps = np.random.randint(0, max_pure_expl_steps+1 , size=env.num_envs)
        self.episode_steps = np.zeros(env.num_envs)
        self.num_normal_steps = 0

    def _create_aliases(self) -> None:
        super()._create_aliases()
        self.u_net = self.policy.u_net
        self.u_net_target = self.policy.u_net_target
        self.policy._set_uncertainty(self.uncertainty)

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        super()._on_step()
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.u_net.parameters(), self.u_net_target.parameters(), self.u_tau)

        self.logger.record("rollout/beta", self.beta)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        u_losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                if self.double_q:
                    # Compute the next Q-values using the current network
                    next_q_values_current = self.q_net(replay_data.next_observations)
                    # Determine argmax based on the current network values
                    actions = next_q_values_current.max(dim=1)[1].unsqueeze(dim=1)
                    next_q_values = next_q_values.gather(dim=1, index=actions)
                else:
                    actions = next_q_values.max(dim=1)[1].unsqueeze(dim=1)
                    next_q_values = next_q_values.gather(dim=1, index=actions)

                    # 1-step TD target
                    target_q_values = replay_data.rewards[0] + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item())

            if not np.all(self.betas == 0):
                with th.no_grad():
                    if self.uncertainty is not None:
                        next_obs_shape = replay_data.next_observations.shape
                        actions = th.as_tensor(range(self.action_space.n), device=self.device).repeat(next_obs_shape[0]).unsqueeze(1)
                        next_obs_repeated = th.repeat_interleave(replay_data.next_observations, self.action_space.n, dim=0)
                        novelties = th.concatenate([ir for ir in replay_data.rewards[1:]], dim=-1) * self.uncertainty(next_obs_repeated, actions, global_only=True).reshape(next_obs_shape[0], -1)

                    # Compute the next uncertainties using the target network
                    next_u_values = self.u_net_target(replay_data.next_observations)
                    if self.double_q:
                        # Compute the next Q-values using the current network
                        next_u_values_current = self.u_net(replay_data.next_observations)
                        # Determine argmax based on the current network values
                        if self.uncertainty is not None:
                            actions = (next_u_values_current + novelties).max(dim=1)[1].unsqueeze(dim=1)
                        else:
                            actions = next_u_values_current.max(dim=1)[1].unsqueeze(dim=1)
                    else:
                        if self.uncertainty is not None:
                            actions = (next_u_values + novelties).max(dim=1)[1].unsqueeze(dim=1)
                        else:
                            actions = next_u_values.max(dim=1)[1].unsqueeze(dim=1)

                    if self.uncertainty is not None:
                        next_u_values = (next_u_values + novelties).gather(dim=1, index=actions)
                        # 1-step TD target
                        # target_u_values = (1 - replay_data.dones) * self.gamma * next_u_values
                        target_u_values = self.gamma * next_u_values
                    else:
                        next_u_values = next_u_values.gather(dim=1, index=actions)
                        # 1-step TD target
                        # target_u_values = replay_data.rewards[1] + (1 - replay_data.dones) * self.gamma * next_u_values
                        target_u_values = replay_data.rewards[1] + self.gamma * next_u_values
                

                # Get current uncertainty estimates
                current_u_values = self.u_net(replay_data.observations)

                # Retrieve the uncertainties for the actions from the replay buffer
                current_u_values = th.gather(current_u_values, dim=1, index=replay_data.actions.long())

                # Compute Huber loss (less sensitive to outliers)
                u_loss = F.smooth_l1_loss(current_u_values, target_u_values)

                # Optimize the policy
                self.policy.u_optimizer.zero_grad()
                u_loss.backward()
                # Clip gradient norm
                th.nn.utils.clip_grad_norm_(self.policy.u_net.parameters(), self.max_grad_norm)
                self.policy.u_optimizer.step()
                u_losses.append(u_loss.item())
            else:
                u_losses.append(0)


        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/u_loss", np.mean(u_losses))

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ['replay_buffer_kwargs']
    
    ###
    ### Explore-Go Changes
    ###

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            self.num_normal_steps += n_envs
            normal_inds = np.ones(self.episode_steps.shape, dtype=np.bool_)
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action = self.policy._predict_pure(self._last_obs)
            normal_inds = self.episode_steps >= self.num_pure_expl_steps
            if sum(normal_inds) > 0:
                unscaled_action_normal, _ = self.predict(self._last_obs, deterministic=False)
                unscaled_action[normal_inds] = unscaled_action_normal[normal_inds]
            self.num_normal_steps += sum(normal_inds)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action, normal_inds
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ExploreGoUncertaintyReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions, normal_inds = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # If the last step of the pure exploration phase, set done to True
            buffer_dones = deepcopy(dones)
            last_pure_indices = self.episode_steps == (self.num_pure_expl_steps - 1)
            buffer_dones[last_pure_indices] = np.array([True for _ in range(last_pure_indices.sum())])

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, buffer_dones, infos, normal_inds)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            self.episode_steps += 1
            for idx, done in enumerate(dones):
                if done:
                    self.episode_steps[idx] = 0
                    self.num_pure_expl_steps[idx] = np.random.randint(0, self.max_pure_expl_steps+1)
                    
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
    
    def _store_transition(
        self,
        replay_buffer: ExploreGoUncertaintyReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        normal_inds: np.ndarray,
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type]
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
            normal_inds,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_