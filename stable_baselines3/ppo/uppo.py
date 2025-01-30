import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from copy import deepcopy

from stable_baselines3.common.buffers import RolloutBuffer, AsyncRolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm, ExploreGoOnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common import utils
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

SelfPPO = TypeVar("SelfPPO", bound="UncertaintyPPO")


class UncertaintyPPO(PPO):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version) using intrinsic rewards

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        uncertainty = None,
        beta: float = 0,
        pure_exploration: bool = False,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=_init_setup_model,
        )

        self.beta = beta
        self.uncertainty = uncertainty
        self.pure_exploration = pure_exploration

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
        ) -> bool:
            """
            Collect experiences using the current policy and fill a ``RolloutBuffer``.
            The term rollout here refers to the model-free notion and should not
            be used with the concept of rollout used in model-based RL or planning.

            :param env: The training environment
            :param callback: Callback that will be called at each step
                (and at the beginning and end of the rollout)
            :param rollout_buffer: Buffer to fill with rollouts
            :param n_rollout_steps: Number of experiences to collect per environment
            :return: True if function returned with at least `n_rollout_steps`
                collected, False if callback terminated rollout prematurely.
            """
            assert self._last_obs is not None, "No previous observation was provided"
            # Switch to eval mode (this affects batch norm / dropout)
            self.policy.set_training_mode(False)

            n_steps = 0
            rollout_buffer.reset()
            # Sample new weights for the state dependent exploration
            if self.use_sde:
                self.policy.reset_noise(env.num_envs)

            callback.on_rollout_start()

            while n_steps < n_rollout_steps:
                if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.policy.reset_noise(env.num_envs)

                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions

                if isinstance(self.action_space, spaces.Box):
                    if self.policy.squash_output:
                        # Unscale the actions to match env bounds
                        # if they were previously squashed (scaled in [-1, 1])
                        clipped_actions = self.policy.unscale_action(clipped_actions)
                    else:
                        # Otherwise, clip the actions to avoid out of bound error
                        # as we are sampling from an unbounded Gaussian distribution
                        clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

                new_obs, rewards, dones, infos = env.step(clipped_actions)

                if self.num_timesteps < 500_000 or self.pure_exploration:
                    normalise = True
                else:
                    normalise = False
                self.uncertainty.observe(self._last_obs, clipped_actions, dones, update_rms=normalise)
                intrinsic_rewards = self.uncertainty(self._last_obs, clipped_actions).detach().cpu().numpy()

                if self.pure_exploration:
                    rewards = self.beta * intrinsic_rewards
                else:
                    rewards += self.beta * intrinsic_rewards

                self.num_timesteps += env.num_envs

                # Give access to local variables
                callback.update_locals(locals())
                if not callback.on_step():
                    return False

                self._update_info_buffer(infos, dones)
                n_steps += 1

                if isinstance(self.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)

                # Handle timeout by bootstrapping with value function
                # see GitHub issue #633
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                        rewards[idx] += self.gamma * terminal_value

                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                )
                self._last_obs = new_obs  # type: ignore[assignment]
                self._last_episode_starts = dones

            with th.no_grad():
                # Compute value for the last timestep
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

            callback.update_locals(locals())

            callback.on_rollout_end()

            return True
    
    
class ExploreGoPPO(UncertaintyPPO):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version) using intrinsic rewards

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        uncertainty = None,
        pure_uncertainty = None,
        beta: float = 0,
        pure_beta: float = 0,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[AsyncRolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        max_pure_expl_steps: int = 0,
    ):
        super().__init__(
            policy,
            env,
            uncertainty=uncertainty,
            beta=beta,
            pure_exploration=False,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=_init_setup_model,
        )

        self.pure_agent = UncertaintyPPO(
            policy,
            env,
            uncertainty=pure_uncertainty,
            beta=pure_beta,
            pure_exploration=True,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=_init_setup_model,
        )

        self.max_pure_expl_steps = max_pure_expl_steps
        self.num_pure_expl_steps = np.random.randint(0, max_pure_expl_steps+1 , size=env.num_envs)
        self.episode_steps = np.zeros(env.num_envs)
        self.num_normal_steps = 0

        self.pure_beta = pure_beta


    def learn(
        self: PPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> PPO:
        iteration = 0
        normal_training_updates = 0
        pure_training_updates = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        self._pure_last_obs = deepcopy(self._last_obs)
        # _, _ = self.pure_agent._setup_learn(
        #     total_timesteps,
        #     callback,
        #     reset_num_timesteps,
        #     tb_log_name,
        #     progress_bar,
        # )
        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self.pure_agent._logger = utils.configure_logger(self.verbose, None, tb_log_name + '_pure', reset_num_timesteps)

        callback.on_training_start(locals(), globals())
        # pure_callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training, normal_inds = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=1)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % (log_interval * self.n_steps) == 0:
                assert self.ep_info_buffer is not None
                self.logger.record("train/num_normal_steps", self.num_normal_steps)
                self.logger.record("train/num_normal_updates", normal_training_updates)
                self.logger.record("train/num_pure_updates", pure_training_updates)
                self._dump_logs(iteration // self.n_steps)

            if self.rollout_buffer.full:
                with th.no_grad():
                    # Compute value for the last timestep
                    last_episode_starts = deepcopy(self._last_episode_starts)
                    last_episode_starts[~normal_inds] = np.ones((sum(~normal_inds),), dtype=bool)
                    values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))  # type: ignore[arg-type]
                self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=last_episode_starts)
                # callback.update_locals(locals())
                callback.on_rollout_end()

                self.train()
                normal_training_updates += 1

                self.rollout_buffer.reset()
                # Sample new weights for the state dependent exploration
                if self.use_sde:
                    self.policy.reset_noise(self.env.num_envs)
                callback.on_rollout_start()

            if self.pure_agent.rollout_buffer.full:
                with th.no_grad():
                    # Compute value for the last timestep
                    last_episode_starts_pure = deepcopy(self._last_episode_starts)
                    last_episode_starts_pure[normal_inds] = np.ones((sum(normal_inds),), dtype=bool)
                    values = self.pure_agent.policy.predict_values(obs_as_tensor(self._last_obs, self.device))  # type: ignore[arg-type]
                self.pure_agent.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=last_episode_starts_pure)
                # callback.update_locals(locals())
                # callback.on_rollout_end()

                self.pure_agent.train()
                pure_training_updates += 1

                self.pure_agent.rollout_buffer.reset()
                # Sample new weights for the state dependent exploration
                if self.pure_agent.use_sde:
                    self.pure_agent.policy.reset_noise(self.env.num_envs)
                # pure_callback.on_rollout_start()

        callback.on_training_end()

        return self
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: AsyncRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                pure_actions, pure_values, pure_log_probs = self.pure_agent.policy(obs_tensor)

            actions = actions.cpu().numpy()
            pure_actions = pure_actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            pure_clipped_actions = pure_actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                    pure_clipped_actions = self.pure_agent.policy.unscale_action(pure_clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
                    pure_clipped_actions = np.clip(pure_actions, self.action_space.low, self.action_space.high)

            pure_inds = self.episode_steps < self.num_pure_expl_steps
            normal_inds = self.episode_steps >= self.num_pure_expl_steps

            clipped_actions[pure_inds] = pure_clipped_actions[pure_inds]

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            self.num_normal_steps += sum(normal_inds)

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
                pure_actions = pure_actions.reshape(-1, 1)

            # If the last step of the pure exploration phase, set done to True
            buffer_dones = deepcopy(dones)
            last_pure_indices = self.episode_steps == (self.num_pure_expl_steps - 1)
            buffer_dones[last_pure_indices] = np.array([True for _ in range(last_pure_indices.sum())])
            self._pure_last_obs[last_pure_indices] = new_obs[last_pure_indices]

            if self.num_normal_steps < 500_000:
                normalise = True
            else:
                normalise = False
            if sum(normal_inds) > 0:
                self.uncertainty.observe(self._last_obs[normal_inds], clipped_actions[normal_inds], buffer_dones[normal_inds], update_rms=normalise, indices=np.where(normal_inds)[0])
                intrinsic_rewards = self.uncertainty(self._last_obs[normal_inds], clipped_actions[normal_inds], indices=np.where(normal_inds)[0]).detach().cpu().numpy()
                rewards[normal_inds] += self.beta * intrinsic_rewards

            if sum(pure_inds) > 0:
                self.pure_agent.uncertainty.observe(self._last_obs[pure_inds], pure_clipped_actions[pure_inds], buffer_dones[pure_inds], update_rms=True, indices=np.where(pure_inds)[0])
                pure_intrinsic_rewards = self.pure_agent.uncertainty(self._last_obs[pure_inds], pure_clipped_actions[pure_inds], indices=np.where(pure_inds)[0]).detach().cpu().numpy()
                rewards[pure_inds] = self.pure_agent.beta * pure_intrinsic_rewards

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        if normal_inds[idx]:
                            terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                        else:
                            terminal_value = self.pure_agent.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            if sum(last_pure_indices) > 0:
                with th.no_grad():
                    pure_terminal_value = self.pure_agent.policy.predict_values(self.pure_agent.policy.obs_to_tensor(new_obs)[0])[last_pure_indices]
                rewards[last_pure_indices] += self.gamma * pure_terminal_value.squeeze(-1).cpu().numpy()

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                normal_inds,
            )
            self.pure_agent.rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                pure_actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                pure_values,
                pure_log_probs,
                pure_inds,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = buffer_dones

            self.episode_steps += 1
            for idx, done in enumerate(dones):
                if done:
                    self.episode_steps[idx] = 0
                    self.num_pure_expl_steps[idx] = np.random.randint(0, self.max_pure_expl_steps+1)

        return True, normal_inds


