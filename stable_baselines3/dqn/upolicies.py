from typing import Any, Dict, List, Optional, Type

import gymnasium as gym
import torch as th
import numpy as np
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.utils import get_schedule_fn

class UncertaintyMlpPolicy(DQNPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        beta: float,
        u_lr: float,
        n_envs: int,
        lam: float = 1,
        alpha: float = 0,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.u_net, self.u_net_target = None, None
        u_lr_schedule = get_schedule_fn(u_lr)
        self._build_unet(u_lr_schedule)
        self.betas = th.tensor([beta * lam ** (1 + (k / (n_envs-1))*alpha) for k in range(n_envs)])
        self.uncertainty = None

    def _set_uncertainty(self, uncertainty):
        self.uncertainty = uncertainty

    def _build_unet(self, lr_schedule: Schedule) -> None:
        self.u_net = self.make_q_net()
        self.u_net_target = self.make_q_net()
        self.u_net_target.load_state_dict(self.u_net.state_dict())
        self.u_net_target.set_training_mode(False)

        self.u_optimizer = self.optimizer_class(self.u_net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        if not self.betas.device == self.device:
            self.betas = self.betas.to(self.device)
        q_values = self.q_net(obs)
        if th.all(self.betas == 0):
            return q_values
        else:
            uncertainties = self.u_net(obs)

            if self.uncertainty is not None:
                if len(obs.shape) == 1 or len(obs.shape) == 3:
                    # there is no batch dimension
                    no_batch_dim = True
                    # novelties = th.zeros((self.action_space.n), device=obs.device)
                    obs = obs.unsqueeze(0)
                else:
                    no_batch_dim = False
                
                actions = th.as_tensor(range(self.action_space.n), device=self.device).repeat(obs.shape[0]).unsqueeze(1)
                obs_repeated = th.repeat_interleave(obs, self.action_space.n, dim=0)
                novelties = self.uncertainty(obs_repeated, actions).reshape(obs.shape[0], uncertainties.shape[-1])

                if no_batch_dim:
                    novelties.squeeze(0)

                # assume that if obs.shape[0] is smaller than self.betas.shape[0], we are in a setting where beta is the same everywhere
                if obs.shape[0] == self.betas.shape[0]:
                    return q_values + self.betas.unsqueeze(-1) * (uncertainties + novelties)
                else:
                    return q_values + self.betas[0] * (uncertainties + novelties)
            else:
                if obs.shape[0] == self.betas.shape[0]:
                    return q_values + self.betas.unsqueeze(-1) * uncertainties
                else:
                    return q_values + self.betas[0] * uncertainties

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        if deterministic:
            # use only Q, not U
            values = self.q_net(obs)
        else:
            values = self(obs)
        # Greedy action
        action = values.argmax(dim=1).reshape(-1)
        return action
    
    def _predict_pure(self, obs: th.Tensor) -> th.Tensor:
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )
        
        obs_tensor, vectorized_env = self.obs_to_tensor(obs)

        with th.no_grad():
            if th.all(self.betas == 0):
                # use only Q, not U
                values = self.q_net(obs_tensor)
            else:
                uncertainties = self.u_net(obs_tensor)
                if self.uncertainty is not None:
                    if len(obs_tensor.shape) == 1 or len(obs_tensor.shape) == 3:
                        # there is no batch dimension
                        no_batch_dim = True
                        # novelties = th.zeros((self.action_space.n), device=obs.device)
                        obs_tensor = obs_tensor.unsqueeze(0)
                    else:
                        no_batch_dim = False
                    
                    actions = th.as_tensor(range(self.action_space.n), device=self.device).repeat(obs_tensor.shape[0]).unsqueeze(1)
                    obs_repeated = th.repeat_interleave(obs_tensor, self.action_space.n, dim=0)
                    novelties = self.uncertainty(obs_repeated, actions).reshape(obs_tensor.shape[0], uncertainties.shape[-1])

                    if no_batch_dim:
                        novelties.squeeze(0)

                    values = uncertainties + novelties
                else:
                    values = uncertainties

            # Greedy pure exploration action
            pure_action = values.argmax(dim=1).reshape(-1)

        # Convert to numpy, and reshape to the original action shape
        pure_action = pure_action.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                pure_action = self.unscale_action(pure_action)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                pure_action = np.clip(pure_action, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(pure_action, np.ndarray)
            pure_action = pure_action.squeeze(axis=0)

        return pure_action
