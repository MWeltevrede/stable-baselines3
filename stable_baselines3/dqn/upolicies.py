from typing import Any, Dict, List, Optional, Type

import gymnasium as gym
import torch as th
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
        self.beta = beta
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
        q_values = self.q_net(obs)
        if self.beta == 0:
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

                return q_values + self.beta * (uncertainties + novelties)
            else:
                return q_values + self.beta * uncertainties

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        if deterministic:
            # use only Q, not U
            values = self.q_net(obs)
        else:
            values = self(obs)
        # Greedy action
        action = values.argmax(dim=1).reshape(-1)
        return action