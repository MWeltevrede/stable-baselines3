import torch as th
import numpy as np

class RNDUncertaintyState:
    """This class uses Random Network Distillation to estimate the uncertainty/novelty of states."""

    def __init__(self, scale, env, embed_dim, policy_kwargs, device="cpu", flatten_input=False, normalize_images=False, **kwargs):
        self.scale = scale
        self.criterion = th.nn.MSELoss(reduction="none")
        activation = policy_kwargs["activation_fn"]
        hidden_dims = policy_kwargs["net_arch"]
        learning_rate = policy_kwargs["learning_rate"]
        self.normalize_images = normalize_images

        if flatten_input:
            flattened_dim = np.prod(env.observation_space.shape)

        self.target_net = []
        if flatten_input:
            self.target_net.append(th.nn.Flatten())
            self.target_net.append(th.nn.Linear(flattened_dim, hidden_dims[0]))
        else:
            # input already flat
            self.target_net.append(th.nn.Linear(env.observation_space.shape[0], hidden_dims[0]))

        self.target_net.append(activation())
        for i in range(len(hidden_dims) - 1):
            self.target_net.append(th.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.target_net.append(activation())
        self.target_net.append(th.nn.Linear(hidden_dims[-1], embed_dim))
        self.target_net = th.nn.Sequential(*self.target_net).to(th.device(device))

        self.predict_net = []
        if flatten_input:
            self.predict_net.append(th.nn.Flatten())
            self.predict_net.append(th.nn.Linear(flattened_dim, hidden_dims[0]))
        else:
            # input already flat
            self.predict_net.append(th.nn.Linear(env.observation_space.shape[0], hidden_dims[0]))
        self.predict_net.append(activation())
        for i in range(len(hidden_dims) - 1):
            self.predict_net.append(th.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.predict_net.append(activation())
        self.predict_net.append(th.nn.Linear(hidden_dims[-1], embed_dim))
        self.predict_net = th.nn.Sequential(*self.predict_net).to(th.device(device))

        self.optimizer = th.optim.Adam(self.predict_net.parameters(), lr=learning_rate)

    def error(self, state):
        """Computes the error between the prediction and target network."""
        if not isinstance(state, th.Tensor):
            state = th.tensor(state)
        if len(state.shape) == 1:
            # need to add batch dimension to flat input
            state = state.unsqueeze(dim=0)
        if len(state.shape) == 3:
            # need to add batch dimension to image input
            state = state.unsqueeze(dim=0)
        state = state.float()

        if self.normalize_images:
            state = state / 255.

        return self.criterion(self.predict_net(state), self.target_net(state))

    def observe(self, state, **kwargs):
        """Observes state(s) and 'remembers' them using Random Network Distillation"""
        self.optimizer.zero_grad()
        self.error(state).mean().backward()
        self.optimizer.step()

    def __call__(self, state, **kwargs):
        """Returns the estimated uncertainty for observing a (minibatch of) state(s) as Tensor."""
        return self.error(state).mean(dim=-1)


class RNDUncertaintyStateAction:
    """This class uses Random Network Distillation to estimate the uncertainty/novelty of state-actions."""
    def __init__(self, scale, env, embed_dim, buffer_size, policy_kwargs, device="cpu", flatten_input=False, normalize_images=False, **kwargs):
        self.scale = scale
        self.criterion = th.nn.MSELoss(reduction="none")
        activation = policy_kwargs["activation_fn"]
        hidden_dims = policy_kwargs["net_arch"]
        learning_rate = policy_kwargs["learning_rate"]
        self.device=th.device(device)
        self.n_actions = env.action_space.n
        self.normalize_images = normalize_images

        self.target_net = []
        self.predict_net = []

        if flatten_input:
            flattened_dim = np.prod(env.observation_space.shape) + self.n_actions

        if flatten_input:
            self.target_net.append(th.nn.Linear(flattened_dim, hidden_dims[0]))
        else:
            # input already flat
            self.target_net.append(th.nn.Linear(env.observation_space.shape[0], hidden_dims[0]))

        self.target_net.append(activation())
        for i in range(len(hidden_dims) - 1):
            self.target_net.append(th.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.target_net.append(activation())
        self.target_net.append(th.nn.Linear(hidden_dims[-1], embed_dim))
        self.target_net = th.nn.Sequential(*self.target_net).to(th.device(device))

        self.predict_net = []
        if flatten_input:
            self.predict_net.append(th.nn.Linear(flattened_dim, hidden_dims[0]))
        else:
            # input already flat
            self.predict_net.append(th.nn.Linear(env.observation_space.shape[0], hidden_dims[0]))
        self.predict_net.append(activation())
        for i in range(len(hidden_dims) - 1):
            self.predict_net.append(th.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.predict_net.append(activation())
        self.predict_net.append(th.nn.Linear(hidden_dims[-1], embed_dim))
        self.predict_net = th.nn.Sequential(*self.predict_net).to(th.device(device))

        self.optimizer = th.optim.Adam(self.predict_net.parameters(), lr=learning_rate)

    def error(self, state, action):
        """Computes the error between the prediction and target network."""
        if not isinstance(state, th.Tensor):
            state = th.as_tensor(state, device=self.device)
            action = th.as_tensor(action, device=self.device)
        if len(state.shape) == 1:
            # need to add batch dimension to flat input
            state = state.unsqueeze(dim=0)
        if len(state.shape) == 3:
            # need to add batch dimension to image input
            state = state.unsqueeze(dim=0)

        if len(action.shape) == 2:
            # it has an unnecessary dimension that we want to squeeze
            action = action.squeeze(dim=-1)
        if len(action.shape) == 0:
            # need to add a dimension
            action = action.unsqueeze(dim=0)

        if self.normalize_images:
            state = state / 255.
        onehot_action =  th.nn.functional.one_hot(action.long(), num_classes=self.n_actions).float()

        x_predict = th.flatten(state, start_dim=1)
        x_target = th.flatten(state, start_dim=1)
        x_predict = th.concat([x_predict,  onehot_action], dim=-1)
        x_target = th.concat([x_target,  onehot_action], dim=-1)

        return self.criterion(self.predict_net(x_predict), self.target_net(x_target))

    def observe(self, state, action, **kwargs):
        """Observes state(s) and 'remembers' them using Random Network Distillation"""
        self.optimizer.zero_grad()
        loss = self.error(state, action).mean().backward()
        self.optimizer.step()

    def __call__(self, state, action, **kwargs):
        """Returns the estimated uncertainty for observing a (minibatch of) state(s) as Tensor."""
        return self.error(state, action).mean(dim=-1)