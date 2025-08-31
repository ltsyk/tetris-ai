from collections import deque
import random
from typing import List, Optional

import numpy as np
import torch
from torch import nn


def _get_activation(name: str):
    return {
        'relu': nn.ReLU(),
        'linear': nn.Identity(),
        'tanh': nn.Tanh(),
    }[name]


class DQNAgent:
    """Deep Q-Network agent implemented with PyTorch."""

    def __init__(
        self,
        state_size: int,
        mem_size: int = 10000,
        discount: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.0,
        epsilon_stop_episode: int = 0,
        n_neurons: Optional[List[int]] = None,
        activations: Optional[List[str]] = None,
        loss: str = 'mse',
        optimizer: str = 'adam',
        replay_start_size: Optional[int] = None,
        modelFile: Optional[str] = None,
    ) -> None:
        state_dict = None
        if modelFile is not None:
            state_dict = torch.load(modelFile, map_location='cpu')

        if n_neurons is None:
            if state_dict is not None:
                weight_keys = sorted(
                    [k for k in state_dict if k.endswith('weight')],
                    key=lambda x: int(x.split('.')[0]),
                )
                n_neurons = [state_dict[k].shape[0] for k in weight_keys[:-1]]
            else:
                n_neurons = [32, 32]
        if activations is None:
            activations = ['relu'] * len(n_neurons) + ['linear']
        if len(activations) != len(n_neurons) + 1:
            raise ValueError(
                'n_neurons and activations do not match, '
                f'expected a n_neurons list of length {len(activations) - 1}',
            )
        if replay_start_size is not None and replay_start_size > mem_size:
            raise ValueError('replay_start_size must be <= mem_size')
        if mem_size <= 0:
            raise ValueError('mem_size must be > 0')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_size = state_size
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        if epsilon_stop_episode > 0:
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_stop_episode
        else:
            self.epsilon = 0
            self.epsilon_min = 0
            self.epsilon_decay = 0
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss_name = loss
        self.optimizer_name = optimizer
        if not replay_start_size:
            replay_start_size = mem_size // 2
        self.replay_start_size = replay_start_size

        self.model = self._build_model()
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.train_steps = 0
        self.target_update_frequency = 10

    def _build_model(self) -> nn.Module:
        layers: List[nn.Module] = []
        in_features = self.state_size
        for n, act in zip(self.n_neurons, self.activations[:-1]):
            layers.append(nn.Linear(in_features, n))
            layers.append(_get_activation(act))
            in_features = n
        layers.append(nn.Linear(in_features, 1))
        layers.append(_get_activation(self.activations[-1]))
        return nn.Sequential(*layers)

    # Memory management --------------------------------------------------
    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))

    # Predictions ---------------------------------------------------------
    def random_value(self):
        return random.random()

    def predict_value(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(state_t).item()

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        return self.predict_value(state)

    def best_state(self, states):
        if random.random() <= self.epsilon:
            return random.choice(list(states))
        states_array = np.array(list(states))
        with torch.no_grad():
            tensor = torch.tensor(states_array, dtype=torch.float32, device=self.device)
            values = self.model(tensor).squeeze().cpu().numpy()
        best_index = int(np.argmax(values))
        return tuple(states_array[best_index])

    # Training ------------------------------------------------------------
    def train(self, batch_size=32, epochs=3, num_workers=0):
        if batch_size > self.mem_size:
            print('WARNING: batch size is bigger than mem_size. The agent will not be trained.')
        n = len(self.memory)
        if n >= self.replay_start_size and n >= batch_size:
            batch = random.sample(self.memory, batch_size)
            states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
            next_states = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=self.device)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
            dones = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)

            with torch.no_grad():
                next_qs = self.target_model(next_states).squeeze()
            targets = rewards + (1 - dones) * self.discount * next_qs

            for _ in range(epochs):
                pred = self.model(states).squeeze()
                loss = self.criterion(pred, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.train_steps += 1
            if self.train_steps % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

    # Persistence ---------------------------------------------------------
    def save_model(self, name):
        torch.save(self.model.state_dict(), name)

