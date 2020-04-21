# %%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim

from IPython.display import clear_output
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
## Environment parameters
price_max = 500
price_step = 10
price_grid = np.arange(price_step, price_max, price_step)

T = 20
q_0 = 5000
k = 20
unit_cost = 100
a_q = 300
b_q = 100


# Environment simulator
def plus(x):
    return 0 if x < 0 else x


def minus(x):
    return 0 if x > 0 else -x


def shock(x):
    return np.sqrt(x)


# Demand at time step t for current price p_t and previous price p_t_1
def q_t(p_t, p_t_1, q_0, k, a, b):
    return plus(q_0 - k * p_t - a * shock(plus(p_t - p_t_1)) + b * shock(minus(p_t - p_t_1)))


# Profit at time step t
def profit_t(p_t, p_t_1, q_0, k, a, b, unit_cost):
    return q_t(p_t, p_t_1, q_0, k, a, b) * (p_t - unit_cost)


def profit_t_response(p_t, p_t_1):
    return profit_t(p_t, p_t_1, q_0, k, a_q, b_q, unit_cost)


def env_intial_state():
    return np.repeat(0, 2 * T)


def env_step(t, state, action):
    next_state = np.repeat(0, len(state))
    next_state[0] = price_grid[action]
    next_state[1:T] = state[0 : T - 1]
    next_state[T + t] = 1
    reward = profit_t_response(next_state[0], next_state[1])
    return next_state, reward


# %%
# Code based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN Implementation by ElPO
class PolicyNetworkDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetworkDQN, self).__init__()
        layers = [
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        q_values = self.model(x)
        return q_values


class AnnealedEpsGreedyPolicy(object):
    def __init__(self, eps_start=0.9, eps_end=0.05, eps_decay=400):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def select_action(self, q_values):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold:
            return np.argmax(q_values)
        else:
            return random.randrange(len(q_values))


GAMMA = 1.00
TARGET_UPDATE = 20
BATCH_SIZE = 512


# This is again from same pytorch tutorial above
def optimize_model(memory, policy_net, target_net):
    if BATCH_SIZE > len(memory):
        return  # Ignore initialization issues
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool
    )
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.stack(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = reward_batch[:, 0] + (GAMMA * next_state_values)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def to_tensor(x):
    return torch.from_numpy(np.array(x).astype(np.float32))


def to_tensor_long(x):
    return torch.tensor([[x]], device=device, dtype=torch.long)


# %%

policy_net = PolicyNetworkDQN(2 * T, len(price_grid)).to(device)
target_net = PolicyNetworkDQN(2 * T, len(price_grid)).to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr=0.005)
policy = AnnealedEpsGreedyPolicy()
memory = ReplayMemory(10000)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

num_episodes = 1000
return_trace = []
p_trace = []  # price schedules used in each episode
for i_episode in range(num_episodes):
    state = env_intial_state()
    reward_trace = []
    p = []
    for t in range(T):
        # Select and perform an action
        with torch.no_grad():
            q_values = policy_net(to_tensor(state))
        action = policy.select_action(q_values.detach().numpy())

        next_state, reward = env_step(t, state, action)

        # Store the transition in memory
        memory.push(
            to_tensor(state),
            to_tensor_long(action),
            to_tensor(next_state) if t != T - 1 else None,
            to_tensor([reward]),
        )

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model(memory, policy_net, target_net)

        reward_trace.append(reward)
        p.append(price_grid[action])

    return_trace.append(sum(reward_trace))
    p_trace.append(p)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

        clear_output(wait=True)
        print(f"Episode {i_episode} of {num_episodes} ({i_episode/num_episodes*100:.2f}%)")


# %%
## Visualization functions
def plot_return_trace(returns, smoothing_window=10, range_std=2):
    plt.figure(figsize=(16, 5))
    plt.xlabel("Episode")
    plt.ylabel("Return ($)")
    returns_df = pd.Series(returns)
    ma = returns_df.rolling(window=smoothing_window).mean()
    mstd = returns_df.rolling(window=smoothing_window).std()
    plt.plot(ma, c="blue", alpha=1.00, linewidth=1)
    plt.fill_between(
        mstd.index, ma - range_std * mstd, ma + range_std * mstd, color="blue", alpha=0.2
    )


def plot_price_schedules(p_trace, sampling_ratio, last_highlights, fig_number=None):
    plt.figure(fig_number)
    plt.xlabel("Time step")
    plt.ylabel("Price ($)")
    plt.plot(range(T), np.array(p_trace[0:-1:sampling_ratio]).T, c="k", alpha=0.05)
    return plt.plot(
        range(T), np.array(p_trace[-(last_highlights + 1) : -1]).T, c="red", alpha=0.5, linewidth=2
    )


# %%
plot_return_trace(return_trace)

fig = plt.figure(figsize=(16, 5))
plot_price_schedules(p_trace, 5, 1, fig.number)
# %%
# %%
# tracing the policy
state = env_intial_state()
results = np.zeros(shape=(2, T))
for t in range(T):
    action = policy.compute_single_action(state, state=[])
    best_action = action[0]
    state, reward = env_step(t, state, best_action)
    results[:, t] = np.array([best_action, reward])


# %%
plt.figure()
plt.plot(results[0, :])

plt.figure()
plt.plot(results[1, :])

# %%
