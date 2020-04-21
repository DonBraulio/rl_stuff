# %%
import math
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.spaces import Discrete, Box


import ray  # Using ray==0.8.3, ray==0.8.4 is not working with dqn
from ray.tune.logger import pretty_print
import ray.rllib.agents.dqn as dqn


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


price_change_grid = np.arange(0.5, 2.0, 0.1)
profit_map = np.zeros((len(price_grid), len(price_change_grid)))
q_map = np.zeros((len(price_grid), len(price_change_grid)))
for i in range(len(price_grid)):
    for j in range(len(price_change_grid)):
        profit_map[i, j] = profit_t_response(price_grid[i], price_grid[i] * price_change_grid[j])
        q_map[i, j] = q_t(price_grid[i], price_grid[i] * price_change_grid[j], q_0, k, a_q, b_q)


plt.figure(figsize=(16, 5))
for i in range(len(price_change_grid)):
    if math.isclose(price_change_grid[i], 1.0):
        color = "red"
    else:
        color = (0.6, 0.3, price_change_grid[i] / 2.0)
    plt.plot(price_grid, profit_map[:, i], c=color)
    # plt.plot(price_grid, q_map[:, i], c=color)
plt.xlabel("Price ($)")
plt.ylabel("Profit")
plt.legend(
    np.int_(np.round((1 - price_change_grid) * 100)),
    loc="lower right",
    title="Price change (%)",
    fancybox=False,
    framealpha=0.6,
)


# %%
class HiLoPricingEnv(gym.Env):
    def __init__(self, config):
        self.reset()
        self.action_space = Discrete(len(price_grid))
        self.observation_space = Box(0, 10000, shape=(2 * T,), dtype=np.float32)

    def reset(self):
        self.state = env_intial_state()
        self.t = 0
        return self.state

    # Returns next state, reward, and end-of-the-episode flag
    def step(self, action):
        next_state, reward = env_step(self.t, self.state, action)
        self.t += 1
        self.state = next_state
        return next_state, reward, self.t == T - 1, {}


# %%
config = dqn.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
config["train_batch_size"] = 256
config["buffer_size"] = 10000
config["hiddens"] = [128, 128, 128]
trainer = dqn.DQNTrainer(config=config, env=HiLoPricingEnv)
for i in range(50):
    result = trainer.train()
    print(pretty_print(result))


policy = trainer.get_policy()

# Reinforcement para exploración?

# %%
# tracing the policy
state = env_intial_state()
state[0] = 17  # Set initial price = price_grid[17] = 180
results = np.zeros(shape=(2, T))
for t in range(T):
    action = policy.compute_single_action(state, state=[])
    best_action = action[0]
    state, reward = env_step(t, state, best_action)
    results[:, t] = np.array([best_action, reward])
    print(state)


# %%
plt.figure()
plt.plot(results[0, :])

plt.figure()
plt.plot(results[1, :])

# %%


# %%
