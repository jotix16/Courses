# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import matplotlib.pyplot as plt
import numpy as np
import maze as mz
import maze2 as mz2
import mazep3 as mz3
import pandas as pd

#  [markdown]
# # Problem 1

#
# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])
# with the convention
# 0 = empty cell
# 1 = obstacle
# 2 = exit of the Maze

# mz.draw_maze(maze)

# # Create an environment maze
env = mz.Maze(maze)
env2 = mz2.Maze(maze)
# # env.show()

# # ## Dynamic Programming


# # Finite horizon
# horizon = 30
# # Solve the MDP problem with dynamic programming
# V, policy0 = mz.dynamic_programming(env, horizon)
# V2, policy02 = mz2.dynamic_programming(env2, horizon)


# # Simulate the shortest path starting from position A
method = 'DynProg'
start = (0, 0, 6, 5)
# path = env.simulate(start, policy0, method, horizon)
# path = env2.simulate(start, policy02, method, horizon)


# Show the shortest path
#mz.animate_solution(maze, path)


# # Model minotaur in reward
# results = env.sample(start, policy0, method, horizon, 10000)
# print("Probability of being eaten:", results[0])
# print("Probability exiting:", results[1])
# print("Probability of surviving the T=20 :", results[2])

# print("")
# # Model minotaur in transition probability
# results = env2.sample(start, policy02, method, horizon, 10000)
# print("Probability of being eaten:", results[0])
# print("Probability exiting:", results[1])
# print("Probability of surviving the T=20 :", results[2])


plot_reward_prob = list()
plot_transition_prob = list()

plot_reward_eaten = list()
plot_transition_eaten = list()

plot_t = list()
for t in range(1, 57, 3):
    V, policy0 = mz.dynamic_programming(env, t)
    V2, policy02 = mz2.dynamic_programming(env2, t)
    results = env.sample(start, policy0, method, t, 10000)
    results2 = env2.sample(start, policy02, method, t, 10000)
    plot_reward_prob.append(results[1])
    plot_transition_prob.append(results2[1])

    plot_reward_eaten.append(results[0])
    plot_transition_eaten.append(results2[0])

    plot_t.append(t)
print("Time")
print(plot_t)
print("Minotaurs in reward, prob")
print(plot_reward_prob)
print("Minotaurs in reward, eaten")
print(plot_reward_eaten)

print("Minotaur in transition probs")
print(plot_transition_prob)
print("Minotaur in transition eaten")
print(plot_transition_eaten)

# plt.figure(1)
# plt.plot(plot_t, plot_s, label='Minotaur modeled in reward')
# plt.plot(plot_t, plot_s2, label='Minotaur modeled in transition probability')
# plt.xlabel("Time horizon, T")
# plt.ylabel("Probability of exiting")
# plt.legend()
# plt.show()


# plot_s = list()
# plot_s2 = list()
# plot_t = list()
# for t in range(1, 57, 3):
#     V, policy0 = mz.dynamic_programming(env, t)
#     V2, policy02 = mz2.dynamic_programming(env2, t)
#     results = env.sample(start, policy0, method, t, 10000)
#     results2 = env2.sample(start, policy02, method, t, 10000)
#     plot_s.append(results[0])
#     plot_s2.append(results2[0])
#     plot_t.append(t)
# plt.figure(2)
# plt.plot(plot_t, plot_s2, label='Minotaur modeled in reward')
# plt.plot(plot_t, plot_s, label='Minotaur modeled in transition probability')
# plt.xlabel("Time horizon, T")
# plt.ylabel("Probability of getting eaten")
# plt.legend()
# plt.show()

# %% [markdown]
# ## Value Iteration

# %%
gamma = 0.95
epsilon = 0.0001
V, policy1 = mz.value_iteration(env, gamma, epsilon)


# %%
# Simulate the shortest path starting from position A
method = 'ValIter'
start = (0, 0, 5, 5)
path = env.simulate(start, policy1, method, 20)
# Show the shortest path
#mz.animate_solution(maze, path)


# %%
results = env.sample(start, policy1, method, 20, 10000)
print("Probability of being eaten:", results[0])
print("Probability exiting:", results[1])
print("Probability of surviving the T=20 steps:", results[2])

# %% [markdown]
# ## Policy Iteration

# %%
gamma = 0.95
V, policy2 = mz.policy_iteration(env, gamma)


# %%
# Simulate the shortest path starting from position A
method = 'PolIter'
start = (0, 0, 5, 5)
path = env.simulate(start, policy2, method, horizon)
# Show the shortest path
mz.animate_solution(maze, path)


# %%
results = env.sample(start, policy2, method, 10000, horizon)
print("Probability of being eaten:", results[0])
print("Probability exiting:", results[1])
print("Probability of surviving the T=20 steps:", results[2])

# %% [markdown]
# # Problem 3

# %%
# Description of the maze as a numpy array
maze3 = np.array([
    [0, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])

# with the convention
# 0 = empty cell
# 1 = obstacle
# 2 = exit of the Maze

# mz3.draw_maze(maze3)
env3 = mz3.Maze(maze3)
T = 20

# %% [markdown]
# ## Q-Learning

# %%
gamma3 = 0.8
Q, policy3 = mz3.Q_learning(env3, gamma3)


# %%
# Simulate the shortest path starting from position A
method = 'ValIter'
start = (0, 0, 1, 1)
path = env3.simulate(start, policy3, method, 100)
# Show the shortest path
mz3.animate_solution(maze3, path)

# %% [markdown]
# ## SARSA

# %%
gamma3 = 0.8
epsilon3 = 0.1
Q, policy3 = mz3.SARSA(env3, gamma3, epsilon3)


# %%
# Simulate the shortest path starting from position A
method = 'ValIter'
start = (0, 0, 1, 1)
path = env3.simulate(start, policy3, method, 100)
# Show the shortest path
mz3.animate_solution(maze3, path)
