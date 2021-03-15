import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random as rnd
import copy
import math
plt.style.use('seaborn-darkgrid')
matplotlib.rcParams.update({'font.size': 18})


#  WITH STAY
ws_plot_t = [1, 4, 7, 10, 13, 16, 19, 22, 25,
             28, 31, 34, 37, 40, 43, 46, 49, 52, 55]

ws_plot_reward_prob = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0165, 0.0557, 0.1205, 0.2164,
                       0.3316, 0.4316, 0.5624, 0.6849, 0.7533, 0.8051, 0.8408, 0.8666, 0.883, 0.8988]
ws_plot_transition_prob = [0.0, 0.0, 0.0, 0.0, 0.0, 0.2488, 0.4626, 0.6204, 0.724,
                           0.7892, 0.8338, 0.8637, 0.8826, 0.8959, 0.9053, 0.9004, 0.9098, 0.9103, 0.9132]

ws_plot_reward_eaten = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0089, 0.0258, 0.0373, 0.0473, 0.0527, 0.0632, 0.061, 0.0649]
ws_plot_transition_eaten = [0.0, 0.0, 0.0, 0.0302, 0.0412, 0.0576, 0.0698, 0.0716,
                            0.0734, 0.079, 0.08, 0.0839, 0.0836, 0.0843, 0.0796, 0.0923, 0.0844, 0.0871, 0.0851]

ns_plot_t = [1, 4, 7, 10, 13, 16, 19, 22, 25,
             28, 31, 34, 37, 40, 43, 46, 49, 52, 55]

ns_plot_reward_prob = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0022, 0.0147, 0.1105, 0.2331,
                       0.3552, 0.4851, 0.5999, 0.7029, 0.7665, 0.8162, 0.8607, 0.891, 0.9033, 0.9153]
ns_plot_transition_prob = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1184, 0.356, 0.5293, 0.6408,
                           0.7375, 0.8045, 0.8391, 0.878, 0.9015, 0.9109, 0.9194, 0.9274, 0.9286, 0.9362]

ns_plot_reward_eaten = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0047, 0.0134, 0.0208, 0.027, 0.0296, 0.0325, 0.0362, 0.0402]
ns_plot_transition_eaten = [0.0, 0.0, 0.0, 0.0091, 0.0217, 0.0299, 0.036, 0.0398,
                            0.0465, 0.0513, 0.052, 0.0578, 0.0539, 0.0537, 0.0582, 0.0581, 0.058, 0.0615, 0.0575]


plt.subplot(1, 2, 1)
plt.xlabel("Time horizon, T")
plt.ylabel("Probability of exiting")
plt.plot(ws_plot_t, ws_plot_reward_prob,
         label='Model 1 (with stay)')
plt.plot(ws_plot_t, ws_plot_transition_prob,
         label='Model 2 (with stay)')


plt.plot(ws_plot_t, ns_plot_reward_prob,
         label='Model 1 (no stay)')
plt.plot(ws_plot_t, ns_plot_transition_prob,
         label='Model 2 (no stay)')
plt.legend(loc="upper left", prop={'size': 10})


plt.subplot(1, 2, 2)
plt.xlabel("Time horizon, T")
plt.ylabel("Probability of getting eaten")
plt.plot(ws_plot_t, ws_plot_reward_eaten,
         label='Model 1 (with stay)')
plt.plot(ws_plot_t, ws_plot_transition_eaten,
         label='Model 2 (with stay)')

plt.plot(ws_plot_t, ns_plot_reward_eaten,
         label='Model 1 (no stay)')
plt.plot(ws_plot_t, ns_plot_transition_eaten,
         label='Model 2 (no stay)')

plt.legend(loc="upper left", prop={'size': 10})
plt.show()

# No STAY
