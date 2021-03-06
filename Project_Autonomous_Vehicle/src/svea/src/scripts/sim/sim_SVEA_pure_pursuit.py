#!/usr/bin/env python

import sys
import os
import rospy
import math
import numpy as np
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
svea = os.path.join(dirname, '../../')
sys.path.append(os.path.abspath(svea))

from models.bicycle import SimpleBicycleState
from simulators.sim_SVEA import SimSVEA
from simulators.viz_utils import plot_car
from controllers.control_interfaces import ControlInterface

dirname = os.path.dirname(__file__)
pure_pursuit = os.path.join(dirname,
        '../../PythonRobotics/PathTracking/pure_pursuit/')
sys.path.append(os.path.abspath(pure_pursuit))

import pure_pursuit


## SIMULATION PARAMS ##########################################################
vehicle_name = "SVEA0"
init_state = [0, 0, 0, 0] # [x, y, yaw, v], units: [m, m, rad, m/s]
target_speed = 0.6 # [m/s]
dt = 0.01

# trajectory
cx = np.arange(0, 5, 0.1)
cy = [math.sin(ix) * ix for ix in cx]

# animate results?
show_animation = True
###############################################################################


## PURE PURSUIT PARAMS ########################################################
pure_pursuit.k = 0.4  # look forward gain
pure_pursuit.Lfc = 0.4  # look-ahead distance
pure_pursuit.L = 0.324  # [m] wheel base of vehicle
###############################################################################


def main():

    rospy.init_node('SVEA_sim_purepursuit')

    # initialize simulated model and control interface
    simple_bicycle_model = SimpleBicycleState(*init_state, dt=dt)
    ctrl_interface = ControlInterface(vehicle_name).start()
    rospy.sleep(0.5)

    # start background simulation thread
    simulator = SimSVEA(vehicle_name, simple_bicycle_model, dt)
    simulator.start()
    rospy.sleep(0.5)

    # log data
    x = []
    y = []
    yaw = []
    v = []
    t = []

    # pure pursuit variables
    lastIndex = len(cx) - 1
    target_ind = pure_pursuit.calc_target_index(simple_bicycle_model, cx, cy)

    # simualtion + animation loop
    time = 0.0
    while lastIndex > target_ind and not rospy.is_shutdown():

        # compute control input via pure pursuit
        steering, target_ind = \
            pure_pursuit.pure_pursuit_control(simple_bicycle_model, cx, cy, target_ind)
        ctrl_interface.send_control(steering, target_speed)

        # log data
        x.append(simple_bicycle_model.x)
        y.append(simple_bicycle_model.y)
        yaw.append(simple_bicycle_model.yaw)
        v.append(simple_bicycle_model.v)
        t.append(time)

        # update for animation
        if show_animation:
            to_plot = (simple_bicycle_model,
                    x, y,
                    steering, target_ind)
            animate_pure_pursuit(*to_plot)
        else:
            rospy.loginfo_throttle(1.5, simple_bicycle_model)

        time += dt

    if show_animation:
        plt.close()
        to_plot = (simple_bicycle_model,
                x, y,
                steering, target_ind)
        animate_pure_pursuit(*to_plot)
        plt.show()
    else:
        # just show resulting plot if not animating
        to_plot = (simple_bicycle_model, x, y)
        plot_trajectory(*to_plot)
        plt.show()


def plot_trajectory(car_model, x, y):
    plt.plot(cx, cy, ".r", label="course")
    plt.plot(x, y, "-b", label="trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.title("Heading[deg]: "
                + str(math.degrees(car_model.yaw))[:4]
                +" | Speed[m/s]:"
                + str(car_model.v)[:4])

def animate_pure_pursuit(car_model, x, y, steering, target_ind):
    """ Single animation update for pure pursuit."""
    plt.cla() #clear last plot/frame
    plot_trajectory(car_model, x, y)
    # plot pure pursuit current target
    plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
    # plot car current state
    plot_car(car_model.x,
             car_model.y,
             car_model.yaw,
             steering)
    plt.pause(0.001)

if __name__ == '__main__':
    main()
