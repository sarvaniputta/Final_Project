"""
Simulation of Bus Transit System to analyze and compare bunching reduction techniques.

IS590PR Final Project

Contributors:
Sarvani Putta
Pranay Parmar
Dhruman Shah

Overview:
Bus transit systems often face the problem of bunching. This is often due to the positive feedback effect of delays that
increase over time. This module runs a baseline simulation that is set-up with no bunching after which delays are
introduced to obtain bunching. This provides the right platform to test and compare popular bunching reduction
techniques.

Bunching Reduction Techniques Simulated:
1. Stopping for minimum time at key stops.
2. Overtaking delayed bus.

Simulation results also compare the statistical performance of the above two bunching reduction techniques.

Usage:


"""

import pandas as pd
import numpy as np
import queue


class Passenger:

    def __init__(self, source=None, dest=None):
        self.source = source
        self.dest = dest


class Bus:

    def __init__(self, capacity, start_stop=None):
        self.capacity = capacity
        self.start_stop = start_stop
        self.passengers_onboard = []

    def space_available(self) -> bool:
        return len(self.passengers_onboard) < self.capacity

    def board_passengers(self, bus_stop) -> list:
        if self.space_available():
            self.passengers_onboard.append(bus_stop.passenger_queue.get())

    def alight(self, bus_stop):
        for passenger in self.passengers_onboard:
            if passenger.dest == bus_stop.stop_no:
                self.passengers_onboard.remove(passenger)


class BusStop:

    def __init__(self, stop_no, passenger_arrival_rate=2.0):
        self.passenger_queue = queue.Queue()
        self.stop_no = stop_no
        self.passenger_arrival_rate = passenger_arrival_rate

    def add_passenger_to_bus_queue(self, source, dest):
        self.passenger_queue.put(Passenger(source=source, dest=dest))


class BusLine:

    def __init__(self):
        self.stops = []

    def add_stop(self, bus_stop):
        self.stops.append(bus_stop)


class Environment:

    def __init__(self,
                 num_of_busses=4,
                 start_bus_locations = [0, 5, 15, 20],
                 bus_capacity=20,
                 passender_arrival_rate=2.0,
                 ):
        pass


def run_simulation(num_sims=1000):
    #Code
    print('Setup environment, dist. for variables of uncertainty')


if __name__ == '__main__':
    print('Call')
