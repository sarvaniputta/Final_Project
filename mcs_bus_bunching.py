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

import numpy as np
import queue


class Bus:

    def __init__(self, bus_id, capacity, total_stops, start_stop=None):
        self.id_ = bus_id
        self.capacity = capacity
        self.start_stop = start_stop
        self.passengers_onboard = {x: 0 for x in range(0, total_stops)}
        # this is dictionary that holds all the passengers in the bus
        # the keys are the stop_id and the values are the number of passengers with that
        # particular stop as their destination, initially the number of passengers is 0.
        # Sum of values is the total passengers inside the bus which is 0 initially.
        self.prev_stop = None

    def space_available(self) -> bool:
        return sum(self.passengers_onboard.values()) < self.capacity

    def board_passengers(self, bus_stop):
        if self.space_available():
            self.passengers_onboard.append(bus_stop.passenger_queue.get())

    def alight(self, bus_stop):
        for passenger in self.passengers_onboard:
            if passenger.dest == bus_stop.stop_no:
                self.passengers_onboard.remove(passenger)

    @staticmethod
    def dwell(passengers_board, passengers_alight):
        """Total dwell time at each stop. Assuming it to be 0.1min for onboarding and alighting"""
        return ((passengers_board + passengers_alight) * 0.1) + np.random.uniform(0, 0.3)

    def travel(self, from_id, to_id, start=None):
        """bus travel times between stops. Assuming it to take 20 min on average with some approxiamtion
        """
        return np.random.triangular(18, 20, 25)


class BusStop:

    def __init__(self, stop_id, passenger_arrival_rate=0.5):
        self.id_ = stop_id
        self.passenger_arrival_rate = passenger_arrival_rate
        self.passengers_waiting = 0
        self.arrival_times = []
        self.departure_times = []

    def new_passengers_at_stop(self, start, end):
        """ Number of passengers arriving to that bus stop with in a time interval"""
        return np.random.poisson(self.passenger_arrival_rate * (end - start))

    def passenger_destinations(self, num_passengers_boarding, bus, total_stops):
        """ Number of passengers getting down in the further bus stops"""
        keys = [stop_id for stop_id in range(total_stops) if stop_id != self.id_]
        dist = np.random.multinomial(num_passengers_boarding, np.ones(total_stops - 1) / (total_stops - 1))
        dict_ = {k: v for k, v in zip(keys, dist)}
        # Adding new passengers to bus dict containing stop_id and existing passengers getting down at that stop
        for k, v in bus.passengers_onboard:
            if k != self.id_: # No passenger get down at the same stop he alights
                bus.passengers_onboard[k] = v + dict_[k]


class BusLine:

    def __init__(self, stops, intervals):
        self.stops = stops
        self.intervals = intervals


class Simulation:

    def __init__(self, num_buses=4, num_stops=10):
        # start_time of clock in minutes. Can be thought to correspond to 6 AM
        self.num_buses = num_buses
        self.num_stops = num_stops
        self.stops_list = [BusStop(stop_id=i, passenger_arrival_rate=0.5) for i in range(0, num_stops)]
        self.bus_list = [Bus(bus_id=i) for i in range(0, num_buses)]
        # initialize the buses to arrive at stop 0 at 0, 15, ..., so on intervals from garage
        for i in range(num_buses):
            bus = self.bus_list[i]
            self.arrive_at_stop(bus, self.stops_list[0], time=i * 15)

    def arrive_at_stop(self, bus: Bus, bus_stop: BusStop, time: float):
        # bus has arrived at stop at time
        # record arrival time
        bus_stop.arrival_times.append(time)
        # now bus stops for appropriate dwell time
        if len(bus_stop.arrival_times) < 2:
            start = time - 15
        else:
            start = bus_stop.arrival_times[-2]
        num_boarding = bus_stop.new_passengers_at_stop(start, time)
        num_alighting = bus.passengers_onboard[bus_stop.id_]
        bus.passengers_onboard[bus_stop.id_] = 0  # all passengers in bus to this stop have alighted
        bus_stop.passenger_destinations(num_boarding, bus, self.num_stops)
        for id_ in bus.passengers_onboard:
            bus.passengers_onboard[id_] = bus.passengers_onboard[id_] + num_alighting
        dwell_time = bus.dwell(num_boarding, num_alighting)
        departing_time = time + dwell_time
        bus_stop.departure_times.append(departing_time)
        bus.prev_stop = bus_stop.id_

