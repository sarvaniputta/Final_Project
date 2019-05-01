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
import matplotlib.pyplot as plt


class Bus:

    def __init__(self, bus_id, total_stops, capacity=50, start_stop=None):
        self.id_ = bus_id
        self.capacity = capacity
        self.start_stop = start_stop
        self.passengers_onboard = {x: 0 for x in range(0, total_stops)}
        # this is dictionary that holds all the passengers in the bus
        # the keys are the stop_id and the values are the number of passengers with that
        # particular stop as their destination, initially the number of passengers is 0.
        # Sum of values is the total passengers inside the bus which is 0 initially.
        self.prev_stop = None
        self.time_for_next_stop = None # clock time at which bus hits next stop
        self.trip_count = 0 # counter for the current trip

    def space_available(self):
        return self.capacity - sum(self.passengers_onboard.values())

    def board_passengers(self, bus_stop):
        if self.space_available():
            self.passengers_onboard.append(bus_stop.passenger_queue.get())

    def alight(self, bus_stop):
        for passenger in self.passengers_onboard:
            if passenger.dest == bus_stop.stop_no:
                self.passengers_onboard.remove(passenger)


    # def travel(self, from_id, to_id, start=None):
    #     """bus travel times between stops. Assuming it to take 20 min on average with some approxiamtion
    #     """
    #     return np.random.triangular(18, 20, 25)


class BusStop:

    def __init__(self, stop_id, passenger_arrival_rate=0.1):
        self.id_ = stop_id
        self.passenger_arrival_rate = passenger_arrival_rate
        self.passengers_waiting = 0
        self.arrival_times = []
        self.departure_times = []

    def passengers_at_stop(self, start, end):
        """ Existing plus number of new passengers arriving to that bus stop with in a time interval"""
        self.passengers_waiting += np.random.poisson(self.passenger_arrival_rate * (end - start))
        return self.passengers_waiting

    def passenger_destinations(self, num_passengers_boarding, total_stops):
        """ Number of passengers getting down in the further bus stops"""
        keys = [stop_id for stop_id in range(total_stops) if stop_id != self.id_]
        dist = np.random.multinomial(num_passengers_boarding, np.ones(total_stops - 1) / (total_stops - 1))
        # all destinations equally likely
        dict_ = {k: v for k, v in zip(keys, dist)}
        return dict_


class BusLane:
    """Connector between stops that can simulate overtaking if necessary"""

    def __init__(self, id_, from_stop=0, to_stop=1, overtaking_allowed=False):
        self.buses_on_route = [] # contains buses which are traversing this lane right now
        self.from_stop = from_stop
        self.to_stop = to_stop
        self.overtaking_allowed = overtaking_allowed
        self.id = id_

    def traversal_time(self, start_time=None):
        potential_travel_time = np.random.triangular(14, 15, 18)
        if self.overtaking_allowed:
            return potential_travel_time
        elif self.buses_on_route:
            slowest_bus = max(self.buses_on_route, key=lambda x: x.time_for_next_stop)
            if start_time + potential_travel_time <= slowest_bus.time_for_next_stop:
                potential_travel_time = slowest_bus.time_for_next_stop - start_time + 0.5
            return potential_travel_time
        else:
            return potential_travel_time


class TransitSystem:

    def __init__(self, nstops=4, nbuses=4, headway=15, overtaking_allowed=False):
        self.stops = [BusStop(i, passenger_arrival_rate=0.3) for i in range(nstops)]
        self.bus_lanes = [BusLane(a, from_stop=a, to_stop=(a + 1) % nstops, overtaking_allowed=overtaking_allowed)
                          for a in range(0, nstops)]
        self.buses = [Bus(i, nstops) for i in range(nbuses)]
        self.nstops = nstops
        self.headway = min(headway, 60/nbuses)


    @staticmethod
    def dwell(passengers_board, passengers_alight):
        """Total dwell time at each stop. Assuming it to be 0.1min for onboarding and alighting"""
        return ((passengers_board + passengers_alight) * 0.05) + np.random.uniform(0, 0.1)

    def arrive(self, bus, stop: BusStop, arrival_time):
        # pop the bus from the corresponding bus_lane
        for bus_lane in self.bus_lanes:
            if bus_lane.to_stop == stop.id_:
                break
        # bus can arrive at given stop only on this bus_lane
        if bus_lane.buses_on_route:
            bus_lane.buses_on_route.remove(bus)

        if stop.id_ == 0:
            bus.trip_count += 1

        n_alight = bus.passengers_onboard[stop.id_]
        # how many passengers are waiting
            # before we get there we need how much time elapsed since previous bus picked up passengers
        if stop.arrival_times: # there was atleast one bus pick up here (only false at the start of the simulation)
            previous_arrival = stop.arrival_times[-1] # latest arrival
        else:
            previous_arrival = arrival_time - 10
            # assume only 10 mins worth of pax at stop for the very first trip at this stop of the day

        # Fit passengers based on space available, leave those that cannot fit
        bus_space = bus.space_available()
        n_board = stop.passengers_at_stop(start=previous_arrival, end=arrival_time)
        if bus_space < n_board: # No space in bus for all passengers
            stop.passengers_waiting -= bus_space # reduce passengers at stop by num who get in bus
            total_boarded = bus_space
        else:
            stop.passengers_waiting = 0 # All passengers get in bus
            total_boarded = n_board

        destination_dict = stop.passenger_destinations(n_board, total_stops=self.nstops)
        # Adding new passengers to bus dict containing stop_id and existing passengers getting down at that stop
        for k in destination_dict:
            bus.passengers_onboard[k] = bus.passengers_onboard[k] + destination_dict[k]

        # all passengers with this stop as destination have alighted so update count
        bus.passengers_onboard[stop.id_] = 0

        # time taken to alight and onboard these passengers
        dwell_time = self.dwell(total_boarded, n_alight)

        # update arrival time list at stop
        stop.arrival_times.append(arrival_time)
        stop.departure_times.append((bus.id_, arrival_time + dwell_time))
        bus.prev_stop = stop.id_
        self.travel_to_next_stop(bus, from_stop=stop, start_time=arrival_time + dwell_time)

    def travel_to_next_stop(self, bus, from_stop, start_time):
        bus_lane = self.bus_lanes[from_stop.id_]
        potential_travel_time = bus_lane.traversal_time(start_time) # takes care of overtaking as configured
        bus.time_for_next_stop = start_time + potential_travel_time
        bus_lane.buses_on_route.append(bus)

    def simulate(self, max_trips=10):
        # start simulation by making buses go to stop 0 in headway increments
        for bus in self.buses:
            self.arrive(bus, self.stops[0], arrival_time=bus.id_ * self.headway)

        while all(bus.trip_count < max_trips for bus in self.buses):
            # break when the first bus completes max_trips

            # this is the first bus that needs our attention because it reaches the stop
            earliest_event_bus = min(self.buses, key=lambda bus: bus.time_for_next_stop)
            to_stop = self.stops[(earliest_event_bus.prev_stop + 1) % self.nstops]
            self.arrive(earliest_event_bus, to_stop, earliest_event_bus.time_for_next_stop)

    def print_stats(self):
        bunches = []
        for stop in self.stops:
            arrival_times = np.array(stop.arrival_times)
            bus_time_diff = np.diff(arrival_times)
            num_bunch_events = np.count_nonzero(bus_time_diff <= 2.0)
            bunches.append(num_bunch_events)
        return np.mean(bunches)


def draw_histogram(num_bunch_list, title, color='#c91829'):
    plt.style.use('fivethirtyeight')
    plt.hist(num_bunch_list, alpha=0.8, color=color, edgecolor='black')
    plt.xlabel('Number of bunches')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(np.arange(0, 20, step=2.0))
    plt.rcParams["figure.dpi"] = 600
    plt.show()


def draw_lineplot(num_bunch_list, title, color='#c91829'):
    plt.style.use('fivethirtyeight')
    plt.plot(num_bunch_list, alpha=0.8, color=color, linewidth=1.3)
    plt.axhline(np.mean(num_bunch_list), color='#13294a', linewidth=2.0, alpha=0.85, label='Mean bunches')
    plt.xlabel('Simulation number')
    plt.ylabel('Number of bunches')
    plt.yticks(np.arange(0, 20, step=2.0))
    plt.title(title)
    plt.rcParams["figure.dpi"] = 600
    plt.show()


if __name__ == '__main__':
    mean_bunches = []
    num_of_sims = 1000
    current_sim = 0
    
    while current_sim < num_of_sims:
        model = TransitSystem(nbuses=4, headway=10, overtaking_allowed=False)
        model.simulate()
        mean_bunches.append(model.print_stats())
        del model
        current_sim += 1
    draw_histogram(mean_bunches, 'When no reduction technique used(n=1000)')
    draw_lineplot(mean_bunches, 'When no reduction technique used')

    mean_bunches.clear()
    current_sim = 0

    while current_sim < num_of_sims:
        model = TransitSystem(nbuses=4, headway=10, overtaking_allowed=True)
        model.simulate()
        mean_bunches.append(model.print_stats())
        del model
        current_sim += 1

    draw_histogram(mean_bunches, 'When overtaking allowed (n=1000)', color='#238e7b')
    draw_lineplot(mean_bunches, 'When overtaking allowed', color='#238e7b')
