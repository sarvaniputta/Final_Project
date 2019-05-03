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
import json


class Bus:

    def __init__(self, bus_id, total_stops, capacity=100, start_stop=None):
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
        return max(0, self.capacity - sum(self.passengers_onboard.values()))

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

    def __init__(self, stop_id, passenger_arrival_rate=0.1, dest_prob=None):
        self.dest_prob = dest_prob
        self.id_ = stop_id
        self.passenger_arrival_rate = passenger_arrival_rate
        self.passengers_waiting = 0
        self.arrival_times = []
        self.departure_times = []

    def passengers_at_stop(self, start, end):
        """ Existing plus number of new passengers arriving to that bus stop with in a time interval"""
        self.passengers_waiting += np.random.poisson(self.passenger_arrival_rate * (end - start))
        return self.passengers_waiting

    def passenger_destinations(self, num_passengers_boarding):
        """ Number of passengers getting down in the further bus stops"""
        keys = [stop_id for stop_id in range(len(self.dest_prob)) if stop_id != self.id_]
        dist = np.random.multinomial(num_passengers_boarding, self.dest_prob)
        # all destinations equally likely
        dict_ = {k: v for k, v in zip(keys, dist)}
        return dict_


class BusLane:
    """Connector between stops that can simulate overtaking if necessary"""

    def __init__(self, id_, from_stop=0, to_stop=1, overtaking_allowed=False, travel_time_dist=None):
        self.buses_on_route = [] # contains buses which are traversing this lane right now
        self.from_stop = from_stop
        self.to_stop = to_stop
        self.overtaking_allowed = overtaking_allowed
        self.travel_time_dist = travel_time_dist
        self.id = id_

    def traversal_time(self, start_time=None):
        if self.travel_time_dist:
            potential_travel_time = self.travel_time_dist()
        else:
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

    def __init__(self, nbuses=4, headway=15, overtaking_allowed=False, travel_time_dist=[], passenger_arrival_rate=[0.3, 0.3, 0.3,0.3],
                 dest_prob_matrix = [], maintain_headway=False):
        self.nstops = len(passenger_arrival_rate)
        self.travel_time_dist = travel_time_dist
        self.passenger_arrival_rate = passenger_arrival_rate
        self.dest_prob_matrix = dest_prob_matrix
        self.stops = [BusStop(i, passenger_arrival_rate=rate, dest_prob=dest_prob_matrix[i])
                      for i, rate in enumerate(passenger_arrival_rate)]

        self.bus_lanes = [BusLane(a, from_stop=a, to_stop=(a + 1) % self.nstops,
                                  overtaking_allowed=overtaking_allowed, travel_time_dist=tt)
                          for a, tt in enumerate(self.travel_time_dist)]
        self.buses = [Bus(i, self.nstops) for i in range(nbuses)]

        self.headway = min(headway, 60/nbuses)
        self.maintain_headway = maintain_headway

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

        destination_dict = stop.passenger_destinations(n_board)
        # Adding new passengers to bus dict containing stop_id and existing passengers getting down at that stop
        for k in destination_dict:
            bus.passengers_onboard[k] = bus.passengers_onboard[k] + destination_dict[k]

        # all passengers with this stop as destination have alighted so update count
        bus.passengers_onboard[stop.id_] = 0

        # time taken to alight and onboard these passengers
        #dwell_time = self.dwell(total_boarded, n_alight)
        actual_time = self.dwell(total_boarded, n_alight)
        # If maintain_headway is enabled, whenever the inter-arrival time is less than headway gap,
        # the following bus will wait at the stop to restore the headway gap.
        if self.maintain_headway:
            if not stop.arrival_times:
                dwell_time = actual_time
            else:
                time_diff = arrival_time - stop.arrival_times[-1]
                dwell_time = min(0.1*self.headway, max(self.headway-time_diff, actual_time))



                # dwell_time = self.headway if time_diff < self.headway else actual_time
        else:
            dwell_time = actual_time

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

    def get_stats(self):
        bunches = []
        num_passengers = []
        for stop in self.stops:
            arrival_times = np.array(stop.arrival_times)
            bus_time_diff = np.diff(arrival_times)
            num_bunch_events = np.count_nonzero(bus_time_diff <= 2.0)
            bunches.append(num_bunch_events)
            num_passengers.append(stop.passengers_waiting)
        return {'avg_bunches': np.mean(bunches), 'avg_passenger_waiting': np.mean(num_passengers)}


def draw_histogram(num_bunch_list, title, color='#c91829'):
    plt.rcParams["figure.dpi"] = 200
    plt.style.use('fivethirtyeight')
    plt.hist(num_bunch_list, alpha=0.8, color=color, edgecolor='black')
    plt.xlabel('Number of bunches')
    plt.ylabel('Frequency')
    plt.title(title)
    # plt.xticks(np.arange(0, 20, step=2.0))
    plt.show()


def draw_lineplot(num_bunch_list, title, num_pass_waiting, color='#c91829'):
    plt.rcParams["figure.dpi"] = 200
    plt.style.use('fivethirtyeight')
    plt.plot(num_bunch_list, alpha=0.8, color=color, linewidth=1.3)
    plt.axhline(np.mean(num_bunch_list), color='#13294a', linewidth=2.0, alpha=0.85, label='Mean bunches')
    plt.xlabel('Simulation number')
    plt.ylabel('Number of bunches')
    #plt.yticks(np.arange(0, 28, step=2.0))
    plt.title(title)
    plt.text(0.05, 0.95, 'Avg. # of passengers waiting = ' + str(np.mean(num_pass_waiting).round(2)),
             fontsize=14, transform=plt.gcf().transFigure)
    plt.show()


def simulate_empirical(icdf: np.ndarray):
    """ Generate a sample given an inverse cumulative distribution function using the inverse transform method.
    https://blogs.sas.com/content/iml/2013/07/22/the-inverse-cdf-method.html
    params:
    icdf: values of the inverse cdf at equally spaced probability values [0, 1]

    >>> icdf = np.linspace(0, 4, 2048) # icdf for uniform(0, 4)
    >>> rvs = [simulate_empirical(icdf) for i in range(10000)]
    >>> min(rvs) >= 0
    True
    >>> max(rvs) <= 4
    True
    >>> np.abs(np.mean(rvs) - 2) < 1e-1
    True
    >>> np.abs(np.median(rvs)- 2) < 1e-1
    True
    """
    grid_size = len(icdf)
    return icdf[int(np.random.uniform() * grid_size)]


def travel_time_callable(icdf):
    """A callable to simulate a travel time with given inverse cdf whenever called"
    Uses closure to keep icdf in memory

    >>> icdf = np.linspace(0, 4, 2048) # icdf for uniform(0, 4)
    >>> fn = travel_time_callable(icdf)
    >>> rvs = [fn() for i in range(10000)]
    >>> min(rvs) >= 0
    True
    >>> max(rvs) <= 4
    True
    >>> np.abs(np.mean(rvs) - 2) < 1e-1
    True
    >>> np.abs(np.median(rvs)- 2) < 1e-1
    True
    """
    return lambda: simulate_empirical(icdf)

def simulate_baseline(num_of_sims):
    current_sim = 0
    mean_passengers_waiting = []
    mean_bunches =[]
    while current_sim < num_of_sims:
        model = TransitSystem(nbuses=6, headway=10, overtaking_allowed=False, maintain_headway=False,
                              travel_time_dist=tt_dists,
                              passenger_arrival_rate=pass_arrival_rate,
                              dest_prob_matrix=dest_prob_matrix)
        model.simulate()
        mean_bunches.append(model.get_stats()['avg_bunches'])
        mean_passengers_waiting.append(model.get_stats()['avg_passenger_waiting'])
        del model
        current_sim += 1
    return mean_bunches


def simulate_hyp_1(num_of_sims):
    current_sim = 0
    mean_bunches = []
    mean_passengers_waiting = []
    while current_sim < num_of_sims:
        model = TransitSystem(nbuses=4, headway=10, overtaking_allowed=True, maintain_headway=False,
                              travel_time_dist=tt_dists,
                              passenger_arrival_rate=pass_arrival_rate,
                              dest_prob_matrix=dest_prob_matrix)
        model.simulate()
        mean_bunches.append(model.get_stats()['avg_bunches'])
        mean_passengers_waiting.append(model.get_stats()['avg_passenger_waiting'])
        del model
        current_sim += 1
    return mean_bunches

def simulate_hyp_2(num_of_sims:int)-> list:
    """Simulate minimum dwell time scenario by maintaining a minimum headway"""
    current_sim = 0
    mean_passengers_waiting = []
    mean_bunches = []
    while current_sim < num_of_sims:
        model = TransitSystem(nbuses=4, headway=10, overtaking_allowed=False, maintain_headway=True,
                              travel_time_dist=tt_dists,
                              passenger_arrival_rate=pass_arrival_rate,
                              dest_prob_matrix=dest_prob_matrix)
        model.simulate()
        mean_bunches.append(model.get_stats()['avg_bunches'])
        mean_passengers_waiting.append(model.get_stats()['avg_passenger_waiting'])
        del model
        current_sim += 1
    return mean_bunches





if __name__ == '__main__':
    mean_bunches = []
    mean_passengers_waiting = []
    num_of_sims = 1000
    current_sim = 0
    pass_arrival_rate = [0.88, 0.09, 0.20, 0.63, 0.35, 0.08]
    dest_prob_matrix = [[0.00, 0.08, 0.19, 0.56,	0.13,	0.04],
                        [0.35,	0.00,	0.13,	0.40,	0.09,	0.03],
                        [0.38,	0.06,	0.00,	0.43,	0.10,	0.03],
                        [0.53,	0.08,	0.20,	0.00,	0.14,	0.04],
                        [0.36,	0.06,	0.14,	0.41,	0.00,	0.03],
                        [0.34,	0.05,	0.13,	0.39,	0.09,	0.00]]
    with open('travel_time_icdfs.json') as f:
        tt_icdfs = json.load(f)

    tt_dists = [travel_time_callable(icdf) for icdf in tt_icdfs.values()]



    mean_bunches_baseline = simulate_baseline(num_of_sims)

    mean_bunches_hyp1 = simulate_hyp_1(num_of_sims)

    mean_bunches_hyp2 = simulate_hyp_2(num_of_sims)



    # No reduction technique applied
    draw_histogram(mean_bunches_baseline, 'When no reduction technique used(n=1000)')
    # draw_lineplot(mean_bunches, 'When no reduction technique used', mean_passengers_waiting)

    # Simulation with only overtaking allowed

    draw_histogram(mean_bunches_hyp1, 'When overtaking allowed (n=1000)', color='#238e7b')
    # draw_lineplot(mean_bunches, 'When overtaking allowed', mean_passengers_waiting, color='#238e7b')

    # Simulation with only maintain_headway allowed

    draw_histogram(mean_bunches_hyp2, 'When overtaking allowed (n=1000)', color='#238e7b')
    # draw_lineplot(mean_bunches, 'When headway gap maintained', mean_passengers_waiting, color='#dbae58')

print