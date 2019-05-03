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

import json

import matplotlib.pyplot as plt
import numpy as np


class Bus:
    """ Apart from the Bus ID, the bus class holds information related to stops where passengers will drop,
    time that it will take to reach next stops, bus capacity, previous stop and trip count.

    >>> bus = Bus(1, 6)
    >>> bus.passengers_onboard[1] = 10
    >>> bus.space_available()
    90
    """
    def __init__(self, bus_id: int, total_stops: int, capacity=100, start_stop=None):
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

    def space_available(self) -> int:
        """ For checking the maximum number of passegers that can alight in a bus
        :return: Space left in the bus """
        return max(0, self.capacity - sum(self.passengers_onboard.values()))


class BusStop:
    """ Class to model a bus stop. Hold stop ID, passenger arrival rate, destination probability values,
    number of passengers waiting for a bus.
    Stores time when buses arrive at that stop, and depart, in a list. Keeps a track of queue length before the next
    bus picks up passengers.
    The arrival rate list serves as metric to check whether bunching has occurred at that stop.

    >>> stop = BusStop(1, dest_prob=[0.00, 0.08, 0.19, 0.56, 0.13, 0.04])
    >>> stop.passengers_at_stop(10, 20) >= 0
    True
    >>> dict_ = stop.passenger_destinations(10) #O is expected as the destination probability is 0 in the list above
    >>> dict_[0] == 0
    True
    """

    def __init__(self, stop_id: int, passenger_arrival_rate=0.1, dest_prob=None):
        self.dest_prob = dest_prob
        self.id_ = stop_id
        self.passenger_arrival_rate = passenger_arrival_rate
        self.passengers_waiting = 0
        self.arrival_times = []
        self.departure_times = []
        self.queue_history = []

    def passengers_at_stop(self, start: float, end: float) -> int:
        """ Returns Existing plus number of new passengers arriving to that bus stop with in a time interval"""
        self.passengers_waiting += np.random.poisson(self.passenger_arrival_rate * (end - start))
        self.queue_history.append(self.passengers_waiting)
        return self.passengers_waiting

    def passenger_destinations(self, num_passengers_boarding: int) -> dict:
        """ Number of passengers getting down in the further bus stops"""
        keys = [stop_id for stop_id in range(len(self.dest_prob)) if stop_id != self.id_]
        dist = np.random.multinomial(num_passengers_boarding, self.dest_prob)
        # all destinations equally likely
        dict_ = {k: v for k, v in zip(keys, dist)}
        return dict_


class BusLane:
    """Connector between stops that can simulate overtaking if required.
    traversal_time() method calculates the overall time it would take to for a bus to reach next stop
    depending on whether overtaking is enabled or not.
    travel_time_dist can be provided to simulate travel times for bus in this lane.
    Buses running on this lane are tracked using buses_on_route list.

    >>> lane = BusLane(1, from_stop=1, to_stop=2)
    >>> 14 <= lane.traversal_time() <= 18
    True
    >>> lane2 = BusLane(1, from_stop=1, to_stop=2)
    >>> bus1, bus2 = Bus(2, 6), Bus(3, 6)
    >>> bus1.time_for_next_stop, bus2.time_for_next_stop = 18, 16
    >>> lane2.buses_on_route += [bus1, bus2]
    >>> lane2.traversal_time(0) > 18
    True
    """

    def __init__(self, id_, from_stop=0, to_stop=1, overtaking_allowed=False, travel_time_dist=None):
        self.buses_on_route = [] # contains buses which are traversing this lane right now
        self.from_stop = from_stop
        self.to_stop = to_stop
        self.overtaking_allowed = overtaking_allowed
        self.travel_time_dist = travel_time_dist
        self.id = id_

    def traversal_time(self, start_time=None) -> float:
        """ Calculates the simulated time it would take for a bus to travel in a lane
        If overtaking is disabled, the travel time is 0.5 minutes more for the faster bus
        than the travel time for slowest bus in the same route if one exists. The faster bus would bunch with the
        slower one."""
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

    def __init__(self, nbuses=4, headway=15, overtaking_allowed=False, travel_time_dist=[lambda: 16] * 4,
                 passenger_arrival_rate=[0.3, 0.3, 0.3, 0.3],
                 dest_prob_matrix=[[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], maintain_headway=False):
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

        self.headway = min(headway, 60 / nbuses)
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
        if stop.arrival_times:  # there was atleast one bus pick up here (only false at the start of the simulation)
            previous_arrival = stop.arrival_times[-1]  # latest arrival
        else:
            previous_arrival = arrival_time - 10
            # assume only 10 mins worth of pax at stop for the very first trip at this stop of the day

        # Fit passengers based on space available, leave those that cannot fit
        bus_space = bus.space_available()
        n_board = stop.passengers_at_stop(start=previous_arrival, end=arrival_time)
        if bus_space < n_board:  # No space in bus for all passengers
            stop.passengers_waiting -= bus_space  # reduce passengers at stop by num who get in bus
            total_boarded = bus_space
        else:
            stop.passengers_waiting = 0  # All passengers get in bus
            total_boarded = n_board

        destination_dict = stop.passenger_destinations(n_board)
        # Adding new passengers to bus dict containing stop_id and existing passengers getting down at that stop
        for k in destination_dict:
            bus.passengers_onboard[k] = bus.passengers_onboard[k] + destination_dict[k]

        # all passengers with this stop as destination have alighted so update count
        bus.passengers_onboard[stop.id_] = 0

        # time taken to alight and onboard these passengers
        # dwell_time = self.dwell(total_boarded, n_alight)
        actual_time = self.dwell(total_boarded, n_alight)
        # If maintain_headway is enabled, whenever the inter-arrival time is less than headway gap,
        # the following bus will wait at the stop to restore the headway gap.
        if self.maintain_headway:
            if not stop.arrival_times:
                dwell_time = actual_time
            else:
                time_diff = arrival_time - stop.arrival_times[-1]
                dwell_time = min(0.1 * self.headway, max(self.headway - time_diff, actual_time))

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
        potential_travel_time = bus_lane.traversal_time(start_time)  # takes care of overtaking as configured
        bus.time_for_next_stop = start_time + potential_travel_time
        bus_lane.buses_on_route.append(bus)

    def simulate(self, max_trips=10):

        """ Run simulations until 1 bus in the system completes max trips.
        >>> model = TransitSystem(nbuses=1)
        >>> model.simulate(max_trips=4) # with one trip no bunches should occur
        >>> all([len(s.arrival_times)<=4  for s in model.stops])
        True



        """
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
        """ Obtain statistics on passengers waiting and number of bunching situations occurred after the simulations are run.
        >>> model = TransitSystem()
        >>> model.simulate(max_trips=1) # with one trip no bunches should occur
        >>> stats = model.get_stats()
        >>> stats['avg_bunches']  == 0  #No bunches because of constant travel time.
        True
        >>> stats['avg_passenger_waiting']>0
        True
        """

        bunches = []
        num_passengers = []
        for stop in self.stops:
            arrival_times = np.array(stop.arrival_times)
            bus_time_diff = np.diff(arrival_times)
            num_bunch_events = np.count_nonzero(bus_time_diff <= 2.0)
            bunches.append(num_bunch_events)
            num_passengers += stop.queue_history
        return {'avg_bunches': np.mean(bunches), 'avg_passenger_waiting': np.mean(num_passengers)}


def draw_histogram(num_bunch_list, title, color='#c91829'):
    """
        Draw a plot showing distribution of mean bunches across the simulation runs.
        :param num_bunch_list: List of mean bunches observed across simulation runs.
        :param title: Title to be used displayed on the plot
        :param color: Color of the line.
    """
    plt.rcParams["figure.dpi"] = 200
    plt.style.use('fivethirtyeight')
    plt.hist(num_bunch_list, alpha=0.8, color=color, edgecolor='black')
    plt.xlabel('Number of bunches')
    plt.ylabel('Frequency')
    plt.title(title)
    # plt.xticks(np.arange(0, 20, step=2.0))
    plt.show()


def draw_lineplot(num_bunch_list, num_pass_waiting, title, color='#c91829'):
    """
    Draw a plot showing number of bunches across the simulation runs.
    :param num_bunch_list: List of mean bunches observed across simulation runs.
    :param num_pass_waiting: List of passengers waiting across simulation runs
    :param title: Title to be used displayed on the plot
    :param color: Color of the line.

    """
    plt.rcParams["figure.dpi"] = 200
    plt.style.use('fivethirtyeight')
    plt.plot(num_bunch_list, alpha=0.8, color=color, linewidth=1.3)
    plt.axhline(np.mean(num_bunch_list), color='#13294a', linewidth=2.0, alpha=0.85, label='Mean bunches')
    plt.xlabel('Simulation number')
    plt.ylabel('Number of bunches')
    # plt.yticks(np.arange(0, 28, step=2.0))
    plt.title(title)
    plt.text(0.05, 0.95, 'Avg. passengers queue length = ' + str(np.mean(num_pass_waiting).round(2)),
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


def test_hypothesis(num_of_sims: int = 1000, overtaking_allowed: bool = False, maintain_headway: bool = False) -> list:
    """
    Runs simulations with the given strategies to test hypothesis
    :param num_of_sims: Number of simulation runs to compute stats
    :param overtaking_allowed: Boolean that toggles if buses can overtake each other on the route.
    :param maintain_headway: Boolean that toggles if buses can wait at stops to maintain headway.
    :return: list of the total number of bunches observed across all the simulation runs.
    """

    current_sim = 0
    mean_passengers_waiting = []
    mean_bunches = []

    pass_arrival_rate = [0.88, 0.09, 0.20, 0.63, 0.35, 0.08]
    dest_prob_matrix = [[0.00, 0.08, 0.19, 0.56, 0.13, 0.04],
                        [0.35, 0.00, 0.13, 0.40, 0.09, 0.03],
                        [0.38, 0.06, 0.00, 0.43, 0.10, 0.03],
                        [0.53, 0.08, 0.20, 0.00, 0.14, 0.04],
                        [0.36, 0.06, 0.14, 0.41, 0.00, 0.03],
                        [0.34, 0.05, 0.13, 0.39, 0.09, 0.00]]
    with open('travel_time_icdfs.json') as f:
        tt_icdfs = json.load(f)

    tt_dists = [travel_time_callable(icdf) for icdf in tt_icdfs.values()]

    while current_sim < num_of_sims:
        model = TransitSystem(nbuses=6, headway=8, overtaking_allowed=overtaking_allowed,
                              maintain_headway=maintain_headway,
                              travel_time_dist=tt_dists,
                              passenger_arrival_rate=pass_arrival_rate,
                              dest_prob_matrix=dest_prob_matrix)
        model.simulate()
        mean_bunches.append(model.get_stats()['avg_bunches'])
        mean_passengers_waiting.append(model.get_stats()['avg_passenger_waiting'])
        del model
        current_sim += 1
    return mean_bunches, mean_passengers_waiting


if __name__ == '__main__':
    # No reduction technique applied
    mean_bunches_baseline, mean_pass_list = test_hypothesis()
    draw_lineplot(mean_bunches_baseline, mean_pass_list, 'When no reduction technique used')

    # Simulation with only overtaking allowed
    mean_bunches_hyp1, mean_pass_list1 = test_hypothesis(overtaking_allowed=True)
    draw_lineplot(mean_bunches_hyp1, mean_pass_list1, 'When overtaking allowed', color='#000099')

    # Simulation with only maintain_headway allowed
    mean_bunches_hyp2, mean_pass_list2 = test_hypothesis(maintain_headway=True)
    draw_lineplot(mean_bunches_hyp2, mean_pass_list2, 'When headway gap maintained', color='#dbae58')

    # Simulation with both techniques applied
    mean_bunches_hyp3, mean_pass_list3 = test_hypothesis(overtaking_allowed=True, maintain_headway=True)
    draw_lineplot(mean_bunches_hyp3, mean_pass_list3, 'Both combined', color='#238e7b')
