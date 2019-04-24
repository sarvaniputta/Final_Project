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

    def __init__(self, bus_id, capacity=50, total_stops=4, start_stop=None):
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

    def space_available(self) -> bool:
        return sum(self.passengers_onboard.values()) < self.capacity

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

    def new_passengers_at_stop(self, start, end):
        """ Number of passengers arriving to that bus stop with in a time interval"""
        return np.random.poisson(self.passenger_arrival_rate * (end - start))

    def passenger_destinations(self, num_passengers_boarding, total_stops):
        """ Number of passengers getting down in the further bus stops"""
        keys = [stop_id for stop_id in range(total_stops) if stop_id != self.id_]
        dist = np.random.multinomial(num_passengers_boarding, np.ones(total_stops - 1) / (total_stops - 1))
        # all destinations equally likely
        dict_ = {k: v for k, v in zip(keys, dist)}
        return dict_

class TravelSegment:
    """Connector between stops that can simulate overtaking if necessary"""

    def __init__(self, id_, from_stop=0, to_stop=1, overtaking_allowed=False):
        self.buses_on_route = [] # contains buses which are traversing this segment right now
        self.from_stop = from_stop
        self.to_stop = to_stop
        self.overtaking_allowed = overtaking_allowed
        self.id = id_

    def traversal_time(self, start_time=None):
        potential_travel_time =  np.random.triangular(14, 15, 18)
        if self.overtaking_allowed:
            return potential_travel_time
        elif self.buses_on_route:
            slowest_bus = max(self.buses_on_route, key=lambda x: x.time_for_next_stop)
            if start_time + potential_travel_time <= slowest_bus.time_for_next_stop:
                potential_travel_time = slowest_bus.time_for_next_stop - start_time + 0.5
            return potential_travel_time
        else:
            return potential_travel_time


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


class TransitSystem:

    def __init__(self, nstops=4, nbuses=4, headway=15, overtaking_allowed=False ):
        self.stops = [BusStop(i) for i in range(nstops)]
        self.segments = [TravelSegment(0, from_stop=0, to_stop=1, overtaking_allowed=overtaking_allowed),
                         TravelSegment(1, from_stop=1, to_stop=2, overtaking_allowed=overtaking_allowed),
                         TravelSegment(2, from_stop=2, to_stop=3, overtaking_allowed=overtaking_allowed),
                         TravelSegment(3, from_stop=3, to_stop=0, overtaking_allowed=overtaking_allowed),]
        self.buses = [Bus(i) for i in range(nbuses)]
        self.nstops = nstops
        self.headway = headway




    @staticmethod
    def dwell(passengers_board, passengers_alight):
        """Total dwell time at each stop. Assuming it to be 0.1min for onboarding and alighting"""
        # return ((passengers_board + passengers_alight) * 0.05) + np.random.uniform(0, 0.1)
        return 0.25

    def arrive(self, bus, stop:BusStop, arrival_time):
        # pop the bus from the corresponding segment
        for segment in self.segments:
            if segment.to_stop == stop.id_:
                break
        # bus can arrive at given stop only on this segment
        if segment.buses_on_route:
            segment.buses_on_route.remove(bus)

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
        n_board = stop.new_passengers_at_stop(start=previous_arrival, end=arrival_time)
        destination_dict  = stop.passenger_destinations(n_board, total_stops=self.nstops)
        # Adding new passengers to bus dict containing stop_id and existing passengers getting down at that stop
        for k in destination_dict:
            bus.passengers_onboard[k] = bus.passengers_onboard[k] + destination_dict[k]

        # all passengers with this stop as destination have alighted so update count
        bus.passengers_onboard[stop.id_] = 0

        # time taken to alignt and onboard these passengers
        dwell_time = self.dwell(n_board, n_alight)

        # update arrival time list at stop
        stop.arrival_times.append(arrival_time)
        stop.departure_times.append((bus.id_, arrival_time + dwell_time))
        bus.prev_stop = stop.id_
        self.travel_to_next_stop(bus, from_stop=stop, start_time=arrival_time + dwell_time)


    def travel_to_next_stop(self, bus, from_stop, start_time):
        segment = self.segments[from_stop.id_]
        potential_travel_time = segment.traversal_time(start_time) # takes care of overtaking as configured
        bus.time_for_next_stop = start_time + potential_travel_time
        segment.buses_on_route.append(bus)


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



if __name__ == '__main__':
    model = TransitSystem(overtaking_allowed=True)
    model.simulate()
    print(model.stops[0].arrival_times)
    print(model.stops[0].departure_times)








