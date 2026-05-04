T = None
D = None
flight_limit = None
depot = 0


def set_operator_context(truck_times, drone_times, flight_range, depot_index=0):
    global T, D, flight_limit, depot
    T = truck_times
    D = drone_times
    flight_limit = flight_range
    depot = depot_index


def assert_context_is_set():
    if T is None or D is None or flight_limit is None:
        raise ValueError(
            "Operator context is not set. Call set_operator_context(T, D, flight_limit, depot) "
            "before using operator functions."
        )


def get_operator_context():
    assert_context_is_set()
    return T, D, flight_limit, depot
