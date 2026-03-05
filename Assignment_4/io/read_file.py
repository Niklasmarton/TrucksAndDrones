import numpy as np


def _next_value_line(lines, start_idx):
    i = start_idx
    while i < len(lines):
        s = lines[i].strip()
        if s and not s.startswith("#"):
            return i, s
        i += 1
    return None, None


def _parse_matrix(lines, start_idx, n_rows):
    matrix = []
    i = start_idx
    while i < len(lines) and len(matrix) < n_rows:
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        row = [float(x) for x in s.split("\t") if x != ""]
        matrix.append(row)
        i += 1

    if len(matrix) != n_rows:
        raise ValueError(f"Expected {n_rows} matrix rows, got {len(matrix)}")

    return matrix, i


def _parse_instance_file(instance_path: str):
    with open(instance_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    n_customers = None
    flight_limit = None
    truck_times = None
    drone_times = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("# Number of customers"):
            j, value = _next_value_line(lines, i + 1)
            if value is None:
                raise ValueError("Missing number of customers value")
            n_customers = int(float(value))
            i = j + 1
            continue

        if (
            line.startswith("# Drone flight limit")
            or line.startswith("# Drone range limit")
            or line.startswith("# Flight range limit")
        ):
            j, value = _next_value_line(lines, i + 1)
            if value is None:
                raise ValueError("Missing drone flight limit value")
            flight_limit = float(value)
            i = j + 1
            continue

        if line.startswith("# Travel time matrix for the truck"):
            if n_customers is None:
                raise ValueError("Number of customers must be parsed before truck matrix")
            truck_times, i = _parse_matrix(lines, i + 1, n_customers + 1)
            continue

        if line.startswith("# Travel time matrix for the drones"):
            if n_customers is None:
                raise ValueError("Number of customers must be parsed before drone matrix")
            drone_times, i = _parse_matrix(lines, i + 1, n_customers + 1)
            continue

        i += 1

    if n_customers is None:
        raise ValueError("Could not parse number of customers")
    if flight_limit is None:
        raise ValueError("Could not parse flight limit")
    if truck_times is None:
        raise ValueError("Could not parse truck matrix")
    if drone_times is None:
        raise ValueError("Could not parse drone matrix")

    return {
        "n_customers": n_customers,
        "flight_limit": flight_limit,
        "truck_times": truck_times,
        "drone_times": drone_times,
    }


def instance(instance_path: str):
    parsed = _parse_instance_file(instance_path)
    T = np.array(parsed["truck_times"], dtype=float)
    D = np.array(parsed["drone_times"], dtype=float)
    current = 0
    ending = 0
    truck_route = [0]
    search_queue = [i for i in range(1, parsed["n_customers"] + 1)]
    return T, D, current, ending, truck_route, search_queue


def read_instance(instance_path: str):
    """
    Compatibility wrapper used by op.py and local_search.py.
    Returns a dictionary with expected keys.
    """
    parsed = _parse_instance_file(instance_path)
    n_customers = parsed["n_customers"]

    return {
        "truck_times": parsed["truck_times"],
        "drone_times": parsed["drone_times"],
        "flight_limit": parsed["flight_limit"],
        "n_customers": n_customers,
        "depot_index": 0,
        "current": 0,
        "ending": 0,
        "truck_route": [0],
        "search_queue": [i for i in range(1, n_customers + 1)],
    }
