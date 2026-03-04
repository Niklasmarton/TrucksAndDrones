import numpy as np
import random
from pathlib import Path


INSTANCE_FILE = "Truck_Drone_Contest.txt"

# Context (set by set_construction_context(...) or lazy-loaded fallback)
T = None
D = None
NUM_NODES = None
DRONE_FLIGHT_LIMIT = None

# Route state used during construction
truck_route = []
search_queue = []


def _resolve_default_instance_file() -> str:
    p = Path(INSTANCE_FILE)
    if p.exists():
        return str(p)

    alt = Path(__file__).resolve().parents[1] / "Test_files" / INSTANCE_FILE
    if alt.exists():
        return str(alt)

    return INSTANCE_FILE


def load_context_from_file(instance_file=None):
    """
    Load construction context from a file (legacy behavior), but only on demand.
    """
    global T, D, NUM_NODES, DRONE_FLIGHT_LIMIT

    file_path = instance_file or _resolve_default_instance_file()

    with open(file_path, "r") as f:
        lines = f.readlines()

    n_customers = int(lines[1].strip())
    DRONE_FLIGHT_LIMIT = float(lines[3].strip())
    NUM_NODES = n_customers + 1

    T = np.loadtxt(file_path, delimiter="\t", skiprows=5, max_rows=NUM_NODES)
    D = np.loadtxt(file_path, delimiter="\t", skiprows=5 + NUM_NODES + 1, max_rows=NUM_NODES)


def set_construction_context(truck_times, drone_times, flight_limit):
    """
    Inject context explicitly so callers can use any dataset without relying on
    Construction.py hardcoded file loading.
    """
    global T, D, NUM_NODES, DRONE_FLIGHT_LIMIT
    T = np.array(truck_times)
    D = np.array(drone_times)
    NUM_NODES = len(T)
    DRONE_FLIGHT_LIMIT = float(flight_limit)


def _ensure_context():
    if T is None or D is None or NUM_NODES is None or DRONE_FLIGHT_LIMIT is None:
        load_context_from_file()


def _reset_route_state():
    global truck_route, search_queue
    truck_route = [0]
    search_queue = [i for i in range(1, NUM_NODES)]


def next_node(current_node):
    relevant_row = T[current_node]
    candidates = []

    for index in search_queue:
        if index == current_node:
            continue
        distance = relevant_row[index]
        candidates.append((index, distance))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1])
    # Keep legacy nearest-neighbor behavior (target_size = 1).
    return candidates[0][0]


def truck_only_route() -> list[int]:
    current_node = 0
    while search_queue:
        chosen = next_node(current_node)
        if chosen is None:
            break
        truck_route.append(chosen)
        search_queue.remove(chosen)
        current_node = chosen
    truck_route.append(0)
    return truck_route


def build_truck_route() -> list[int]:
    """
    Public helper to build a truck-only route using current construction context.
    """
    _ensure_context()
    _reset_route_state()
    return truck_only_route()


def map_launching_and_landing(prev_node, next_node, drone_number, drone_launch, drone_land):
    drone_launch[drone_number].append(prev_node)
    drone_land[drone_number].append(next_node)
    return drone_launch, drone_land


def drone_route() -> list[int]:
    """
    Build construction solution in original return format:
    truck, drone1_customers, drone2_customers, drone_launch_nodes, drone_land_nodes
    """
    _ensure_context()
    _reset_route_state()

    truck = truck_only_route()
    drone1 = []
    drone2 = []
    prob_first_node = 0.9
    prob_second_node = 0.7

    i = 1
    drone_busy = {"drone1": False, "drone2": False}
    drone_launch = {1: [], 2: []}
    drone_land = {1: [], 2: []}
    protected_nodes = set()

    while i < len(truck) - 1:
        prev_node = truck[i - 1]
        curr_node = truck[i]
        next_node = truck[i + 1]
        flight_out = D[prev_node][curr_node]
        flight_back = D[curr_node][next_node]

        if curr_node in protected_nodes:
            protected_nodes.remove(curr_node)
            drone_busy["drone1"] = False
            i += 1
            continue

        total_flight = flight_out + flight_back
        if (not drone_busy["drone1"]) and total_flight <= DRONE_FLIGHT_LIMIT:
            if random.random() < prob_first_node:
                drone_busy["drone1"] = True
                drone1.append(curr_node)
                drone_launch, drone_land = map_launching_and_landing(
                    prev_node, next_node, 1, drone_launch, drone_land
                )
                protected_nodes.add(next_node)
                truck.pop(i)
                continue

        if i + 2 < len(truck) and (not drone_busy["drone2"]) and random.random() < prob_second_node:
            if drone_busy["drone1"] and protected_nodes:
                protected_nodes.pop()
            next_node = truck[i + 2]
            curr_node = truck[i + 1]
            drone_busy["drone2"] = True
            drone2.append(curr_node)
            drone_launch, drone_land = map_launching_and_landing(
                prev_node, next_node, 2, drone_launch, drone_land
            )
            protected_nodes.add(next_node)
            truck.pop(i + 1)
            continue

        i += 1

    return truck, drone1, drone2, drone_launch, drone_land


if __name__ == "__main__":
    truck, drone1, drone2, drone_launch, drone_land = drone_route()
    launch_indices_1 = [truck.index(node) + 1 for node in drone_launch[1]]
    landing_indices_1 = [truck.index(node) + 1 for node in drone_land[1]]
    launch_indices_2 = [truck.index(node) + 1 for node in drone_launch[2]]
    landing_indices_2 = [truck.index(node) + 1 for node in drone_land[2]]

    print(f"Truck route is: {truck}")
    print(f"Drone1 visits: {drone1}")
    print(f"Drone2 visits: {drone2}")
    print(f"launching sites: {launch_indices_1, -1, landing_indices_2}")
    print(f"landing sites: {landing_indices_1, -1, landing_indices_2}")
    print(f"Length of truck route is: {len(truck)}")
    print(f"{truck}|{drone1}-1{drone2}|{launch_indices_1}-1{launch_indices_2}|{landing_indices_1}-1{landing_indices_2}")
