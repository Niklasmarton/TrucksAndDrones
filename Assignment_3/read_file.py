import numpy as np

def read_instance(instance_path: str):
    INSTANCE_FILE = instance_path

    # Load distance matrices (truck times T, drone times D). Row/col 0 is the depot.
    T = np.loadtxt(INSTANCE_FILE, delimiter="\t", skiprows=5, max_rows=101)
    NUM_NODES = T.shape[1]  # columns correspond to nodes (hub + 100)
    D = np.loadtxt(INSTANCE_FILE, delimiter="\t", skiprows=107, max_rows=101)

    current = 0
    ending = 0
    # Order of visited cities
    truck_route = T[0]
    # Remaining nodes (only real node columns, exclude hub)
    search_queue = [i for i in range(1, NUM_NODES)]

    return T, D, current, ending, truck_route, search_queue

