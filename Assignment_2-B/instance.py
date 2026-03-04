# instance_io.py
from typing import Dict, List
import re

def read_instance(path: str) -> Dict:
    """
    Read a STRPD instance file like R_100.txt.
    Returns a dictionary with truck and drone time matrices etc.
    """
    tokens: List[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.extend(line.split())

    # First two tokens: number of customers, flight limit
    n_customers = int(tokens[0])
    flight_limit = float(tokens[1])

    dim = n_customers + 1  # depot + customers
    n_entries = dim * dim

    # Next n_entries are truck times, then n_entries drone times
    truck_vals = list(map(float, tokens[2:2 + n_entries]))
    drone_vals = list(map(float, tokens[2 + n_entries:2 + 2 * n_entries]))

    def vals_to_matrix(vals):
        return [vals[i * dim:(i + 1) * dim] for i in range(dim)]

    truck_times = vals_to_matrix(truck_vals)
    drone_times = vals_to_matrix(drone_vals)

    instance = {
        "n_customers": n_customers,
        "flight_limit": flight_limit,
        "truck_times": truck_times,
        "drone_times": drone_times,
        "depot_index": 0,
    }
    return instance
