# RandomSolutionGenerator.py
#
# Creates a random valid (but not necessarily feasible) truck–drone solution.
#
# "Valid" here means:
#   - Every customer 1..N appears exactly once either in the truck route (part1)
#     or in the drone deliveries (part2).
#   - Depot (0) only appears in the truck route.
#   - part2, part3 and part4 are structurally consistent with:
#       - SolutionFeasibility.are_parts_consistent
#       - CalCulateTotalArrivalTime.calculate_total_waiting_time
#
# The function assumes the instance dictionary has at least:
#   - "n_customers": int          (number of customers, N)
#   - "depot_index": int          (usually 0)
#
# It does NOT enforce flight-range or timing constraints, so many generated
# solutions will be infeasible in the problem sense, which is fine for blind
# random search.


from typing import Dict, Any, List
import random


def generate_random_solution(instance: Dict[str, Any], n_drones: int = 3) -> Dict[str, Any]:
    """
    Generate a random valid (representation-wise) solution.

    instance["n_customers"]: number of customers N
    instance["depot_index"]:  depot node index (typically 0)

    Returns a solution dictionary with keys:
      - "part1": truck route (list of nodes, starts and ends at depot)
      - "part2": drone customers + -1 separators between drones
      - "part3": launch cells (1-based indices in part1, aligned with part2)
      - "part4": reconvene cells (1-based indices in part1, aligned with part2)

    Representation guarantees (so it works with your Feasibility & Arrival code):
      - Every customer 1..N appears exactly once across part1 and part2.
      - 0 never appears in part2.
      - For each non-separator customer in part2, there is a corresponding
        launch/reconvene cell in part3/part4 with:
            1 <= launch < reconvene <= len(part1)
      - At positions where part2 contains -1, part3 and part4 also contain -1.
    """
    n_customers: int = instance["n_customers"]
    depot: int = instance.get("depot_index", 0)

    # ------------------------------------------------------------------
    # 1) Randomly decide which customers are served by truck vs drones
    # ------------------------------------------------------------------
    customers = list(range(1, n_customers + 1))
    random.shuffle(customers)

    # Choose a random split point so that:
    #   - at least one customer is served by the truck
    #   - others (possibly none) are served by drones
    split_index = random.randint(1, n_customers)  # inclusive
    truck_customers = customers[:split_index]
    drone_customers = customers[split_index:]

    # Build truck route: depot -> truck_customers -> depot
    part1: List[int] = [depot] + truck_customers + [depot]

    # ------------------------------------------------------------------
    # 2) Distribute drone customers across drones, build part2
    # ------------------------------------------------------------------
    # routes[d] will hold the customers served by drone d in order
    drone_routes: List[List[int]] = [[] for _ in range(n_drones)]
    for c in drone_customers:
        d = random.randint(0, n_drones - 1)
        drone_routes[d].append(c)

    # Build part2 using -1 as separator between drones
    # Example:
    #   routes = [[9,4], [], [2,10,7]], n_drones=3
    #   => part2 = [9,4,-1,2,10,7]
    part2: List[int] = []
    non_empty_indices = [i for i, r in enumerate(drone_routes) if r]

    for idx, d in enumerate(non_empty_indices):
        part2.extend(drone_routes[d])
        # Add separator -1 if there are more non-empty drone routes after this one
        if idx < len(non_empty_indices) - 1:
            part2.append(-1)

    # If there are no drone customers, part2 is simply []
    # and we will keep part3/part4 empty as well.
    if not part2:
        return {
            "part1": part1,
            "part2": [],
            "part3": [],
            "part4": [],
        }

    # ------------------------------------------------------------------
    # 3) Build part3 and part4 (launch & reconvene cells)
    # ------------------------------------------------------------------
    # We must align them with part2, and:
    #   - when part2[i] == -1  -> part3[i] = part4[i] = -1
    #   - when part2[i] != -1  -> part3[i], part4[i] are valid cell indices
    #
    # Cell indices are 1-based positions in part1, and
    # SolutionFeasibility.is_valid_cell requires:
    #   1 <= cell <= len(part1)
    #
    # We also need launch_cell < reconvene_cell.
    #
    part3: List[int] = []
    part4: List[int] = []

    # Helper to sample a valid pair (launch, reconvene)
    def random_launch_reconvene_pair(max_cell: int):
        """
        Sample a random pair of cells (launch, reconvene) such that:
            1 <= launch < reconvene <= max_cell
        """
        # To ensure launch < reconvene, we can sample reconvene >= 2,
        # then sample launch in [1, reconvene-1].
        reconvene = random.randint(2, max_cell)
        launch = random.randint(1, reconvene - 1)
        return launch, reconvene

    max_cell = len(part1)  # 1-based indexing over the truck route cells

    for cust in part2:
        if cust == -1:
            # Separator: all three parts must align on -1 here
            part3.append(-1)
            part4.append(-1)
        else:
            launch_cell, reconvene_cell = random_launch_reconvene_pair(max_cell)
            part3.append(launch_cell)
            part4.append(reconvene_cell)

    solution: Dict[str, Any] = {
        "part1": part1,
        "part2": part2,
        "part3": part3,
        "part4": part4,
    }
    return solution


# Optional: quick manual test (you can delete or comment this out in the final version)
if __name__ == "__main__":
    # Example minimal instance:
    inst_example = {
        "n_customers": 10,
        "depot_index": 0,
    }
    sol = generate_random_solution(inst_example, n_drones=3)
    print("part1 (truck):", sol["part1"])
    print("part2 (drones):", sol["part2"])
    print("part3 (launch cells):", sol["part3"])
    print("part4 (reconvene cells):", sol["part4"])
