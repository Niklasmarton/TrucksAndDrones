import random
import math
import numpy as np

from Construction import drone_route, T, D
from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime


def nearest_neighbor_indices(truck_route, node):
    """
    Pick launch/land positions on the truck tour nearest to a drone-served node.
    Prefer non-depot nodes to avoid everything collapsing to index 0.
    """
    n = len(truck_route)
    if n == 1:
        return 0, 0

    def candidates(exclude_depot_end=False):
        cand = []
        for idx, tn in enumerate(truck_route):
            if exclude_depot_end and (idx == 0 or idx == n - 1):
                continue
            cand.append((D[tn][node], idx))
        return cand

    # Launch: closest non-depot if available, else closest overall
    launch_pool = candidates(exclude_depot_end=True)
    if not launch_pool:
        launch_pool = candidates(exclude_depot_end=False)
    launch_pool.sort(key=lambda x: x[0])
    launch_idx = launch_pool[0][1]

    # Land: closest node after launch, prefer non-depot if possible
    land_pool = [(D[tn][node], idx) for idx, tn in enumerate(truck_route) if idx > launch_idx and idx != n - 1]
    if not land_pool:  # fallback allow depot at end
        land_pool = [(D[tn][node], idx) for idx, tn in enumerate(truck_route) if idx > launch_idx]
    if not land_pool:
        land_idx = n - 1
    else:
        land_pool.sort(key=lambda x: x[0])
        land_idx = land_pool[0][1]

    return launch_idx, land_idx


def rebuild_launch_land(truck, d1, d2):
    """
    Return launch/land positions (indices) on the truck route for each drone stop.
    Keeping positions avoids ambiguity when nodes repeat (e.g., depot at start/end).
    """
    launch = {1: [], 2: []}
    land = {1: [], 2: []}
    for dn in d1:
        l_idx, r_idx = nearest_neighbor_indices(truck, dn)
        launch[1].append(l_idx)
        land[1].append(r_idx)
    for dn in d2:
        l_idx, r_idx = nearest_neighbor_indices(truck, dn)
        launch[2].append(l_idx)
        land[2].append(r_idx)
    return launch, land


def operator(solution):
    """Move one customer between truck/drone routes."""
    truck, d1, d2, launch, land = solution
    truck = truck[:]
    d1 = d1[:]
    d2 = d2[:]

    routes = [truck, d1, d2]

    src_idx = random.randint(0, 2)
    src = routes[src_idx]
    if src_idx == 0:
        if len(src) <= 2:
            return solution
        del_pos = random.randint(1, len(src) - 2)
    else:
        if not src:
            return solution
        del_pos = random.randint(0, len(src) - 1)
    node = src.pop(del_pos)

    dst_idx = random.randint(0, 2)
    dst = routes[dst_idx]
    if dst_idx == 0:
        ins_pos = random.randint(1, len(dst) - 1)
    else:
        ins_pos = random.randint(0, len(dst))
    dst.insert(ins_pos, node)

    launch, land = rebuild_launch_land(truck, d1, d2)
    return (truck, d1, d2, launch, land)


def to_solution_dict(solution):
    truck, d1, d2, launch, land = solution
    # launch/land are stored as positions on truck route; convert to 1-based indices
    launch_indices_1 = [idx + 1 for idx in launch[1]]
    landing_indices_1 = [idx + 1 for idx in land[1]]
    launch_indices_2 = [idx + 1 for idx in launch[2]]
    landing_indices_2 = [idx + 1 for idx in land[2]]

    drone_serving = d1 + [-1] + d2
    drone_total_launches = launch_indices_1 + [-1] + launch_indices_2
    drone_total_landings = landing_indices_1 + [-1] + landing_indices_2

    return {
        "part1": truck,
        "part2": drone_serving,
        "part3": drone_total_launches,
        "part4": drone_total_landings,
    }


def cost(solution, calc):
    sol_dict = to_solution_dict(solution)
    total_time, _, _, feas = calc.calculate_total_waiting_time(sol_dict)
    return total_time if feas else math.inf


def simulated_annealing(calc, start_solution=None, T0=5000, alpha=0.995, iter_per_T=300, Tmin=1e-2):
    if start_solution is None:
        start_solution = drone_route()

    # normalize start to use position-based launch/land
    truck0, d10, d20, l0, r0 = start_solution
    lpos, rpos = rebuild_launch_land(truck0, d10, d20)
    current = (truck0, d10, d20, lpos, rpos)
    current_cost = cost(current, calc)
    best = current
    best_cost = current_cost

    T = T0
    while T > Tmin:
        for _ in range(iter_per_T):
            candidate = operator(current)
            cand_cost = cost(candidate, calc)
            delta = cand_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = candidate
                current_cost = cand_cost
                if cand_cost < best_cost:
                    best = candidate
                    best_cost = cand_cost
        T *= alpha
    return best, best_cost


if __name__ == "__main__":
    calc = CalCulateTotalArrivalTime()
    calc.truck_times = T
    calc.drone_times = D
    calc.flight_range = 5500
    calc.depot_index = 0

    best_sol, best_cost = simulated_annealing(calc)
    print("Best cost", best_cost)
    print(to_solution_dict(best_sol))
