from collections import Counter
from typing import List, Tuple, Dict, Any


class SolutionFeasibility:
    """
    Feasibility checker for truck–drone solutions.

    This version is fully self-contained: it stores the problem instance
    (n_nodes, n_drones, depot_index, drone_times, flight_range) internally
    instead of relying on an external solver object.

    Separator convention:
      - part2 / part3 / part4 use -1 as separator between drones (replaces "X").
    """

    def __init__(
        self,
        n_nodes: int,
        n_drones: int,
        depot_index: int,
        drone_times: List[List[int]],
        flight_range: int,
    ) -> None:
        self.n_nodes = n_nodes
        self.n_drones = n_drones
        self.depot_index = depot_index
        self.drone_times = drone_times
        self.flight_range = flight_range

    # ----------------------------------------------------------------------
    # 1) Truck must start and stop at depot
    # ----------------------------------------------------------------------
    def is_truck_route_feasible(self, solution: Dict[str, Any]) -> bool:
        """
        Check truck-route feasibility:
        1) Truck must start and stop at depot.
        2) All nodes in part1 must be valid (0..n_nodes-1).
        3) No extra depot visits in the middle of the route.
        """
        part1 = solution.get("part1", [])
        if not part1 or len(part1) < 2:
            return False

        depot = self.depot_index

        # start & end
        if part1[0] != depot or part1[-1] != depot:
            return False

        # all nodes valid
        for node in part1:
            if not (0 <= node < self.n_nodes):
                return False

        # no stray depots in the middle
        for node in part1[1:-1]:
            if node == depot:
                return False

        return True

    # ----------------------------------------------------------------------
    # 2) Complete solution: all customers present exactly once truck+drone
    # ----------------------------------------------------------------------
    def is_complete_solution(self, solution: Dict[str, Any]) -> bool:
        """
        Check that the solution is complete and each customer is served exactly once:
        - All customers 1..n_nodes-1 appear exactly once across truck (part1) and drones (part2).
        - Depot (0) must NOT appear in part2.
        """
        part1 = solution.get("part1", [])
        part2 = solution.get("part2", [])
        part3 = solution.get("part3", [])
        part4 = solution.get("part4", [])

        depot = self.depot_index

        # Truck customers (no depot)
        truck_customers = [c for c in part1 if c != depot]

        # Drone customers (ignore separators -1)
        drone_customers = [x for x in part2 if x != -1]

        # No depot in drone customers
        if any(c == depot for c in drone_customers):
            return False
        
        # No more than self.n_drones segments in parts 2, 3, 4
        if part2.count(-1) > self.n_drones - 1 or part3.count(-1) > self.n_drones - 1 or part4.count(-1) > self.n_drones - 1:
            return False 

        freq = Counter(truck_customers + drone_customers)
        expected_customers = set(range(1, self.n_nodes))

        # 1) Every expected customer must appear
        if set(freq.keys()) != expected_customers:
            return False

        # 2) Each must appear exactly once
        for cid in expected_customers:
            if freq[cid] != 1:
                return False

        return True

    # ----------------------------------------------------------------------
    # 3) Structure / separators consistency
    # ----------------------------------------------------------------------
    def are_parts_consistent(self, solution: Dict[str, Any]) -> bool:
        """
        Consistency check for part2 (customers + -1), part3 (launch cells), part4 (reconvene cells).

        Ensures:
        - get_trips_per_drone() can be computed without ValueError.
        - Each non-separator launch/reconvene cell is an int, valid in part1, and launch < reconvene.
        - Separator counts in part3/part4 are either 0 or equal to the number of separators in part2.
        """
        part1 = solution.get("part1", [])
        part2 = solution.get("part2", [])
        part3 = solution.get("part3", [])
        part4 = solution.get("part4", [])

        # NEW CHECK: ensure depot (0) is NOT placed in part2.
        # If 0 appears, it would be interpreted as a customer, which is invalid.
        if any(item == self.depot_index for item in part2 if item != -1):
            return False

        # Basic length equality
        if len(part3) != len(part4):
            return False

        # Ensure get_trips_per_drone doesn't raise ValueError
        try:
            _ = self.get_trips_per_drone(solution)
        except ValueError:
            return False

        # Separator counts
        sep2 = sum(1 for x in part2 if x == -1)
        sep3 = sum(1 for x in part3 if x == -1)
        sep4 = sum(1 for x in part4 if x == -1)

        # If part3/4 use separators, they should mirror part2; using no separators is also allowed.
        if (sep3 > 0 and sep3 != sep2) or (sep4 > 0 and sep4 != sep2):
            return False

        # All non-separator launch/reconvene entries must be valid cells and have launch < reconvene
        part3_clean = [x for x in part3 if x != -1]
        part4_clean = [x for x in part4 if x != -1]

        if len(part3_clean) != len(part4_clean):
            return False

        for lc, cust, rc in zip(part3, part2, part4):
            try:
                launch_cell = int(lc)
                reconvene_cell = int(rc)
            except (TypeError, ValueError):
                return False
            
            if cust == -1 and lc == -1 and rc == -1: 
               continue          

            if lc == -1 or rc == -1 or cust == -1:
               #print(lc,cust,rc)
               return False 

            if not self.is_valid_cell(launch_cell, part1):
                return False
            if not self.is_valid_cell(reconvene_cell, part1):
                return False
            if launch_cell >= reconvene_cell:
                return False

        return True

    # ----------------------------------------------------------------------
    # Basic helpers
    # ----------------------------------------------------------------------
    def is_valid_cell(self, cell: int, part1: List[int]) -> bool:
        """Check if cell number is valid (1-based indexing)."""
        return 1 <= cell <= len(part1)

    def get_customer_from_cell(self, cell: int, part1: List[int]) -> int:
        """Get customer from cell number (1-based indexing)."""
        if not self.is_valid_cell(cell, part1):
            return -1
        return part1[cell - 1]

    # ----------------------------------------------------------------------
    # Decode trips per drone (from part2/3/4)
    # ----------------------------------------------------------------------
    def get_trips_per_drone(self, solution: Dict[str, Any]) -> List[List[Tuple[int, int]]]:
        """
        Return trips per drone as lists of (launch_cell, reconvene_cell) tuples.
        Robust to -1 separators in part2 and -1 entries in part3/part4.
        """
        part2 = solution.get("part2", [])
        part3 = solution.get("part3", [])
        part4 = solution.get("part4", [])

        # Clean part3/part4: keep only non-separator entries
        part3_clean = [x for x in part3 if x != -1]
        part4_clean = [x for x in part4 if x != -1]

        # Count non-separator items in part2
        non_sep_count = sum(1 for x in part2 if x != -1)

        # Sanity check
        if len(part3_clean) != non_sep_count or len(part4_clean) != non_sep_count:
            raise ValueError(
                f"Inconsistent parts: part2 has {non_sep_count} non-separator items, "
                f"but part3 has {len(part3_clean)} and part4 has {len(part4_clean)} cleaned entries."
            )

        # Prepare empty lists for each drone
        trips_per_drone: List[List[Tuple[int, int]]] = [[] for _ in range(self.n_drones)]

        # Walk through part2, track current drone index and assign cleaned part3/4 entries sequentially
        drone_idx = 0
        clean_idx = 0
        for item in part2:
            if item == -1:
                # separator -> next drone
                drone_idx += 1
                continue

            # defensive: clamp drone_idx to available drones
            if drone_idx >= self.n_drones:
                drone_idx = self.n_drones - 1

            # take next cleaned launch/return pair
            launch = part3_clean[clean_idx]
            reconvene = part4_clean[clean_idx]
            trips_per_drone[drone_idx].append((int(launch), int(reconvene)))
            clean_idx += 1

        # Ensure we return exactly n_drones items (pad with empty lists if needed)
        while len(trips_per_drone) < self.n_drones:
            trips_per_drone.append([])

        return trips_per_drone[: self.n_drones]

    # ----------------------------------------------------------------------
    # Decode customer routes per drone from part2
    # ----------------------------------------------------------------------
    def get_drone_routes_from_parts(self, solution: Dict[str, Any]) -> List[List[int]]:
        """
        Decode customers per drone from part2, using -1 as separator.
        Example:
            part2 = [9, 4, -1, 2, 10, 7]
            n_drones = 2
            -> [[9, 4], [2, 10, 7]]
        """
        part2 = solution.get("part2", [])
        routes: List[List[int]] = [[] for _ in range(self.n_drones)]
        drone_idx = 0

        for item in part2:
            if item == -1:
                drone_idx += 1
                continue
            if drone_idx >= self.n_drones:
                drone_idx = self.n_drones - 1
            routes[drone_idx].append(int(item))

        return routes

    # ----------------------------------------------------------------------
    # Single drone-trip feasibility
    # ----------------------------------------------------------------------
    def is_feasible_drone_trip(
        self,
        customer: int,
        launch_cell: int,
        reconvene_cell: int,
        drone_route_idx: int,
        solution: Dict[str, Any],
    ) -> bool:
        """Check if drone trip is feasible with correct per-UAV sequencing."""
        part1 = solution["part1"]

        # Basic validation
        if not self.is_valid_cell(launch_cell, part1) or not self.is_valid_cell(reconvene_cell, part1):
            return False

        # Must launch before reconvene
        if launch_cell >= reconvene_cell:
            return False

        launch_customer = self.get_customer_from_cell(launch_cell, part1)
        reconvene_customer = self.get_customer_from_cell(reconvene_cell, part1)

        # Flight time constraint
        flight_time = (
            self.drone_times[launch_customer][customer]
            + self.drone_times[customer][reconvene_customer]
        )
        if flight_time > self.flight_range:
            return False

        # (Sequencing constraints handled in are_all_drone_trips_feasible via trips_per_drone)
        return True

    # ----------------------------------------------------------------------
    # 4) All drone trips feasible (flight range + sequencing)
    # ----------------------------------------------------------------------
    def are_all_drone_trips_feasible(self, solution: Dict[str, Any]) -> bool:
        """
        Check that **all** drone trips encoded in (part2, part3, part4) are feasible.
        """
        trips_per_drone = self.get_trips_per_drone(solution)

        # sequencing: for each drone, reconvene must be <= next launch
        for trips in trips_per_drone:
            for (l1, r1), (l2, r2) in zip(trips, trips[1:]):
                if l2 < r1:
                    return False

        part3 = solution.get("part3", [])
        part4 = solution.get("part4", [])

        # First ensure structural consistency
        if not self.are_parts_consistent(solution):
            return False

        drone_routes = self.get_drone_routes_from_parts(solution)

        part3_clean = [x for x in part3 if x != -1]
        part4_clean = [x for x in part4 if x != -1]
        expected_pairs = len(part3_clean)

        trip_idx = 0
        for d_idx, route in enumerate(drone_routes):
            for cust in route:
                if trip_idx >= expected_pairs:
                    # more customers than recorded trips
                    return False

                try:
                    lc_raw = part3_clean[trip_idx]
                    rc_raw = part4_clean[trip_idx]
                    launch_cell = int(lc_raw)
                    reconvene_cell = int(rc_raw)
                except (TypeError, ValueError, IndexError):
                    return False

                if not self.is_feasible_drone_trip(
                    cust, launch_cell, reconvene_cell, d_idx, solution
                ):
                    return False

                trip_idx += 1

        # All pairs in part3/part4 should be used
        if trip_idx != expected_pairs:
            return False

        return True

    # ----------------------------------------------------------------------
    # 5) Single “master” check
    # ----------------------------------------------------------------------
    def is_solution_feasible(self, solution: Dict[str, Any]) -> bool:
        """
        Global feasibility check for any solution.

        Conditions:
        1) Truck route feasible (start/stop at depot, valid nodes, no stray depots).
        2) Complete solution: all customers present, each served exactly once (truck+drone).
        3) part2/part3/part4 consistent (separators, lengths, valid cells, launch<reconvene).
        4) All drone trips feasible w.r.t. range and sequencing.
        """
        if not self.is_truck_route_feasible(solution):
            return False

        if not self.is_complete_solution(solution):
            return False

        if not self.are_parts_consistent(solution):
            return False

        if not self.are_all_drone_trips_feasible(solution):
            return False

        return True
