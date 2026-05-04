# Truck-and-Drone Routing — ALNS Solver

A self-contained Adaptive Large Neighborhood Search (ALNS) solver for the
single-truck / two-drone routing problem (STRPD). Given an instance file,
it produces a feasible solution, its objective value (total arrival time),
the runtime, and the solution encoded in pipe format.

## Requirements

- **Python 3.9 or newer** (tested on 3.10, 3.11, 3.12, 3.13).
- The Python standard library is sufficient — there are **no third-party
  dependencies**, no `pip install` step is needed.

The solver uses only `random`, `time`, `pathlib`, `collections`, `itertools`,
`argparse`, and `typing`.

## Folder layout

```
Final_Project/
├── solve.py                  # Command-line entry point
├── README.md                 # This file
├── algorithm/
│   ├── truck_and_drone.py    # ALNS main loop + initial solution
│   └── local_search.py       # End-of-run exhaustive local search
├── core/
│   ├── operator_context.py
│   ├── drone_route_utils.py
│   ├── CalCulateTotalArrivalTime.py    # Cost evaluator
│   └── FeasibiltyCheck.py              # Feasibility checker
├── instance_io/
│   └── read_file.py
├── operators/
│   ├── op1_reinsert.py             ├── op2_destroy_repair.py
│   ├── op3_or_opt.py               ├── op4_related_destroy.py
│   ├── op5_truck_2opt.py            ├── op6_TSP_drone_rebuild.py
│   ├── op7_truck_drone_swap.py      ├── op8_drone_sync_tuner.py
│   ├── op9_drone_relocate.py        ├── op10_full_rebuild.py
│   └── escape.py
└── instances/
    ├── R_10.txt   F_10.txt
    ├── R_20.txt   F_20.txt
    ├── R_50.txt   F_50.txt
    └── R_100.txt  F_100.txt
```

All paths inside the code are resolved relative to the source files via
`pathlib.Path(__file__)`, so the project runs identically on Windows,
macOS and Linux as long as Python ≥ 3.9 is on the `PATH`.

## How to run

From a terminal in this folder:

### Solve a bundled instance

```
python solve.py F_100.txt
```

### Solve a custom file

```
python solve.py path/to/your_instance.txt
```

If the argument is not an absolute path and the file does not exist in
the current working directory, the script next tries
`./instances/<argument>`.

### Optional flags

| Flag             | Default | Meaning                                   |
| ---------------- | ------- | ----------------------------------------- |
| `--time-limit S` | `600`   | ALNS wall-clock budget in seconds.        |
| `--seed N`       | random  | RNG seed for a reproducible run.          |

End-of-run local search adds up to ~40 s on top of the ALNS budget on
n>30 instances (10 s on smaller ones).

### Output

```
============================================================
Instance:        F_100.txt
Customers:       100
Best objective:  28572.34
Runtime:         613.41s
Solution (pipe):
0,17,55,... | 12,7,...,-1,3,21,... | 1,4,...,-1,2,6,... | 5,9,...,-1,7,11,...
============================================================
```

The pipe format encodes the four solution parts separated by ` | `:

1. **Truck route** — depot, customer ids in visit order, depot.
2. **Drone customers** — `drone1_nodes, -1, drone2_nodes`.
3. **Launch indices** (1-indexed into the truck route) — `drone1_launch, -1, drone2_launch`.
4. **Land indices** (1-indexed into the truck route) — `drone1_land, -1, drone2_land`.

## Programmatic use

```python
from algorithm.truck_and_drone import solve

result = solve("instances/F_100.txt", time_limit_seconds=600)
print(result["best_objective"], result["runtime"])
print(result["pipe"])
```

Returns a dict with `best_objective`, `pipe`, `runtime`, `solution`,
`n_customers`.

## Algorithm overview

- **Construction**: Nearest-neighbor walk with cyclic vehicle assignment
  produces a feasible initial solution.
- **ALNS** with 6–9 destroy/repair operators selected adaptively per
  instance size.
- **Acceptance**: Record-to-Record Travel (RRT) with convex decay
  (`p=1.5`) for n>60 to widen mid-run exploration.
- **Escape**: When the search stalls, a double-bridge perturbation
  (op14) restarts from the best known solution; falls back to
  related-large destroy (op9) and an aggressive destroy/repair if
  needed.
- **End-of-run intensification**: Three exhaustive best-improvement
  sweeps (truck 2-opt, truck↔drone reassign, drone window retiming)
  cycle until no further improvement.

## Reproducing reported results

The `--seed N` flag is the only knob needed. Without it, every run draws
a fresh seed from the OS entropy source.
