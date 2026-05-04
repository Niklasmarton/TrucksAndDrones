"""
Single-run exam-condition simulation across all datasets.
One run per dataset, 10000 iterations (matches exam constraint).
"""
import sys
import time
from pathlib import Path

ASSIGNMENT_DIR = Path(__file__).resolve().parent
ALG_DIR = ASSIGNMENT_DIR / "algorithms"
sys.path.append(str(ALG_DIR))

import Truck_and_Drone as td

TEST_FILES_DIR = ASSIGNMENT_DIR.parent / "Test_files"
DATASETS = ["R_10.txt", "F_10.txt", "R_20.txt", "F_20.txt",
            "R_50.txt", "F_50.txt", "R_100.txt", "F_100.txt"]

ITERATIONS = 10000

results = []
overall_start = time.time()

for fname in DATASETS:
    ds_path = TEST_FILES_DIR / fname
    if not ds_path.exists():
        print(f"SKIP {fname} — not found")
        continue
    td.file_name = fname
    instance_data = td.load_instance(ds_path)
    n = instance_data["n_customers"]

    if n <= 15:
        warmup, escape_stall, seg = 500, max(150, 4 * n), max(15, n)
    elif n <= 30:
        warmup, escape_stall, seg = 500, max(200, 6 * n), max(20, n)
    else:
        warmup, escape_stall, seg = 500, 1000, max(20, n)

    print(f"\n{'='*60}\nDataset: {fname}  (n={n}, iters={ITERATIONS})\n{'='*60}", flush=True)
    t0 = time.time()
    best_sol, best_cost = td.run_statistics(
        None,
        instance_data=instance_data,
        solution_factory=lambda id=instance_data: td.build_initial_solution(id),
        runs=1,
        warmup_iterations=warmup,
        iterations=ITERATIONS,
        final_temperature=0.1,
        cache_limit=200000,
        reaction_factor=0.15,
        segment_length=seg,
        escape_stall_limit=escape_stall,
        verbose=False,
        return_metrics=False,
        print_solution_pipe=False,
        plot_delta_scatter_best_run=False,
        plot_weights_best_run=False,
        plot_temperature_best_run=False,
        plot_acceptance_probability_best_run=False,
        plot_accepted_objective_best_run=False,
        save_best_runs_log=False,
        save_best_solution_plot=False,
    )
    elapsed = time.time() - t0
    results.append((fname, n, best_cost, elapsed))
    print(f">>> {fname}: best={best_cost:.2f}  time={elapsed:.1f}s", flush=True)

total_time = time.time() - overall_start
print(f"\n\n{'='*60}\nSUMMARY (single run, {ITERATIONS} iters each)\n{'='*60}")
print(f"{'Dataset':<12} {'n':>4} {'Best':>12} {'Time(s)':>10}")
for fname, n, cost, elapsed in results:
    print(f"{fname:<12} {n:>4} {cost:>12.2f} {elapsed:>10.1f}")
print(f"\nTotal wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
