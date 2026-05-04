"""
Robustness study: multiple runs per dataset, reporting average and best.
- n=10, n=20: 5 runs
- n=50: 4 runs
- n=100: 3 runs
"""
import sys
import time
from pathlib import Path

ASSIGNMENT_DIR = Path(__file__).resolve().parent
ALG_DIR = ASSIGNMENT_DIR / "algorithms"
sys.path.append(str(ALG_DIR))

import Truck_and_Drone as td

TEST_FILES_DIR = ASSIGNMENT_DIR.parent / "Test_files"
DATASETS = [
    ("R_10.txt",  5),
    ("F_10.txt",  5),
    ("R_20.txt",  5),
    ("F_20.txt",  5),
    ("R_50.txt",  4),
    ("F_50.txt",  4),
    ("R_100.txt", 3),
    ("F_100.txt", 3),
]

ITERATIONS = 10000

results = []
overall_start = time.time()

for fname, n_runs in DATASETS:
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

    print(f"\n{'='*60}\nDataset: {fname}  (n={n}, runs={n_runs}, iters={ITERATIONS})\n{'='*60}", flush=True)
    t0 = time.time()
    _, _, metrics = td.run_statistics(
        None,
        instance_data=instance_data,
        solution_factory=lambda id=instance_data: td.build_initial_solution(id),
        runs=n_runs,
        warmup_iterations=warmup,
        iterations=ITERATIONS,
        final_temperature=0.1,
        cache_limit=200000,
        reaction_factor=0.15,
        segment_length=seg,
        escape_stall_limit=escape_stall,
        verbose=False,
        return_metrics=True,
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
    avg = metrics["average_score"]
    best = metrics["best_score"]
    results.append((fname, n, n_runs, avg, best, elapsed))
    print(f">>> {fname}: avg={avg:.2f}  best={best:.2f}  total_time={elapsed:.1f}s  ({elapsed/n_runs:.1f}s/run)", flush=True)

total_time = time.time() - overall_start
print(f"\n\n{'='*72}\nROBUSTNESS SUMMARY\n{'='*72}")
print(f"{'Dataset':<12} {'n':>4} {'runs':>5} {'Avg':>12} {'Best':>12} {'Time(s)':>10}")
for fname, n, n_runs, avg, best, elapsed in results:
    print(f"{fname:<12} {n:>4} {n_runs:>5} {avg:>12.2f} {best:>12.2f} {elapsed:>10.1f}")
print(f"\nTotal wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
