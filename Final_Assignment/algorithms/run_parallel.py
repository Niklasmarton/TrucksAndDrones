"""Run all 8 STRPD datasets in parallel, one process per dataset.

Mirrors the exam evaluation profile: 1 run per dataset, 10000 iterations,
all 8 datasets dispatched concurrently via multiprocessing. Use spawn
start method so each worker gets its own copy of Truck_and_Drone module
state (operator_context, file_name, etc.).

Usage:
    python Final_Assignment/algorithms/run_parallel.py
    python Final_Assignment/algorithms/run_parallel.py --runs 5 --iters 10000
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
ALG_DIR = ASSIGNMENT_DIR / "algorithms"
TEST_FILES_DIR = ASSIGNMENT_DIR.parent / "Test_files"

DATASETS = [
    "R_10.txt",
    "F_10.txt",
    "R_20.txt",
    "F_20.txt",
    "R_50.txt",
    "F_50.txt",
    "R_100.txt",
    "F_100.txt",
]


def _worker(args):
    """Run `runs` invocations of run_statistics on a single dataset.

    Imports happen inside the worker so each process initializes
    Truck_and_Drone module-level state independently (operator_context
    is module-level and would be shared under fork — spawn isolates it).
    """
    fname, n_runs, iterations, time_limit = args

    if str(ALG_DIR) not in sys.path:
        sys.path.append(str(ALG_DIR))
    import Truck_and_Drone as td

    ds_path = TEST_FILES_DIR / fname
    td.file_name = fname
    instance_data = td.load_instance(ds_path)
    n = instance_data["n_customers"]

    # Size-tiered escape_stall_limit. Small instances stay at 4·n (responsive
    # escape); n>30 raised to 8·n so escape no longer interrupts intensification.
    if n > 30:
        esc_stall = max(800, 8 * n)
    else:
        esc_stall = max(150, 4 * n)

    scores = []
    runtimes = []
    escape_stats = []  # (calls, improving_calls)
    t_total = time.time()
    for _ in range(n_runs):
        t_run = time.time()
        _, _, metrics = td.run_statistics(
            None,
            instance_data=instance_data,
            solution_factory=lambda id=instance_data: td.build_initial_solution(id),
            runs=1,
            warmup_iterations=500,
            iterations=iterations,
            final_temperature=0.1,
            cache_limit=200000,
            reaction_factor=0.15,
            segment_length=max(15, n),
            escape_stall_limit=esc_stall,
            verbose=False,
            return_metrics=True,
            print_solution_pipe=False,
            plot_delta_scatter_best_run=False,
            plot_weights_best_run=False,
            plot_temperature_best_run=False,
            plot_acceptance_probability_best_run=False,
            plot_accepted_objective_best_run=True,
            save_best_runs_log=False,
            save_best_solution_plot=True,
            time_limit_seconds=time_limit,
        )
        scores.append(metrics["best_score"])
        runtimes.append(time.time() - t_run)
        esc_log = metrics.get("escape_log") or []
        escape_stats.append((len(esc_log), sum(1 for e in esc_log if e[3])))

    total_elapsed = time.time() - t_total
    total_esc_calls = sum(c for c, _ in escape_stats)
    total_esc_imp = sum(i for _, i in escape_stats)

    return {
        "dataset": fname,
        "n_customers": n,
        "esc_stall": esc_stall,
        "scores": scores,
        "best": min(scores),
        "avg": sum(scores) / len(scores),
        "total_elapsed": total_elapsed,
        "avg_runtime": sum(runtimes) / len(runtimes),
        "esc_calls": total_esc_calls,
        "esc_improving": total_esc_imp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per dataset (exam profile = 1)")
    parser.add_argument("--iters", type=int, default=10_000_000,
                        help="Iteration cap (set very high so time-limit binds first)")
    parser.add_argument("--time-limit", type=float, default=600.0,
                        help="Wall-time budget per run in seconds (0 = disabled)")
    parser.add_argument("--workers", type=int, default=len(DATASETS),
                        help="Process pool size (default: one per dataset)")
    args = parser.parse_args()

    time_limit = args.time_limit if args.time_limit > 0 else None
    jobs = [(fname, args.runs, args.iters, time_limit) for fname in DATASETS]

    budget = f"{time_limit:.0f}s" if time_limit else f"{args.iters} iters"
    print(f"Parallel run: {len(jobs)} datasets × {args.runs} runs × "
          f"{budget}  (workers={args.workers})", flush=True)
    print("=" * 70, flush=True)

    # Classmate baselines (best reported averages from peers).
    # R_10/F_10/R_20/R_50/F_50/R_100/F_100: Martin Sollesnes Kummeneje
    # F_20: Sigurd Dårflot Olsen
    BASELINE = {
        "R_10.txt": 585.00, "F_10.txt": 1412.00,
        "R_20.txt": 2006.80, "F_20.txt": 3263.00,
        "R_50.txt": 7657.50, "F_50.txt": 10938.90,
        "R_100.txt": 20965.90, "F_100.txt": 28602.10,
    }

    t_wall = time.time()
    results = []
    with mp.Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(_worker, jobs):
            results.append(r)
            esc_pct = (100.0 * r["esc_improving"] / r["esc_calls"]) if r["esc_calls"] else 0.0
            tag = (
                f"{r['dataset']:10s}  best={r['best']:>10.2f}  "
                f"runtime={r['avg_runtime']:>5.1f}s  "
                f"esc_stall={r['esc_stall']:>4d}  "
                f"esc_calls={r['esc_calls']:>4d}  "
                f"esc_imp={r['esc_improving']} ({esc_pct:.1f}%)"
            )
            print(tag, flush=True)
    wall = time.time() - t_wall

    results.sort(key=lambda r: DATASETS.index(r["dataset"]))

    print("=" * 70, flush=True)
    print("Scoreboard vs prior 600s baseline:", flush=True)
    print(f"{'dataset':<10}  {'baseline':>10}  {'new':>10}  {'delta':>10}  "
          f"{'esc_calls':>10}  {'esc_imp%':>9}", flush=True)
    total_delta = 0.0
    for r in results:
        base = BASELINE.get(r["dataset"], float("nan"))
        delta = r["best"] - base
        total_delta += delta
        esc_pct = (100.0 * r["esc_improving"] / r["esc_calls"]) if r["esc_calls"] else 0.0
        marker = " ✓" if delta < 0 else (" ✗" if delta > 0 else "  ")
        print(f"{r['dataset']:<10}  {base:>10.2f}  {r['best']:>10.2f}  "
              f"{delta:>+10.2f}{marker}  {r['esc_calls']:>10d}  {esc_pct:>8.1f}%",
              flush=True)
    print(f"\nNet delta vs baseline: {total_delta:+.2f} (negative = improvement)", flush=True)
    print(f"Wall-clock total: {wall:.1f}s", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
