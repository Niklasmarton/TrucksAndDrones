# CLI for the truck and drone ALNS solver.
#
# usage:
#   # solve every *.txt in ./instances/ in parallel (default with no arg)
#   python solve.py
#
#   # solve a single instance
#   python solve.py F_100.txt
#   python solve.py path/to/your_instance.txt
#
#   # tweak the per-instance time budget, fix a seed, or set worker count
#   python solve.py --time-limit 300
#   python solve.py F_100.txt --seed 42
#   python solve.py --workers 4
#
# each solved instance prints best objective, runtime, customers and the
# pipe-encoded solution. when solving the whole folder, each result is also
# written to ./outputs/<instance>.result.txt and a summary table is printed
# at the end.

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ALG_DIR = PROJECT_DIR / "algorithm"
INSTANCES_DIR = PROJECT_DIR / "instances"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
if str(ALG_DIR) not in sys.path:
    sys.path.insert(0, str(ALG_DIR))


def _resolve_instance(arg):
    p = Path(arg)
    if p.is_file():
        return p
    fallback = INSTANCES_DIR / arg
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"Instance file not found: {arg}\n"
        f"  tried: {p.resolve()}\n"
        f"  tried: {fallback.resolve()}"
    )


def _list_instances():
    if not INSTANCES_DIR.is_dir():
        raise FileNotFoundError(
            f"instances/ folder not found at {INSTANCES_DIR}"
        )
    return sorted(INSTANCES_DIR.glob("*.txt"))


def _print_result(name, result, header_width=72):
    print("=" * header_width)
    print(f"Instance:        {name}")
    print(f"Customers:       {result['n_customers']}")
    print(f"Best objective:  {result['best_objective']:.2f}")
    print(f"Runtime:         {result['runtime']:.2f}s")
    print("Solution (pipe):")
    print(result["pipe"])
    print("=" * header_width, flush=True)


def _save_result(instance_path, result):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"{instance_path.stem}.result.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"instance: {instance_path.name}\n")
        f.write(f"customers: {result['n_customers']}\n")
        f.write(f"best_objective: {result['best_objective']:.2f}\n")
        f.write(f"runtime_seconds: {result['runtime']:.2f}\n")
        f.write(f"pipe: {result['pipe']}\n")
    return out_path


# direct (non-pool) single-instance solve, used by single-arg CLI mode
def _solve_single(instance_path, time_limit, seed):
    from truck_and_drone import solve
    print(f"\n>>> Solving {instance_path.name}  "
          f"(time_limit={time_limit:.0f}s"
          + (f", seed={seed}" if seed is not None else "")
          + ")", flush=True)
    return solve(instance_path,
                 time_limit_seconds=time_limit,
                 seed=seed)


# pool worker, solves one instance in its own process. re-imports the solver
# because spawn-mode workers don't inherit parent-process module state.
# returns a dict the parent can format.
def _worker(args):
    instance_path, time_limit, seed = args
    if str(ALG_DIR) not in sys.path:
        sys.path.insert(0, str(ALG_DIR))
    from truck_and_drone import solve

    t0 = time.perf_counter()
    try:
        result = solve(instance_path,
                       time_limit_seconds=time_limit,
                       seed=seed)
        return {
            "instance_path": instance_path,
            "name": instance_path.name,
            "ok": True,
            "result": result,
            "wall": time.perf_counter() - t0,
        }
    except Exception as e:
        return {
            "instance_path": instance_path,
            "name": instance_path.name,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "wall": time.perf_counter() - t0,
        }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Solve Truck-and-Drone instances with ALNS. "
            "Without an INSTANCE argument, every *.txt file in "
            "./instances/ is solved in parallel."
        ),
    )
    parser.add_argument(
        "instance", nargs="?", default=None,
        help=("Instance filename (e.g. F_100.txt). If omitted, every *.txt "
              "file in ./instances/ is solved in parallel."),
    )
    parser.add_argument(
        "--time-limit", type=float, default=600.0,
        help=("ALNS time budget per instance in seconds (default: 600). "
              "End-of-run local search adds up to ~40s on top."),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed (default: random per instance).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=("Number of parallel worker processes for batch mode "
              "(default: min(CPU count, number of instances))."),
    )
    args = parser.parse_args()

    if args.instance is not None:
        instance_path = _resolve_instance(args.instance)
        result = _solve_single(instance_path, args.time_limit, args.seed)
        _print_result(instance_path.name, result)
        return

    # Batch mode: solve every *.txt file in instances/ in parallel.
    instance_files = _list_instances()
    if not instance_files:
        print(f"No *.txt instances found in {INSTANCES_DIR}", file=sys.stderr)
        sys.exit(1)

    cpu_count = os.cpu_count() or 1
    n_workers = args.workers if args.workers is not None else min(cpu_count, len(instance_files))
    n_workers = max(1, min(n_workers, len(instance_files)))

    print(f"Batch mode: {len(instance_files)} instance(s) in {INSTANCES_DIR}")
    print(f"Workers:                 {n_workers} (CPU count: {cpu_count})")
    print(f"Per-instance time limit: {args.time_limit:.0f}s")
    if args.seed is not None:
        print(f"Seed: {args.seed} (same seed used for every instance)")
    waves = (len(instance_files) + n_workers - 1) // n_workers
    print(f"Estimated wall-clock total: "
          f"~{waves * (args.time_limit + 40) / 60:.0f} min "
          f"({waves} wave of up to {n_workers} parallel solves)", flush=True)
    print("=" * 72, flush=True)

    jobs = [(p, args.time_limit, args.seed) for p in instance_files]

    summary = []
    t_total = time.perf_counter()
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(_worker, jobs):
            elapsed = time.perf_counter() - t_total
            if r["ok"]:
                result = r["result"]
                _print_result(r["name"], result)
                out_path = _save_result(r["instance_path"], result)
                print(f"    -> result saved to "
                      f"{out_path.relative_to(PROJECT_DIR)}  "
                      f"(elapsed {elapsed/60:.1f} min)", flush=True)
                summary.append((
                    r["name"],
                    result["n_customers"],
                    result["best_objective"],
                    result["runtime"],
                    None,
                ))
            else:
                print(f"  !! {r['name']} failed: {r['error']}  "
                      f"(elapsed {elapsed/60:.1f} min)", flush=True)
                summary.append((r["name"], None, None, None, r["error"]))

    total_wall = time.perf_counter() - t_total

    # Sort summary by instance name for a stable report (workers complete in any order).
    summary.sort(key=lambda row: row[0])

    print("\n" + "=" * 72)
    print(f"BATCH SUMMARY  ({total_wall/60:.1f} min wall-clock, "
          f"{n_workers} workers)")
    print("=" * 72)
    print(f"{'instance':<25} {'n':>5} {'best':>14} {'runtime':>10}  status")
    print("-" * 72)
    for name, n, best, rt, err in summary:
        if err is not None:
            print(f"{name:<25} {'-':>5} {'-':>14} {'-':>10}  FAILED ({err})")
        else:
            print(f"{name:<25} {n:>5} {best:>14.2f} {rt:>9.2f}s  ok")
    print("=" * 72)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
