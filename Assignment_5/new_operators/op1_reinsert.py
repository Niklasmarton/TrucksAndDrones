from pathlib import Path
import sys
import importlib.util

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
OPS_DIR = ASSIGNMENT_DIR / "operators"
spec = importlib.util.spec_from_file_location("assignment5_base_op1", OPS_DIR / "op1_reinsert.py")
_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_base)


def set_operator_context(truck_times, drone_times, flight_range, depot_index=0):
    _base.set_operator_context(truck_times, drone_times, flight_range, depot_index)


def set_search_progress(progress):
    if hasattr(_base, "set_search_progress"):
        _base.set_search_progress(progress)


def operator(solution):
    return _base.operator(solution)
