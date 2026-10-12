"""Exp-066-local adapters for the three fixed member recipes."""
from __future__ import annotations

THREAD_CAP = 3


def effective_settings(member):
    if member == "F1-fresh":
        return {"lightgbm": {"num_threads": 3}, "catboost": {"thread_count": 3}}
    if member == "L2-hub8725":
        return {"lightgbm": {"num_threads": 3}}
    if member == "T-tailv4":
        return {"tail_model": {"n_jobs": 3}, "lightgbm": {"num_threads": 3}}
    raise ValueError(member)


def assert_effective(settings):
    seen = []
    for library in settings.values():
        for key in ("num_threads", "thread_count", "n_jobs"):
            if key in library:
                seen.append(library[key])
                if not 1 <= int(library[key]) <= THREAD_CAP:
                    raise ValueError(f"effective {key} exceeds {THREAD_CAP}")
    if not seen:
        raise ValueError("no effective library thread setting")
    return True


def adapter(member):
    settings = effective_settings(member)
    assert_effective(settings)
    return settings
