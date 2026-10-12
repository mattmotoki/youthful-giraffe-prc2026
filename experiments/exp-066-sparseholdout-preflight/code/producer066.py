"""Exp-066 May producer entry point; old producers are callable dependencies only."""
from routes import guarded_route


def execute(data_dir, inputs, fit, member):
    return guarded_route("produce", data_dir, inputs, fit, member=member)
