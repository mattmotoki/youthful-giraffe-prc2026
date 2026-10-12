"""The only executable exp-066 producer, measurement, and report routes."""
from __future__ import annotations

import argparse
from pathlib import Path

import future_wrapper as gates
import may_protocol as may
from member_adapters import adapter


ROUTES = ("produce", "measure", "report")


def guarded_route(route, data_dir, producing_inputs, continuation, *, draws=None,
                  draw_digest=None, ids=None, dates=None, null_record=None,
                  null_digest=None, member="F1-fresh"):
    if route not in ROUTES:
        raise ValueError(route)
    gates.reject_forbidden(producing_inputs)
    gates.verify_raw(data_dir)
    settings = adapter(member)
    if route in ("measure", "report"):
        may.require_draws(Path(draws), draw_digest, ids, dates)
        may.require_matched_null(Path(null_record), null_digest)
    return continuation(settings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route", choices=ROUTES)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--input", action="append", default=[])
    parser.parse_args()
    raise SystemExit("registered caller must supply the sealed continuation and gate records")


if __name__ == "__main__": main()
