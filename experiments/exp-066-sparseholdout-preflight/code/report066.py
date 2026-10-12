"""Exp-066 single-May report entry point."""
from routes import guarded_route


def execute(data_dir, inputs, read_values, *, draws, draw_digest, ids, dates,
            null_record, null_digest, member="F1-fresh"):
    return guarded_route("report", data_dir, inputs, read_values, draws=draws,
                         draw_digest=draw_digest, ids=ids, dates=dates,
                         null_record=null_record, null_digest=null_digest, member=member)
