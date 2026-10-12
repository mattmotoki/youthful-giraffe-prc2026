"""Exp-066 single-May measurement entry point."""
from routes import guarded_route


def execute(data_dir, inputs, read_labels, *, draws, draw_digest, ids, dates,
            null_record, null_digest, member="F1-fresh"):
    return guarded_route("measure", data_dir, inputs, read_labels, draws=draws,
                         draw_digest=draw_digest, ids=ids, dates=dates,
                         null_record=null_record, null_digest=null_digest, member=member)
