"""Upload a transport dataset version and retain the authoritative API response."""
import argparse
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("output", type=Path)
args = parser.parse_args()
if args.output.exists(): raise FileExistsError(args.output)
api = KaggleApi(); api.authenticate()
response = api.dataset_create_version(args.folder, "accepted round-1 repair authoritative version",
                                      quiet=True, dir_mode="skip")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(str(response) + "\n")
