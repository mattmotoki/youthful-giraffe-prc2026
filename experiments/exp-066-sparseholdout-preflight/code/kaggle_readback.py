"""Write authoritative Kaggle dataset or kernel API metadata once."""
import argparse
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=("dataset", "kernel")); parser.add_argument("ref")
parser.add_argument("output", type=Path); args = parser.parse_args()
if args.output.exists(): raise FileExistsError(args.output)
owner, slug = args.ref.split("/", 1)
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as client:
    if args.kind == "dataset":
        request = ApiGetDatasetRequest(); request.owner_slug=owner; request.dataset_slug=slug
        response = client.datasets.dataset_api_client.get_dataset(request)
    else:
        request = ApiGetKernelRequest(); request.user_name=owner; request.kernel_slug=slug
        response = client.kernels.kernels_api_client.get_kernel(request).metadata
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(str(response) + "\n")
