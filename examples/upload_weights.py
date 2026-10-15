"""Upload a checkpoint's folded coefficients into your own EE assets once."""

import argparse
import json
from pathlib import Path
import uuid

import ee
import torch

from tensor_ee import effective_convolutions, export_weights
from tensor_ee.segwater_pytorch import load_segwater_checkpoint
from examples._ee import add_credentials_argument, initialize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    add_credentials_argument(parser)
    args = parser.parse_args(argv)
    if args.manifest.exists():
        parser.error("Manifest already exists. Inspect its task IDs before uploading again.")
    torch.set_num_threads(2)
    bundle = load_segwater_checkpoint(args.checkpoint)
    parameters = effective_convolutions(bundle.compilable_model)
    initialize(args.project, args.credentials)
    namespace = f"projects/{args.project}/assets/tensor_ee_weights_{uuid.uuid4().hex[:12]}"

    def progress(manifest):
        states = {}
        for shard in manifest["shards"]:
            state = shard["status"]
            states[state] = states.get(state, 0) + 1
        print(json.dumps({"status": manifest["status"], "shards": states}), flush=True)

    export_weights(parameters, namespace, args.manifest, on_update=progress)
    print(f"Ready: {args.manifest}")


if __name__ == "__main__":
    main()
