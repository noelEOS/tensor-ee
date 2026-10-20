"""Upload folded weights from any supported CNN using a local checkpoint factory."""

import argparse
import importlib
import json
from pathlib import Path
import uuid

import torch

from examples._ee import add_credentials_argument, initialize
from tensor_ee import effective_convolutions, export_weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--factory", required=True,
                        help="Trusted module:callable taking a checkpoint Path and returning a compilable eval CNN")
    parser.add_argument("--manifest", type=Path, required=True)
    add_credentials_argument(parser)
    args = parser.parse_args(argv)
    if args.manifest.exists():
        parser.error("Manifest already exists. Inspect its task IDs before uploading again.")
    module_name, separator, name = args.factory.partition(":")
    if not separator or not module_name or not name:
        parser.error("--factory must be module:callable")
    torch.set_num_threads(2)
    factory = getattr(importlib.import_module(module_name), name)
    model = factory(args.checkpoint)
    parameters = effective_convolutions(model)
    initialize(args.project, args.credentials)
    namespace = f"projects/{args.project}/assets/tensor_ee_weights_{uuid.uuid4().hex[:12]}"

    def progress(manifest):
        states = {}
        for shard in manifest["shards"]:
            state = shard["status"]
            states[state] = states.get(state, 0) + 1
        print(json.dumps({"status": manifest["status"], "shards": states}), flush=True)

    export_weights(parameters, namespace, args.manifest, on_update=progress)
    print(f"Completed weight manifest: {args.manifest}")


if __name__ == "__main__":
    main()
