from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import __version__
from .predict import predict, resource, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="SPARK: structure-to-XANES prediction")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "predict"):
        command = commands.add_parser(name, help="Run the bundled CSIRO example" if name == "demo" else "Predict your structures")
        if name == "demo":
            command.add_argument("--output", type=Path, default=Path("."), help="Create structure/ and xanes/ here (default: current directory)")
        else:
            command.add_argument("--root", type=Path, default=Path("."), help="Directory containing structure/ (default: current directory)")
            command.add_argument("--ids", nargs="+", help="Structure IDs without extensions")
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--member", type=int, help="Use one model, identified by its initialization seed; omit for eight-model uncertainty")
        command.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
        command.add_argument("--overwrite", action="store_true", help="Replace generated predictions")
        command.add_argument("--checkpoints", type=Path, help="Checkpoint directory; defaults to the bundled publication ensemble")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            root = args.output.resolve()
            example = resource("example")
            for relative in ("structure/104795.xyz", "xanes/104795_feff_reference.dat"):
                source, destination = example / relative, root / relative
                if destination.exists():
                    if sha256(source) != sha256(destination):
                        raise FileExistsError(f"Demo input differs from bundled example: {destination}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            ids = ["104795"]
        else:
            root, ids = args.root, args.ids
        predict(root, ids=ids, member=args.member, device=args.device, threads=args.threads,
                overwrite=args.overwrite, checkpoints=args.checkpoints)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"spark: {error}\n")
