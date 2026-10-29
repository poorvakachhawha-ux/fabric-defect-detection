"""Download the trained Roboflow dataset version in folder format into dataset/."""
import argparse
import os

from roboflow import Roboflow

from common import PROJECT, VERSION, WORKSPACE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--version", type=int, default=VERSION)
    parser.add_argument("--location", default="dataset")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY first")

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(args.workspace).project(args.project).version(args.version)
    version.download("folder", location=args.location, overwrite=True)
    print(f"Dataset written to {args.location}/ (train, valid, test)")


if __name__ == "__main__":
    main()
