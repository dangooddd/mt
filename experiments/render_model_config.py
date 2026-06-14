#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-model-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--dropout", type=float, required=True)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
config = json.loads((args.base_model_dir / "config.json").read_text())
config["args"]["dropout"] = args.dropout
(args.output_dir / "config.json").write_text(json.dumps(config, indent=4) + "\n")
shutil.copy2(args.base_model_dir / "tokenizer.json", args.output_dir / "tokenizer.json")
