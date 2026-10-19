"""Command line entry points for a bounded static analysis pass."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

from .core import Budget, analyze
from .inputs import load_fixture, load_manifest, pythia_ffn
from .render import render_html


def write_analysis(package, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.mgl.json").write_text(
        json.dumps(package, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "index.html").write_text(render_html(package), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Statically disassemble an explicit neural FFN")
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="Import a source from the published neural XOR fixture")
    fixture.add_argument("input", type=Path)
    fixture.add_argument("--source", default="src-xor")
    manifest = sub.add_parser("analyze", help="Import a pinn.ffn-input/1 manifest of .npy files")
    manifest.add_argument("input", type=Path)
    pythia = sub.add_parser("pythia", help="Import one GPT-NeoX FFN and its pre-LayerNorm")
    pythia.add_argument("--checkpoint", type=Path, required=True)
    pythia.add_argument("--config", type=Path, required=True)
    pythia.add_argument("--layer", type=int, required=True)
    for child in (fixture, manifest, pythia):
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--max-parameter-bytes", type=int, default=64 * 1024 * 1024)
        child.add_argument("--max-contributions", type=int, default=4096)
        child.add_argument("--block-rows", type=int, default=128)
        child.add_argument("--sample-width", type=int, default=8)
    args = parser.parse_args(argv)
    budget = Budget(args.max_parameter_bytes, args.max_contributions, args.block_rows, args.sample_width)
    try:
        if args.command == "fixture":
            context = nullcontext(load_fixture(args.input, args.source))
        elif args.command == "analyze":
            context = nullcontext(load_manifest(args.input))
        else:
            context = pythia_ffn(args.checkpoint, args.config, args.layer)
        with context as ffn:
            package = analyze(ffn, budget)
        write_analysis(package, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    run = package["pass_runs"][0]
    print(json.dumps({"status": run["status"], "coverage": run["coverage"],
                      "cost": run["cost"], "output": str(args.output)}, indent=2))

