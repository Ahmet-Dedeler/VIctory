#!/usr/bin/env python3
"""Run the descriptor allocator benchmark on Apple Silicon and Rosetta."""

from pathlib import Path
import argparse
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = "10e3c562"
SOURCE = "crates/victory-gpu/src/descriptor/host.rs"
ALLOCATOR = ROOT / "crates" / "victory-gpu" / "src" / "descriptor" / "dynamic.rs"


def method(source, name):
    lines = iter(source.splitlines(keepends=True))
    prefix = f"    pub(crate) fn {name}("
    result = [next(line for line in lines if line.startswith(prefix))]
    for line in lines:
        result.append(line)
        if line == "    }\n":
            return "".join(result)
    raise ValueError(f"unterminated method: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=61, type=int)
    parser.add_argument("--arch", choices=("aarch64", "x86_64", "both"), default="both")
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:{SOURCE}"], cwd=ROOT, text=True
    )
    baseline = "impl HostHeap {\n" + method(source, "dynamic") + method(source, "release") + "}\n"
    targets = ["aarch64", "x86_64"] if args.arch == "both" else [args.arch]
    build_parent = ROOT / "target" / "dynamic-slot-bench"
    build_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=build_parent) as tmp:
        build = Path(tmp)
        (build / "baseline.rs").write_text(baseline)
        shutil.copy(HERE / "bench.rs", build)
        shutil.copy(ALLOCATOR, build)
        subprocess.run(["rustc", "--version"], cwd=build, check=True)
        for architecture in targets:
            print(f"baseline {BASE} | {architecture}-apple-darwin", flush=True)
            subprocess.run(
                [
                    "nice", "-n", "15", "rustc", "--edition=2024", "-O",
                    "-C", "codegen-units=1", "--target", f"{architecture}-apple-darwin",
                    "bench.rs", "-o", "bench",
                ],
                cwd=build,
                check=True,
            )
            # The Mach-O architecture invokes Rosetta for the x86_64 build.
            subprocess.run(
                ["nice", "-n", "15", str(build / "bench"), str(args.samples)], check=True
            )


if __name__ == "__main__":
    main()
