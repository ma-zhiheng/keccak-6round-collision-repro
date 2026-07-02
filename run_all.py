"""Run the currently implemented reproduction checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n== {script} ==", flush=True)
    subprocess.run([sys.executable, script], cwd=ROOT, check=True)


def main() -> None:
    run("check_observations.py")
    run("demo_sbox_constraints.py")
    run("state_lift.py")
    run("equation_assembler.py")
    run("trail_parser.py")
    run("trail_verify.py")
    run("trail_data.py")
    run("beta_selector.py")
    run("beta0_selector.py")
    run("connector_equations.py")
    run("incremental_connector.py")
    run("connector_runner.py")
    run("sample_connector.py")
    run("demo_nonzero_connector.py")
    run("reproduce_core2_connector.py")


if __name__ == "__main__":
    main()
