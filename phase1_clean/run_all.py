from __future__ import annotations

import argparse
import subprocess
import sys

from core import PROJECT_CONFIG


def parse_args():
    p = argparse.ArgumentParser(description="Run the clean Phase-1 novelty across all baseline projects.")
    p.add_argument("--shuffle", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-commits", type=int, default=1000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument(
        "--projects",
        nargs="+",
        default=list(PROJECT_CONFIG),
        choices=sorted(PROJECT_CONFIG),
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining projects if one project fails.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    failures = []

    for project in args.projects:
        cmd = [
            sys.executable,
            "phase1_clean/run.py",
            "--project",
            project,
            "--shuffle",
            str(args.shuffle),
            "--device",
            args.device,
            "--max-commits",
            str(args.max_commits),
            "--max-epochs",
            str(args.max_epochs),
            "--threshold-step",
            str(args.threshold_step),
        ]
        print("\n" + "=" * 80)
        print("Running:", " ".join(cmd))
        print("=" * 80)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append(project)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        print("\nFailed projects:", ", ".join(failures))
        raise SystemExit(1)

    print("\nAll requested projects completed.")


if __name__ == "__main__":
    main()
