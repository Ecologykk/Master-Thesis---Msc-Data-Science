"""CLI entry point for regenerating EDA figures without the notebook.

Usage:
    python -m src.eda.run --case-type {dv,boc}
"""

from __future__ import annotations

import argparse

from src.eda.eda import run_case_eda


def main() -> None:
    """Parse CLI arguments and regenerate the EDA figure set for the requested case type."""
    parser = argparse.ArgumentParser(
        description="Regenerate EDA figures (class distribution, temporal drift, word/n-gram "
        "frequency, tribunal and judge-gender breakdowns) for one case type into eda_viz/."
    )
    parser.add_argument("--case-type", choices=["dv", "boc"], required=True)
    args = parser.parse_args()

    run_case_eda(case_type=args.case_type)


if __name__ == "__main__":
    main()
