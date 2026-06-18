#!/usr/bin/env python
"""Generate analysis report from experiment results.

Usage:
    python scripts/make_report.py --input results/raw/core_results.csv --output_dir results/processed
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.experiments.analyze_results import analyze_results


def main():
    parser = argparse.ArgumentParser(description="Generate analysis report from experiment results.")
    parser.add_argument("--input", required=True, help="Input CSV file with episode results.")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed results.")
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    input_path = args.input if os.path.isabs(args.input) else os.path.join(project_root, args.input)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(project_root, args.output_dir)

    summary = analyze_results(
        results_csv=input_path,
        output_dir=output_dir,
        generate_plots=True,
    )

    print(f"\nAnalysis complete.")
    print(f"Methods evaluated: {list(summary.keys())}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
