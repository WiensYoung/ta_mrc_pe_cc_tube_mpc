#!/usr/bin/env python
"""Preprocess AIS data: load, clean, interpolate, convert to local coordinates.

Supports both standard AIS column names and NOAA bulk AIS format.

Usage:
    # Process a single file:
    python scripts/preprocess_ais.py --input data/raw/ais/waterway.csv --output data/processed/

    # Process all waterways in batch (default):
    python scripts/preprocess_ais.py --all
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.data.ais_preprocess import (
    convert_ais_to_local,
    interpolate_track,
    normalize_noaa_columns,
    preprocess_ais_chunked,
    remove_outliers_speed,
)
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import ensure_dir

# Predefined waterways with reference coordinates
WATERWAY_CONFIGS = [
    {
        "name": "juan_de_fuca_puget_sound",
        "input": "data/raw/ais/juan_de_fuca_puget_sound_2025.csv",
        "output": "data/processed/ais_juan_de_fuca_puget_sound.csv",
        "lon_ref": -122.75,
        "lat_ref": 48.15,
    },
    {
        "name": "new_york_harbor",
        "input": "data/raw/ais/new_york_harbor_2025.csv",
        "output": "data/processed/ais_new_york_harbor.csv",
        "lon_ref": -74.02,
        "lat_ref": 40.67,
    },
    {
        "name": "san_francisco_bay",
        "input": "data/raw/ais/san_francisco_bay_golden_gate_2025.csv",
        "output": "data/processed/ais_san_francisco_bay.csv",
        "lon_ref": -122.45,
        "lat_ref": 37.80,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Preprocess AIS data.")
    parser.add_argument("--input", help="Input AIS CSV file (single file mode).")
    parser.add_argument("--output", help="Output directory (single file mode) or output path.")
    parser.add_argument("--lon-ref", type=float, default=0.0, help="Reference longitude.")
    parser.add_argument("--lat-ref", type=float, default=0.0, help="Reference latitude.")
    parser.add_argument("--all", action="store_true", help="Process all predefined waterways.")
    parser.add_argument("--chunksize", type=int, default=1_000_000,
                        help="Rows per chunk (default: 1,000,000).")
    parser.add_argument("--no-column-map", action="store_true",
                        help="Skip NOAA column mapping (use if CSV already has standard columns).")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.all:
        for ww in WATERWAY_CONFIGS:
            input_path = os.path.join(base_dir, ww["input"])
            output_path = os.path.join(base_dir, ww["output"])

            if not os.path.exists(input_path):
                print(f"SKIP {ww['name']}: input not found at {input_path}")
                continue

            print(f"\n{'='*60}")
            print(f"Processing {ww['name']}...")
            print(f"  Input:  {input_path}")
            print(f"  Output: {output_path}")
            print(f"  Ref:    lon={ww['lon_ref']}, lat={ww['lat_ref']}")
            print(f"{'='*60}")

            preprocess_ais_chunked(
                input_path=input_path,
                output_path=output_path,
                lon_ref=ww["lon_ref"],
                lat_ref=ww["lat_ref"],
                chunksize=args.chunksize,
            )
        print("\nAll waterways processed.")
    else:
        if not args.input:
            parser.error("--input is required (or use --all for batch mode)")
        input_path = os.path.join(base_dir, args.input) if not os.path.isabs(args.input) else args.input
        output_dir = os.path.join(base_dir, args.output) if args.output else os.path.join(base_dir, "data/processed")

        if not os.path.exists(input_path):
            print(f"Error: input file not found: {input_path}")
            sys.exit(1)

        if args.no_column_map:
            # Original pipeline
            import pandas as pd
            df = pd.read_csv(input_path)  # use resolved absolute path, not raw args.input
            print(f"Loaded {len(df)} AIS records.")
            df = remove_outliers_speed(df)
            df = interpolate_track(df)
            df = convert_ais_to_local(df, args.lon_ref, args.lat_ref)
            ensure_dir(output_dir)
            output_path = os.path.join(output_dir, "processed_ais.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} processed records to {output_path}")
        else:
            # Use chunked pipeline with NOAA column mapping
            ensure_dir(output_dir)
            output_path = os.path.join(output_dir,
                                       os.path.basename(input_path).replace(".csv", "_processed.csv"))
            preprocess_ais_chunked(
                input_path=input_path,
                output_path=output_path,
                lon_ref=args.lon_ref,
                lat_ref=args.lat_ref,
                chunksize=args.chunksize,
            )


if __name__ == "__main__":
    main()
