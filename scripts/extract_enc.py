#!/usr/bin/env python
"""Extract ENC data from all four zip archives and create EncLayer JSON files.

Usage:
    python scripts/extract_enc.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.data.s57_parser import build_enc_layer_from_zip
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import ensure_dir, save_json


# Waterway reference points and bounding boxes
WATERWAYS = [
    {
        "waterway_id": "puget_sound",
        "zip_path": "data/raw/enc/Strait of Juan de Fuca-Puget Sound-HaroRosario Strait/WA_ENCs.zip",
        "ref_lon": -122.75,
        "ref_lat": 48.15,
        "bounds": (-125.5, -122.0, 47.0, 49.5),  # lon_min, lon_max, lat_min, lat_max
    },
    {
        "waterway_id": "new_york_harbor",
        "zip_path": "data/raw/enc/New York Harbor-Kill Van Kull-East River/NY_ENCs.zip",
        "ref_lon": -74.02,
        "ref_lat": 40.67,
        "bounds": (-74.3, -73.6, 40.4, 40.95),
    },
    {
        "waterway_id": "new_york_harbor_nj",
        "zip_path": "data/raw/enc/New York Harbor-Kill Van Kull-East River/NJ_ENCs.zip",
        "ref_lon": -74.05,
        "ref_lat": 40.68,
        "bounds": (-74.3, -73.85, 40.4, 40.85),
    },
    {
        "waterway_id": "san_francisco_bay",
        "zip_path": "data/raw/enc/San Francisco Bay-Golden Gate/CA_ENCs.zip",
        "ref_lon": -122.45,
        "ref_lat": 37.80,
        "bounds": (-122.65, -122.1, 37.55, 38.05),
    },
]


def enc_layer_to_dict(layer) -> dict:
    """Convert EncLayer to serializable dict."""
    result = {
        "layer_name": layer.layer_name,
        "waterway_id": layer.waterway_id,
        "source": layer.source,
        "depth_min": layer.depth_min,
        "depth_max": layer.depth_max,
        "buoy_positions": layer.buoy_positions,
        "beacon_positions": layer.beacon_positions,
        "tss_lanes": layer.tss_lanes,
        "separation_zones": layer.separation_zones,
        "precautionary_areas": layer.precautionary_areas,
        "atba_zones": layer.atba_zones,
        "inshore_traffic_zones": layer.inshore_traffic_zones,
        "recommended_routes": layer.recommended_routes,
        "bridge_piers": layer.bridge_piers,
        "channel_boundaries": layer.channel_boundaries,
        "fairway_boundaries": layer.fairway_boundaries,
        "metadata": layer.metadata,
    }
    # Convert land_polygons to coordinate lists for JSON serialization
    land_coords = []
    for poly in layer.land_polygons:
        if hasattr(poly, "exterior"):
            land_coords.append(list(poly.exterior.coords))
        elif isinstance(poly, (list, tuple)):
            land_coords.append(list(poly))
    result["land_polygons"] = land_coords
    # depth_grid might be a number or callable
    if callable(layer.depth_grid):
        result["depth_grid"] = None
    else:
        result["depth_grid"] = layer.depth_grid
    return result


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extract_base = os.path.join(base_dir, "data", "processed", "enc")
    ensure_dir(extract_base)

    output_dir = os.path.join(base_dir, "data", "processed")

    for ww in WATERWAYS:
        waterway_id = ww["waterway_id"]
        zip_path = os.path.join(base_dir, ww["zip_path"])
        extract_dir = os.path.join(extract_base, waterway_id)

        if not os.path.exists(zip_path):
            print(f"SKIP {waterway_id}: zip not found at {zip_path}")
            continue

        print(f"Extracting ENC for {waterway_id}...")
        bounds = ww.get("bounds", (-180, 180, -90, 90))
        try:
            layer = build_enc_layer_from_zip(
                zip_path=zip_path,
                waterway_id=waterway_id,
                reference_lon=ww["ref_lon"],
                reference_lat=ww["ref_lat"],
                extract_dir=extract_dir,
                lon_min=bounds[0],
                lon_max=bounds[1],
                lat_min=bounds[2],
                lat_max=bounds[3],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Save EncLayer as JSON
        json_path = os.path.join(output_dir, f"enc_layer_{waterway_id}.json")
        layer_dict = enc_layer_to_dict(layer)
        save_json(layer_dict, json_path)
        print(f"  Saved to {json_path}")
        print(f"  Features: buoys={len(layer.buoy_positions)}, "
              f"beacons={len(layer.beacon_positions)}, "
              f"tss_lanes={len(layer.tss_lanes)}, "
              f"land_polygons={len(layer.land_polygons)}, "
              f"depth=[{layer.depth_min}, {layer.depth_max}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
