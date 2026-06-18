"""ENC (Electronic Navigational Chart) layer abstractions.

Provides a standard interface for both real ENC data and synthetic substitutes.
"""

import json
from dataclasses import dataclass, field
from numbers import Number
from typing import Optional

try:
    from shapely.geometry import Point as ShapelyPoint, Polygon
    _HAS_SHAPELY = True
except ImportError:
    ShapelyPoint = None
    Polygon = None
    _HAS_SHAPELY = False


def _make_point(x: float, y: float):
    """Create a point for polygon containment tests, with or without shapely."""
    if _HAS_SHAPELY:
        return ShapelyPoint(x, y)
    return (x, y)


@dataclass
class EncLayer:
    """Generic ENC layer representation.

    Supports both real ENC-derived data and synthetic substitutes
    through the same interface. When real data is unavailable,
    synthetic data populates the same fields.
    """
    layer_name: str = ""
    # Depth/bathymetry
    depth_grid: Optional[object] = None     # 2D array or shapely geometry
    depth_min: float = 0.0
    depth_max: float = 0.0
    # Land / shoreline
    land_polygons: list = field(default_factory=list)
    # Navigation aids
    buoy_positions: list = field(default_factory=list)
    beacon_positions: list = field(default_factory=list)
    # Traffic zones
    tss_lanes: list = field(default_factory=list)
    separation_zones: list = field(default_factory=list)
    precautionary_areas: list = field(default_factory=list)
    atba_zones: list = field(default_factory=list)
    inshore_traffic_zones: list = field(default_factory=list)
    recommended_routes: list = field(default_factory=list)
    # Structures
    bridge_piers: list = field(default_factory=list)
    # Channel
    channel_boundaries: list = field(default_factory=list)
    fairway_boundaries: list = field(default_factory=list)
    # Metadata
    waterway_id: str = ""
    source: str = "synthetic"  # "enc" or "synthetic"
    # ── Serialization schema version ─────────────────────────────────────
    SCHEMA_VERSION: str = "1.0"  # class-level default
    metadata: dict = field(default_factory=dict)

    # ── Serialization ────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for multiprocessing / checkpointing.

        Shapely Polygon objects are serialized as ``[exterior_coords, ...holes]``
        where each coord list is ``[[x, y], ...]``.  Non-Shapely entries
        are passed through unchanged.
        """
        return {
            "schema_version": "enc_layer_1.0",
            "layer_name": self.layer_name,
            "waterway_id": self.waterway_id,
            "source": self.source,
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
            "depth_grid": self.depth_grid if isinstance(self.depth_grid, (int, float, Number, type(None))) else None,
            "buoy_positions": self.buoy_positions,
            "beacon_positions": self.beacon_positions,
            "recommended_routes": self.recommended_routes,
            "bridge_piers": self.bridge_piers,
            "channel_boundaries": self.channel_boundaries,
            "fairway_boundaries": self.fairway_boundaries,
            "land_polygons": _serialize_polygons(self.land_polygons),
            "tss_lanes": _serialize_polygons(self.tss_lanes),
            "separation_zones": _serialize_polygons(self.separation_zones),
            "precautionary_areas": _serialize_polygons(self.precautionary_areas),
            "atba_zones": _serialize_polygons(self.atba_zones),
            "inshore_traffic_zones": _serialize_polygons(self.inshore_traffic_zones),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EncLayer":
        """Reconstruct an EncLayer from a dict (inverse of ``to_dict()``).

        Inverse of ``to_dict()``.  Rebuilds Shapely Polygon objects
        from serialised coordinate lists so ``is_navigable()`` works.
        """
        return cls(
            layer_name=data.get("layer_name", ""),
            waterway_id=data.get("waterway_id", ""),
            source=data.get("source", "synthetic"),
            depth_min=data.get("depth_min", 0.0),
            depth_max=data.get("depth_max", 0.0),
            depth_grid=data.get("depth_grid"),
            buoy_positions=data.get("buoy_positions", []),
            beacon_positions=data.get("beacon_positions", []),
            recommended_routes=data.get("recommended_routes", []),
            bridge_piers=data.get("bridge_piers", []),
            channel_boundaries=data.get("channel_boundaries", []),
            fairway_boundaries=data.get("fairway_boundaries", []),
            land_polygons=_rebuild_polygons(data.get("land_polygons", [])),
            tss_lanes=_rebuild_polygons(data.get("tss_lanes", [])),
            separation_zones=_rebuild_polygons(data.get("separation_zones", [])),
            precautionary_areas=_rebuild_polygons(data.get("precautionary_areas", [])),
            atba_zones=_rebuild_polygons(data.get("atba_zones", [])),
            inshore_traffic_zones=_rebuild_polygons(data.get("inshore_traffic_zones", [])),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: str) -> "EncLayer":
        """Load an EncLayer from a processed JSON file.

        Rebuilds Shapely geometry objects from serialized coordinate lists,
        ensuring is_navigable() works correctly on loaded data.

        Args:
            path: Path to the enc_layer JSON file.

        Returns:
            EncLayer instance with Shapely geometry objects.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Rebuild Shapely Polygon objects from serialized coordinate lists.
        # JSON cannot natively store Shapely geometries, so extract_enc.py
        # serializes them as [exterior_coords, ...holes]. We rebuild them here.
        land_polygons_raw = data.get("land_polygons", [])
        land_polygons = _rebuild_polygons(land_polygons_raw)

        return cls(
            layer_name=data.get("layer_name", ""),
            waterway_id=data.get("waterway_id", ""),
            source=data.get("source", "enc"),
            depth_min=data.get("depth_min", 0.0),
            depth_max=data.get("depth_max", 0.0),
            depth_grid=data.get("depth_grid"),
            buoy_positions=data.get("buoy_positions", []),
            beacon_positions=data.get("beacon_positions", []),
            tss_lanes=_rebuild_polygons(data.get("tss_lanes", [])),
            separation_zones=_rebuild_polygons(data.get("separation_zones", [])),
            precautionary_areas=_rebuild_polygons(data.get("precautionary_areas", [])),
            atba_zones=_rebuild_polygons(data.get("atba_zones", [])),
            inshore_traffic_zones=_rebuild_polygons(data.get("inshore_traffic_zones", [])),
            recommended_routes=data.get("recommended_routes", []),
            bridge_piers=data.get("bridge_piers", []),
            channel_boundaries=data.get("channel_boundaries", []),
            fairway_boundaries=data.get("fairway_boundaries", []),
            land_polygons=land_polygons,
            metadata=data.get("metadata", {}),
        )

    def get_depth_at(self, x: float, y: float) -> Optional[float]:
        """Return water depth at a given position, if available."""
        if self.depth_grid is None:
            return None
        # Simplified: assume depth_grid is a callable or constant
        if callable(self.depth_grid):
            return float(self.depth_grid(x, y))
        if isinstance(self.depth_grid, (int, float, Number)):
            return float(self.depth_grid)
        return None

    def is_navigable(self, x: float, y: float, min_depth: float = 0.0) -> bool:
        """Check if a position is navigable (not land, deep enough)."""
        depth = self.get_depth_at(x, y)
        if depth is not None and depth < min_depth:
            return False
        # Check land polygons
        for poly in self.land_polygons:
            if hasattr(poly, "contains"):
                if poly.contains(_make_point(x, y)):
                    return False
        return True

    def is_inland_water(self, x: float, y: float) -> bool:
        """Determine if position (x, y) falls under Inland Rules jurisdiction.

        Per 33 CFR 80, COLREGs demarcation lines separate International
        Rules waters from Inland Rules waters. This method uses ENC metadata
        and geographic features to determine which rules apply.

        For waterway_id values containing inland waterway identifiers
        (western_rivers, great_lakes, etc.), returns True.
        Otherwise falls back to checking if the position is inside
        channel_boundaries (suggesting restricted inland water).

        Args:
            x, y: World-frame position [m].

        Returns:
            True if Inland Rules (not International COLREGs) apply.
        """
        # Check waterway_id for known inland waterways
        inland_ids = ["western_rivers", "mississippi", "ohio_river",
                      "great_lakes", "kill_van_kull", "east_river",
                      "hudson_river", "chicago_sanitary_canal"]
        if any(w in self.waterway_id.lower() for w in inland_ids):
            return True

        # Check if position is within a channel boundary (restricted water)
        for boundary in self.channel_boundaries:
            if hasattr(boundary, "contains"):
                if boundary.contains(_make_point(x, y)):
                    return True
            elif isinstance(boundary, (list, tuple)) and len(boundary) >= 2:
                # Simple bounding box check
                if isinstance(boundary[0], (list, tuple)):
                    xs = [pt[0] for pt in boundary]
                    ys = [pt[1] for pt in boundary]
                    if min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
                        return True

        return False


def make_synthetic_enc(
    waterway_id: str = "default",
    depth: float = 30.0,
    channel_width: float = 500.0,
    bank_left: float = 250.0,
    bank_right: float = 250.0,
) -> EncLayer:
    """Create a synthetic ENC layer with a simple rectangular channel.

    Args:
        waterway_id: Identifier for the waterway.
        depth: Uniform water depth [m].
        channel_width: Channel width [m].
        bank_left: Distance from centerline to left bank [m].
        bank_right: Distance from centerline to right bank [m].

    Returns:
        EncLayer with synthetic geometry.
    """
    if not _HAS_SHAPELY:
        raise ImportError(
            "shapely is required for synthetic ENC generation. "
            "Install it with: pip install shapely"
        )
    # Polygon now imported at module level (line 11)

    layer = EncLayer(
        layer_name=f"synthetic_{waterway_id}",
        waterway_id=waterway_id,
        source="synthetic",
        depth_min=depth,
        depth_max=depth,
    )
    layer.depth_grid = depth  # uniform depth

    # Channel boundaries as polygons
    half_width = channel_width / 2
    # Left and right bank polygons (land)
    layer.land_polygons = [
        # Left bank
        Polygon([
            (-bank_left, -10000), (-bank_left - 500, -10000),
            (-bank_left - 500, 50000), (-bank_left, 50000),
        ]),
        # Right bank
        Polygon([
            (bank_right, -10000), (bank_right + 500, -10000),
            (bank_right + 500, 50000), (bank_right, 50000),
        ]),
    ]
    layer.channel_boundaries = [(-half_width, half_width)]  # (left, right)
    return layer


def _serialize_polygons(poly_list: list) -> list:
    """Serialize a list of Shapely Polygon objects to JSON-safe coord lists."""
    result = []
    for p in poly_list:
        if hasattr(p, "exterior"):
            # Shapely Polygon → [exterior_coords, ...holes]
            ext = [[float(c[0]), float(c[1])] for c in p.exterior.coords]
            holes = [[[float(c[0]), float(c[1])] for c in h.coords] for h in p.interiors]
            result.append([ext] + holes if holes else [ext])
        elif isinstance(p, (list, tuple)):
            result.append(list(p))
        else:
            result.append(p)
    return result


def _rebuild_polygons(raw_list: list) -> list:
    """Rebuild Shapely Polygon objects from serialized coordinate lists.

    JSON serialization stores polygons as [exterior_coords, ...holes].
    Non-polygon entries (dicts, strings) are passed through unchanged.
    Entries that are already Shapely objects (have a 'contains' method)
    are also passed through.
    """
    if not raw_list:
        return raw_list

    result = []
    for item in raw_list:
        if hasattr(item, "contains"):
            # Already a Shapely geometry object
            result.append(item)
        elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
            # Looks like [exterior_coords, ...holes] where exterior_coords
            # is a list of (x, y) pairs
            try:
                if _HAS_SHAPELY:
                    poly = Polygon(shell=item[0], holes=item[1:] if len(item) > 1 else None)
                    result.append(poly)
                else:
                    result.append(item)
            except Exception:
                result.append(item)
        else:
            result.append(item)
    return result
