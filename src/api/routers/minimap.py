"""Map positions, zones and radar geometry."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.deps import (
    _db,
)
from src.callouts import (
    game_to_pixel,
    get_all_zones_pixel,
    get_callout,
    get_zone_center,
    is_map_supported,
)
from src.database import get_enriched_json_for_map
from src.domain.blobs import decode_dict

router = APIRouter()


@router.post("/api/minimap/zones")
def resolve_minimap_zones(body: dict):
    """Convert callout names to pixel coordinates for highlighting.

    Body: {"map_name": "de_mirage", "callouts": ["B Apartments", "A Ramp"]}
    """
    map_name = body.get("map_name", "")
    callouts = body.get("callouts", [])

    if not is_map_supported(map_name):
        raise HTTPException(status_code=400, detail=f"Map {map_name} not supported")

    zones = []
    for name in callouts:
        center = get_zone_center(map_name, name)
        if center:
            zones.append({"callout": name, "px": center[0], "py": center[1]})
    return {"map_name": map_name, "zones": zones}

@router.get("/api/minimap/{map_name}/schematic")
def get_minimap_schematic(map_name: str):
    """Return all zone rectangles in pixel-space for the schematic renderer."""
    rects = get_all_zones_pixel(map_name)
    if rects is None:
        raise HTTPException(status_code=400, detail=f"Map {map_name} not supported")

    return {"map_name": map_name, "zones": rects}

@router.get("/api/minimap/{map_name}/debug-positions")
def get_debug_positions(map_name: str):
    """Return every event position from all matches on this map.

    Used for visual diagnostics: plot all dots on the radar to verify
    that the game-to-pixel transform is correct.
    """

    if not is_map_supported(map_name):
        raise HTTPException(status_code=400, detail=f"Map {map_name} not supported")

    conn = _db()
    try:
        enriched_rows = get_enriched_json_for_map(conn, map_name)
    finally:
        conn.close()

    points: list[dict] = []
    for ej_raw in enriched_rows:
        data = decode_dict(ej_raw)
        for k in data.get("kills_detail", []):
            for role in ("attacker", "victim"):
                xy = k.get(f"{role}_xy")
                if xy:
                    px = game_to_pixel(map_name, xy[0], xy[1])
                    if px:
                        points.append({
                            "gx": xy[0], "gy": xy[1],
                            "px": px[0], "py": px[1],
                            "zone": get_callout(map_name, xy[0], xy[1]),
                            "role": role,
                        })
        dd = data.get("death_detail")
        if dd:
            for role in ("victim", "killer"):
                xy = dd.get(f"{role}_xy")
                if xy:
                    px = game_to_pixel(map_name, xy[0], xy[1])
                    if px:
                        points.append({
                            "gx": xy[0], "gy": xy[1],
                            "px": px[0], "py": px[1],
                            "zone": get_callout(map_name, xy[0], xy[1]),
                            "role": role,
                        })
    return {"map_name": map_name, "count": len(points), "positions": points}
