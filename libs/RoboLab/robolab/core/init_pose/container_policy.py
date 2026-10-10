"""Which scene objects the initial-state randomization moves."""

from __future__ import annotations

# Sentinel ``asset_cfg``: decide the randomized assets at runtime from the scene, so every movable
# object is randomized, not only those named in the success termination.
ALL_MOVABLE = "__all_movable__"


def movable_asset_names(all_names: list[str], table_name: str = "table") -> list[str]:
    """Scene rigid objects to randomize: all except the table."""
    return [n for n in all_names if n != table_name]
