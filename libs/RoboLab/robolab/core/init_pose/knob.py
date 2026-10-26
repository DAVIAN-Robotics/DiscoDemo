"""``init_pose`` knob: one string selects the initial-state distribution.

Accepted forms:

* ``box:...``: absolute box (see ``robolab.core.init_pose.box``);
* ``pregrasp:...``: start with an object in hand (see ``robolab.core.init_pose.pregrasp``).
"""

from __future__ import annotations

from robolab.core.init_pose.box import BoxRange, parse_box
from robolab.core.init_pose.pregrasp import PregraspRange, parse_pregrasp


def parse_init_pose(value: str) -> BoxRange | PregraspRange:
    """Parse an ``init_pose`` string; raise if no form matches."""
    text = str(value).strip()
    box = parse_box(text)
    if box is not None:
        return box
    pregrasp = parse_pregrasp(text)
    if pregrasp is not None:
        return pregrasp
    raise ValueError(
        f"cannot parse init_pose={value!r}. Expected 'box:x_lo,x_hi,y_lo,y_hi,yaw_half_deg,min_dist_m' "
        "or 'pregrasp:xy_half_m,yaw_half_deg'."
    )
