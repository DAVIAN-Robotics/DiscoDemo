# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finger body groups read by gripper contact predicates.

IsaacLab's ContactSensor supports filtered contacts (``force_matrix_w``) only one-to-many, i.e. the sensor prim
must resolve to a single body per env. The gripper therefore has one sensor per finger, and predicates read them
as a group. "Detached" is true only when every finger is out of contact.

This module does not import isaaclab so it can be tested without the Isaac app.
"""

from typing import Any, Sequence

import torch

# Predicate gripper name -> contact sensor names of its fingers (sensors are named ``{key}__{obj}``).
GRIPPER_FINGER_GROUPS: dict[str, tuple[str, ...]] = {
    "gripper": ("gripper", "gripper_right"),
}


def resolve_gripper_bodies(gripper_name: str | Sequence[str]) -> list[str]:
    """Expand a predicate ``gripper_name`` into a list of contact sensor names.

    A str is always expanded to its finger group; an explicit list/tuple is used as is (e.g. to measure a single
    finger).

    Parameters
    ----------
    gripper_name : str or Sequence[str]
        A key of :data:`GRIPPER_FINGER_GROUPS`, or explicit sensor names.

    Returns
    -------
    list of str
        Contact sensor names (at least one).

    Raises
    ------
    KeyError
        If a str is not in :data:`GRIPPER_FINGER_GROUPS`.
    ValueError
        If the sequence is empty.
    """
    if isinstance(gripper_name, str):
        if gripper_name not in GRIPPER_FINGER_GROUPS:
            raise KeyError(
                f"unknown gripper_name {gripper_name!r}; registered groups: "
                f"{sorted(GRIPPER_FINGER_GROUPS)}. To measure a single finger pass a list "
                f"(e.g. [{gripper_name!r}])."
            )
        return list(GRIPPER_FINGER_GROUPS[gripper_name])
    bodies = list(gripper_name)
    if not bodies:
        raise ValueError("gripper_name is an empty sequence; no body to measure contact on.")
    return bodies


def detached_from_contacts(contacts: Sequence[Any]) -> Any:
    """Detached = no finger in contact.

    Parameters
    ----------
    contacts : Sequence
        Per-finger contact flags: bools and/or ``(N,)`` bool tensors.

    Returns
    -------
    bool or torch.Tensor
        A tensor if any input is a tensor, otherwise a bool.

    Raises
    ------
    ValueError
        If ``contacts`` is empty.
    """
    items = list(contacts)
    if not items:
        raise ValueError("contacts is empty; cannot decide detachment without finger contacts.")
    result: Any = None
    for contact in items:
        # Same rule as predicate_logic's _and/_not, duplicated to avoid importing isaaclab.
        not_contact = ~contact if isinstance(contact, torch.Tensor) else not contact
        if result is None:
            result = not_contact
        elif isinstance(result, torch.Tensor) or isinstance(not_contact, torch.Tensor):
            result = result & not_contact
        else:
            result = result and not_contact
    return result


def any_contact(contacts: Sequence[Any]) -> Any:
    """Any finger in contact; the exact complement of :func:`detached_from_contacts`.

    Parameters
    ----------
    contacts : Sequence
        Per-finger contact flags: bools and/or ``(N,)`` bool tensors.

    Returns
    -------
    bool or torch.Tensor
        A tensor if any input is a tensor, otherwise a bool.

    Raises
    ------
    ValueError
        If ``contacts`` is empty.
    """
    detached = detached_from_contacts(contacts)
    return ~detached if isinstance(detached, torch.Tensor) else not detached
