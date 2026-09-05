from __future__ import annotations

import hashlib
import random

from models import Ramo


def assign_branch(username: str) -> Ramo:
    cleaned = username.strip().upper()
    if cleaned.startswith("J_DV_"):
        return Ramo.DV
    if cleaned.startswith("J_IC_"):
        return Ramo.IC

    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return Ramo.DV if int(digest[:2], 16) % 2 == 0 else Ramo.IC


def build_fixed_case_order(
    case_ids: list[str], username: str, branch: Ramo
) -> list[str]:
    seed_source = f"{username.strip()}|{branch.value}"
    seed_hash = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
    rng = random.Random(int(seed_hash[:16], 16))
    ordered = list(case_ids)
    rng.shuffle(ordered)
    return ordered
