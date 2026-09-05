from __future__ import annotations


def compute_progress(
    case_ids: list[str],
    responses: dict[str, dict[str, str]],
) -> tuple[int, int, float]:
    total = len(case_ids)
    if total == 0:
        return 0, 0, 0.0

    answered = 0
    for case_id in case_ids:
        response = responses.get(case_id, {})
        has_decision = bool(str(response.get("decisao", "")).strip())
        has_confidence = bool(str(response.get("confianca", "")).strip())
        if has_decision and has_confidence:
            answered += 1

    progress = answered / total
    return answered, total, progress


def first_incomplete_index(
    ordered_case_ids: list[str],
    responses: dict[str, dict[str, str]],
) -> int:
    for index, case_id in enumerate(ordered_case_ids):
        response = responses.get(case_id, {})
        has_decision = bool(str(response.get("decisao", "")).strip())
        has_confidence = bool(str(response.get("confianca", "")).strip())
        if not (has_decision and has_confidence):
            return index
    return 0
