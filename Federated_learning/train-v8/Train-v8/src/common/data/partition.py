"""Patient-preserving, deterministic client partitioning from a clinical covariate."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any


def partition_by_gap(rows: Sequence[Mapping[str, Any]], num_clients: int,
                     covariate: str = "gap_days") -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """Assign whole patients to ordered gap bands, balancing patient counts by quantiles.

    If a patient has multiple scans, the median of their finite gap values determines that
    patient's band and all their scans stay together. No date or raw patient identifier is
    written to the returned summary.
    """
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        patient = row.get("patient_id")
        value = row.get(covariate)
        if not isinstance(patient, str) or not patient:
            raise ValueError("Each row needs a pseudonymized patient_id")
        if value is None or isinstance(value, bool):
            raise ValueError(f"Each row needs numeric {covariate}")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{covariate} must be numeric days") from exc
        if not math.isfinite(value):
            raise ValueError(f"{covariate} must be finite")
        normalized = dict(row)
        normalized[covariate] = value
        grouped[patient].append(normalized)
    if len(grouped) < num_clients:
        raise ValueError("Need at least one patient group per client")

    ordered = []
    for patient, patient_rows in grouped.items():
        patient_gap = median(row[covariate] for row in patient_rows)
        ordered.append((patient_gap, patient, patient_rows))
    ordered.sort(key=lambda item: (item[0], item[1]))
    clients = {f"client_{i + 1}": [] for i in range(num_clients)}
    gap_ranges = {key: [] for key in clients}
    for rank, (gap, _patient, patient_rows) in enumerate(ordered):
        index = min(num_clients - 1, rank * num_clients // len(ordered))
        client = f"client_{index + 1}"
        clients[client].extend(patient_rows)
        gap_ranges[client].append(gap)
    summary = {
        "partition_method": "patient median gap-days quantile bands; stable tie-break by pseudonym",
        "covariate": covariate,
        "n_patients": len(grouped),
        "clients": {
            client: {"n_patients": len({row["patient_id"] for row in client_rows}),
                     "n_images": len(client_rows),
                     "gap_min": min(gap_ranges[client]),
                     "gap_median": median(gap_ranges[client]),
                     "gap_max": max(gap_ranges[client])}
            for client, client_rows in clients.items()
        },
    }
    return clients, summary
