"""Small, model-agnostic FedAvg/FedProx primitives for the T10-T12 simulation."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


def fedavg(global_state: Mapping[str, torch.Tensor],
           client_states: Sequence[Mapping[str, torch.Tensor]],
           sample_counts: Sequence[int]) -> dict[str, torch.Tensor]:
    """Aggregate client states weighted by their number of training images."""
    if not client_states or len(client_states) != len(sample_counts):
        raise ValueError("Provide one positive sample count for every client state")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("Every participating client must have at least one training image")
    keys = set(global_state)
    if any(set(state) != keys for state in client_states):
        raise ValueError("Global and client model states must have identical keys")
    total = sum(sample_counts)
    weights = [count / total for count in sample_counts]
    largest_client = max(range(len(sample_counts)), key=sample_counts.__getitem__)
    result: dict[str, torch.Tensor] = {}
    for key, original in global_state.items():
        values = [state[key].detach().cpu() for state in client_states]
        if any(value.shape != original.shape or value.dtype != original.dtype for value in values):
            raise ValueError(f"Client tensor shape or dtype differs for {key}")
        if original.is_floating_point():
            result[key] = sum(value.to(torch.float64) * weight
                              for value, weight in zip(values, weights)).to(original.dtype)
        elif original.is_complex():
            result[key] = sum(value.to(torch.complex128) * weight
                              for value, weight in zip(values, weights)).to(original.dtype)
        else:
            # Integer counters and other non-averagable buffers follow the largest client.
            result[key] = values[largest_client].clone()
    return result


def fedprox_penalty(model: torch.nn.Module,
                    global_state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return 1/2 * ||local parameters - round-start parameters||^2."""
    terms = [
        (parameter - global_state[name].to(device=parameter.device, dtype=parameter.dtype)).pow(2).sum()
        for name, parameter in model.named_parameters()
        if name in global_state
    ]
    if not terms:
        raise ValueError("No model parameters matched the global round-start state")
    return 0.5 * torch.stack(terms).sum()


def run_federated_round(
    global_state: Mapping[str, torch.Tensor],
    clients: Mapping[str, Any],
    local_update: Callable[[str, Any, Mapping[str, torch.Tensor], float],
                           tuple[Mapping[str, torch.Tensor], int, Mapping[str, float]]],
    *,
    method: str = "fedavg",
    prox_mu: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run every client locally, then aggregate by sample count.

    `local_update` owns model construction and local epochs, and must return its updated
    state, number of training examples actually used, and scalar summaries. This keeps the
    coordinator independent of each detector's implementation.
    """
    method = method.lower()
    if method not in {"fedavg", "fedprox"}:
        raise ValueError("method must be 'fedavg' or 'fedprox'")
    if not clients:
        raise ValueError("At least one client is required")
    if prox_mu < 0:
        raise ValueError("prox_mu must be nonnegative")
    coefficient = prox_mu if method == "fedprox" else 0.0
    states, counts, summaries = [], [], {}
    for client_id, client_data in clients.items():
        state, count, metrics = local_update(client_id, client_data, global_state, coefficient)
        if count <= 0:
            raise ValueError(f"Client {client_id!r} returned no training samples")
        states.append(state)
        counts.append(count)
        summaries[client_id] = {"n_train": count, **dict(metrics)}
    new_state = fedavg(global_state, states, counts)
    return new_state, {"method": method, "prox_mu": coefficient,
                       "n_clients": len(states), "n_train": sum(counts),
                       "clients": summaries}
