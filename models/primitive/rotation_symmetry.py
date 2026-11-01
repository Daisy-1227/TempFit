from __future__ import annotations

import torch


# Conservative full-template geometry audit (mean <= 1e-3, q95 <= 2.5e-3).
# All remaining templates have order one.  See
# results/industry_recon_protocol/rotation_symmetry_audit.json.
AUDITED_ROTATION_SYMMETRY_ORDERS = {
    0: 2,
    6: 2,
    15: 2,
    43: 2,
    59: 2,
    61: 2,
    69: 2,
    71: 2,
    75: 2,
}


def symmetry_order_table(num_primitives: int, *, audited: bool) -> torch.Tensor:
    """Return an integer symmetry-order table, defaulting every entry to one."""

    if num_primitives <= 0:
        raise ValueError("num_primitives must be positive")
    orders = torch.ones(num_primitives, dtype=torch.long)
    if audited:
        for primitive_id, order in AUDITED_ROTATION_SYMMETRY_ORDERS.items():
            if primitive_id >= num_primitives:
                raise ValueError(
                    f"audited primitive {primitive_id} is outside library size "
                    f"{num_primitives}"
                )
            orders[primitive_id] = int(order)
    return orders


def gather_symmetry_orders(
    primitive_id: torch.Tensor,
    order_table: torch.Tensor,
) -> torch.Tensor:
    """Gather orders with padding-safe primitive indices."""

    safe_id = primitive_id.long().clamp(0, order_table.numel() - 1)
    return order_table.to(primitive_id.device)[safe_id]


def raw_angle_to_phase(
    angle_degrees: torch.Tensor,
    primitive_id: torch.Tensor,
    order_table: torch.Tensor,
) -> torch.Tensor:
    """Map an absolute angle to the shared 360-degree symmetry phase."""

    orders = gather_symmetry_orders(primitive_id, order_table).to(angle_degrees.dtype)
    return torch.remainder(angle_degrees * orders, 360.0)


def phase_to_canonical_angle(
    phase_degrees: torch.Tensor,
    primitive_id: torch.Tensor,
    order_table: torch.Tensor,
) -> torch.Tensor:
    """Decode shared phase to the physical canonical angle in ``[0, 360/N)``."""

    orders = gather_symmetry_orders(primitive_id, order_table).to(phase_degrees.dtype)
    while orders.ndim < phase_degrees.ndim:
        orders = orders.unsqueeze(-1)
    return torch.remainder(phase_degrees, 360.0) / orders


def canonicalize_raw_angle(
    angle_degrees: torch.Tensor,
    primitive_id: torch.Tensor,
    order_table: torch.Tensor,
) -> torch.Tensor:
    """Canonicalize a raw physical angle using the primitive's basic period."""

    return phase_to_canonical_angle(
        raw_angle_to_phase(angle_degrees, primitive_id, order_table),
        primitive_id,
        order_table,
    )
