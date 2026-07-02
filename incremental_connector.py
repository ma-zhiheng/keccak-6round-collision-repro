"""Incremental connector builder, closer to the paper's basic procedure."""

from __future__ import annotations

import random
from dataclasses import dataclass

from connector_equations import (
    SBoxTransition,
    fixed_initial_bit_equations,
    second_round_g_equations,
    substitute_second_round_equation,
)
from gf2 import GF2System
from linear_layer import (
    STATE_BITS,
    apply_l_int,
    load_or_build_matrices,
    round_constant_state,
    transpose_apply,
)
from sbox_constraints import AffineSBoxModel, all_two_dim_linearizable_models, linearizable_models_for_transition
from state_lift import SBOXES_1600, lift_model


@dataclass(frozen=True)
class IncrementalConnector:
    system: GF2System
    y_coeffs: tuple[int, ...]
    y_consts: tuple[int, ...]
    assigned_sboxes: int
    added_g_rows: int
    total_g_rows: int
    nodes: int = 0
    failed_row: int | None = None
    failed_sboxes: tuple[int, ...] = ()
    failure_reason: str = ""


@dataclass(frozen=True)
class AssignedModel:
    model: AffineSBoxModel
    added_equations: tuple[tuple[int, int], ...]
    output_masks: tuple[int, ...]
    output_constants: tuple[int, ...]


@dataclass(frozen=True)
class LiftedCandidate:
    model: AffineSBoxModel
    input_equations: tuple[tuple[int, int], ...]
    output_masks: tuple[int, ...]
    output_constants: tuple[int, ...]


@dataclass(frozen=True)
class PreparedBitRow:
    original_index: int
    z_mask: int
    rhs: int
    bits: tuple[int, ...]
    deps: tuple[int, ...]


@dataclass(frozen=True)
class ModelSearchFrame:
    row_index: int
    remaining_bits: tuple[int, ...]
    system: GF2System
    assigned: dict[int, AssignedModel]
    equation_coeff: int
    equation_const: int


def bit_to_sbox(bit: int) -> int:
    lane = bit // 64
    z = bit % 64
    y = lane // 5
    return y * 64 + z


def ymask_sboxes(mask: int) -> set[int]:
    result: set[int] = set()
    value = mask
    while value:
        low = value & -value
        bit = low.bit_length() - 1
        result.add(bit_to_sbox(bit))
        value ^= low
    return result


def bit_local_index(bit: int) -> int:
    lane = bit // 64
    return lane % 5


def model_candidates_by_sbox(beta0_alpha1: list[SBoxTransition]) -> list[list[AffineSBoxModel]]:
    by_index = {transition.sbox_index: transition for transition in beta0_alpha1}
    inactive = all_two_dim_linearizable_models()
    candidates: list[list[AffineSBoxModel]] = []
    for sbox_index in range(SBOXES_1600):
        transition = by_index.get(sbox_index)
        if transition is None or (transition.delta_in == 0 and transition.delta_out == 0):
            models = inactive
        else:
            models = linearizable_models_for_transition(transition.delta_in, transition.delta_out)
        if not models:
            raise ValueError(f"no model candidates for S-box {sbox_index}")
        candidates.append(models)
    return candidates


def assign_sbox(
    sbox_index: int,
    candidates: list[AffineSBoxModel],
    system: GF2System,
    y_coeffs: list[int],
    y_consts: list[int],
    rng: random.Random,
) -> bool:
    order = list(range(len(candidates)))
    rng.shuffle(order)
    for index in order:
        lifted = lift_model(sbox_index, candidates[index])
        if not system.is_consistent_with(lifted.input_equations):
            continue
        system.add_equations(lifted.input_equations)
        positions = [(sbox_index // 64) * 5 * 64 + x * 64 + (sbox_index % 64) for x in range(5)]
        for local_bit, out_bit in enumerate(positions):
            y_coeffs[out_bit] = lifted.output_masks[local_bit]
            y_consts[out_bit] = lifted.output_constants[local_bit]
        return True
    return False


def lifted_candidates_by_sbox(beta0_alpha1: list[SBoxTransition]) -> list[list[LiftedCandidate]]:
    raw = model_candidates_by_sbox(beta0_alpha1)
    lifted_all: list[list[LiftedCandidate]] = []
    for sbox_index, models in enumerate(raw):
        lifted_models: list[LiftedCandidate] = []
        for model in models:
            lifted = lift_model(sbox_index, model)
            lifted_models.append(
                LiftedCandidate(
                    model=model,
                    input_equations=lifted.input_equations,
                    output_masks=lifted.output_masks,
                    output_constants=lifted.output_constants,
                )
            )
        lifted_all.append(lifted_models)
    return lifted_all


def assign_sbox_model(
    sbox_index: int,
    candidates: list[LiftedCandidate],
    system: GF2System,
    rng: random.Random,
    max_candidates: int | None = None,
    forced_index: int | None = None,
) -> AssignedModel | None:
    if forced_index is None:
        order = list(range(len(candidates)))
        rng.shuffle(order)
        if max_candidates is not None:
            order = order[:max_candidates]
    else:
        order = [forced_index % len(candidates)]
    for index in order:
        candidate = candidates[index]
        if not system.is_consistent_with(candidate.input_equations):
            continue
        system.add_equations(candidate.input_equations)
        return AssignedModel(
            candidate.model,
            candidate.input_equations,
            candidate.output_masks,
            candidate.output_constants,
        )
    return None


def model_output_equation(sbox_index: int, model: AffineSBoxModel, local_bit: int) -> tuple[int, int]:
    lifted = lift_model(sbox_index, model)
    return lifted.output_masks[local_bit], lifted.output_constants[local_bit]


def prepared_bit_rows(
    g_rows: list[tuple[int, int]],
    l_columns: list[int],
    row_order: str,
    rng: random.Random,
    candidate_counts: tuple[int, ...] | None = None,
) -> list[PreparedBitRow]:
    rows: list[PreparedBitRow] = []
    for original_index, (z_mask, rhs) in enumerate(g_rows):
        y_mask = transpose_apply(l_columns, z_mask)
        bits = []
        value = y_mask
        while value:
            low = value & -value
            bits.append(low.bit_length() - 1)
            value ^= low
        rows.append(
            PreparedBitRow(
                original_index=original_index,
                z_mask=z_mask,
                rhs=rhs,
                bits=tuple(bits),
                deps=tuple(sorted(ymask_sboxes(y_mask))),
            )
        )

    def candidate_sum(row: PreparedBitRow) -> int:
        if candidate_counts is None:
            return len(row.deps)
        return sum(candidate_counts[sbox] for sbox in row.deps)

    def candidate_max(row: PreparedBitRow) -> int:
        if candidate_counts is None or not row.deps:
            return 0
        return max(candidate_counts[sbox] for sbox in row.deps)

    if row_order == "small":
        rows.sort(key=lambda row: len(row.deps))
    elif row_order == "large":
        rows.sort(key=lambda row: len(row.deps), reverse=True)
    elif row_order == "random":
        rng.shuffle(rows)
    elif row_order == "mixed":
        rows.sort(key=lambda row: (len(row.deps), rng.random()))
    elif row_order == "bits-large":
        rows.sort(key=lambda row: (len(row.bits), len(row.deps), rng.random()), reverse=True)
    elif row_order == "tight":
        rows.sort(key=lambda row: (len(row.deps), -len(row.bits), rng.random()))
    elif row_order == "scarce":
        rows.sort(key=lambda row: (candidate_sum(row), candidate_max(row), len(row.deps), rng.random()))
    elif row_order == "scarce-large":
        rows.sort(key=lambda row: (candidate_sum(row), -len(row.deps), -len(row.bits), rng.random()))
    elif row_order == "pressure":
        rows.sort(
            key=lambda row: (
                candidate_sum(row) / max(1, len(row.deps)),
                -len(row.bits),
                -len(row.deps),
                rng.random(),
            )
        )
    else:
        raise ValueError(f"unknown row_order: {row_order}")
    return rows


def choose_next_row_bit(
    bits: tuple[int, ...],
    assigned: dict[int, AssignedModel],
    candidates: list[list[LiftedCandidate]],
) -> int:
    best_pos = 0
    best_score = 10**9
    for pos, bit in enumerate(bits):
        sbox_index = bit_to_sbox(bit)
        if sbox_index in assigned:
            return pos
        score = len(candidates[sbox_index])
        if score < best_score:
            best_score = score
            best_pos = pos
    return best_pos


def ordered_candidate_indexes(
    count: int,
    rng: random.Random,
    limit: int | None,
) -> list[int]:
    indexes = list(range(count))
    rng.shuffle(indexes)
    if limit is not None:
        indexes = indexes[:limit]
    return indexes


def materialize_y_affine_from_assignments(
    assigned: dict[int, AssignedModel],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    y_coeffs = [0] * STATE_BITS
    y_consts = [0] * STATE_BITS
    for sbox_index, assigned_model in assigned.items():
        positions = [(sbox_index // 64) * 5 * 64 + x * 64 + (sbox_index % 64) for x in range(5)]
        for local_bit, out_bit in enumerate(positions):
            y_coeffs[out_bit] = assigned_model.output_masks[local_bit]
            y_consts[out_bit] = assigned_model.output_constants[local_bit]
    return tuple(y_coeffs), tuple(y_consts)


def build_model_backtracking_connector(
    beta0_alpha1: list[SBoxTransition],
    beta1_alpha2: list[SBoxTransition],
    rate: int,
    padding_bits: int,
    seed: int = 1,
    row_order: str = "mixed",
    max_nodes: int = 50000,
    max_candidates_per_sbox: int | None = 12,
    start_system: GF2System | None = None,
    start_assigned: dict[int, AssignedModel] | None = None,
    start_row_index: int = 0,
    prepared_rows: list[PreparedBitRow] | None = None,
) -> IncrementalConnector:
    """Backtrack on first-round S-box model choices while adding G rows.

    This keeps the paper-like bitwise basic procedure, but makes the search
    state explicit: each branch chooses one affine S-box model, not a whole-row
    random attempt. The returned diagnostics identify the next G row and its
    dependent first-round S-boxes when the budget is exhausted.
    """
    l_columns, linv_columns = load_or_build_matrices()
    candidates = lifted_candidates_by_sbox(beta0_alpha1)
    rng = random.Random(seed)
    rows = prepared_rows
    if rows is None:
        candidate_counts = tuple(len(options) for options in candidates)
        rows = prepared_bit_rows(
            second_round_g_equations(beta1_alpha2),
            l_columns,
            row_order,
            rng,
            candidate_counts=candidate_counts,
        )

    if start_system is None:
        base = GF2System(STATE_BITS)
        base.add_equations(fixed_initial_bit_equations(rate, padding_bits, linv_columns))
    else:
        base = start_system.copy()
        base.inconsistent = False
    initial_assigned = dict(start_assigned or {})
    stack = [
        ModelSearchFrame(
            row_index=start_row_index,
            remaining_bits=rows[start_row_index].bits if start_row_index < len(rows) else tuple(),
            system=base,
            assigned=initial_assigned,
            equation_coeff=0,
            equation_const=0,
        )
    ]
    nodes = 0
    best_rows = start_row_index
    best_system = base
    best_assigned: dict[int, AssignedModel] = initial_assigned
    failure_reason = "search exhausted"

    while stack:
        if nodes >= max_nodes:
            failure_reason = "node limit"
            break
        frame = stack.pop()
        nodes += 1

        if frame.row_index > best_rows:
            best_rows = frame.row_index
            best_system = frame.system
            best_assigned = frame.assigned

        if frame.row_index == len(rows):
            y_coeffs, y_consts = materialize_y_affine_from_assignments(frame.assigned)
            return IncrementalConnector(
                system=frame.system,
                y_coeffs=y_coeffs,
                y_consts=y_consts,
                assigned_sboxes=len(frame.assigned),
                added_g_rows=len(rows),
                total_g_rows=len(rows),
                nodes=nodes,
            )

        row = rows[frame.row_index]
        if not frame.remaining_bits:
            trial_system = frame.system.copy()
            rc_const = (row.z_mask & apply_l_int(round_constant_state(0))).bit_count() & 1
            final_rhs = row.rhs ^ frame.equation_const ^ rc_const
            if trial_system.add_equation(frame.equation_coeff, final_rhs):
                next_row_index = frame.row_index + 1
                stack.append(
                    ModelSearchFrame(
                        row_index=next_row_index,
                        remaining_bits=rows[next_row_index].bits if next_row_index < len(rows) else tuple(),
                        system=trial_system,
                        assigned=frame.assigned,
                        equation_coeff=0,
                        equation_const=0,
                    )
                )
            continue

        bit_pos = choose_next_row_bit(frame.remaining_bits, frame.assigned, candidates)
        bit = frame.remaining_bits[bit_pos]
        rest_bits = frame.remaining_bits[:bit_pos] + frame.remaining_bits[bit_pos + 1 :]
        sbox_index = bit_to_sbox(bit)
        local_bit = bit_local_index(bit)
        assigned_model = frame.assigned.get(sbox_index)
        if assigned_model is not None:
            stack.append(
                ModelSearchFrame(
                    row_index=frame.row_index,
                    remaining_bits=rest_bits,
                    system=frame.system,
                    assigned=frame.assigned,
                    equation_coeff=frame.equation_coeff ^ assigned_model.output_masks[local_bit],
                    equation_const=frame.equation_const ^ assigned_model.output_constants[local_bit],
                )
            )
            continue

        candidate_indexes = ordered_candidate_indexes(
            len(candidates[sbox_index]),
            rng,
            max_candidates_per_sbox,
        )
        for candidate_index in reversed(candidate_indexes):
            candidate = candidates[sbox_index][candidate_index]
            trial_system = frame.system.copy()
            if not trial_system.add_equations(candidate.input_equations):
                continue
            trial_assigned = dict(frame.assigned)
            assigned_candidate = AssignedModel(
                candidate.model,
                candidate.input_equations,
                candidate.output_masks,
                candidate.output_constants,
            )
            trial_assigned[sbox_index] = assigned_candidate
            stack.append(
                ModelSearchFrame(
                    row_index=frame.row_index,
                    remaining_bits=rest_bits,
                    system=trial_system,
                    assigned=trial_assigned,
                    equation_coeff=frame.equation_coeff ^ candidate.output_masks[local_bit],
                    equation_const=frame.equation_const ^ candidate.output_constants[local_bit],
                )
            )

    best_system = best_system.copy()
    best_system.inconsistent = True
    y_coeffs, y_consts = materialize_y_affine_from_assignments(best_assigned)
    failed_row = rows[best_rows].original_index if best_rows < len(rows) else None
    failed_sboxes = rows[best_rows].deps if best_rows < len(rows) else tuple()
    return IncrementalConnector(
        system=best_system,
        y_coeffs=y_coeffs,
        y_consts=y_consts,
        assigned_sboxes=len(best_assigned),
        added_g_rows=best_rows,
        total_g_rows=len(rows),
        nodes=nodes,
        failed_row=failed_row,
        failed_sboxes=failed_sboxes,
        failure_reason=failure_reason,
    )


def build_incremental_connector(
    beta0_alpha1: list[SBoxTransition],
    beta1_alpha2: list[SBoxTransition],
    rate: int,
    padding_bits: int,
    seed: int = 1,
    row_retries: int = 20,
    row_order: str = "mixed",
) -> IncrementalConnector:
    l_columns, linv_columns = load_or_build_matrices()
    candidates = model_candidates_by_sbox(beta0_alpha1)
    g_rows = second_round_g_equations(beta1_alpha2)
    prepared = []
    for z_mask, rhs in g_rows:
        y_mask = transpose_apply(l_columns, z_mask)
        prepared.append((z_mask, rhs, y_mask, ymask_sboxes(y_mask)))
    if row_order == "small":
        prepared.sort(key=lambda item: len(item[3]))
    elif row_order == "large":
        prepared.sort(key=lambda item: len(item[3]), reverse=True)
    elif row_order == "random":
        rng = random.Random(seed)
        rng.shuffle(prepared)
    elif row_order == "mixed":
        rng = random.Random(seed)
        prepared.sort(key=lambda item: (len(item[3]), rng.random()))
    else:
        raise ValueError(f"unknown row_order: {row_order}")

    rng = random.Random(seed)
    system = GF2System(STATE_BITS)
    system.add_equations(fixed_initial_bit_equations(rate, padding_bits, linv_columns))
    y_coeffs = [0] * STATE_BITS
    y_consts = [0] * STATE_BITS
    assigned: set[int] = set()
    added_g_rows = 0

    for z_mask, rhs, _y_mask, deps in prepared:
        missing = [sbox for sbox in deps if sbox not in assigned]
        success = False
        for _ in range(row_retries):
            trial_system = system.copy()
            trial_coeffs = list(y_coeffs)
            trial_consts = list(y_consts)
            trial_assigned = set(assigned)
            rng.shuffle(missing)
            if all(
                assign_sbox(
                    sbox,
                    candidates[sbox],
                    trial_system,
                    trial_coeffs,
                    trial_consts,
                    rng,
                )
                for sbox in missing
            ):
                equation = substitute_second_round_equation(
                    z_mask,
                    rhs,
                    trial_coeffs,
                    trial_consts,
                    l_columns,
                    round_constant_state(0),
                )
                if trial_system.add_equation(*equation):
                    system = trial_system
                    y_coeffs = trial_coeffs
                    y_consts = trial_consts
                    assigned = trial_assigned | set(missing)
                    added_g_rows += 1
                    success = True
                    break
        if not success:
            system.inconsistent = True
            break

    return IncrementalConnector(
        system=system,
        y_coeffs=tuple(y_coeffs),
        y_consts=tuple(y_consts),
        assigned_sboxes=len(assigned),
        added_g_rows=added_g_rows,
        total_g_rows=len(g_rows),
    )


def build_bitwise_connector(
    beta0_alpha1: list[SBoxTransition],
    beta1_alpha2: list[SBoxTransition],
    rate: int,
    padding_bits: int,
    seed: int = 1,
    row_order: str = "mixed",
    row_retries: int = 100,
) -> IncrementalConnector:
    """Basic linearization closer to the paper: process needed y bits."""
    l_columns, linv_columns = load_or_build_matrices()
    candidates = lifted_candidates_by_sbox(beta0_alpha1)
    g_rows = second_round_g_equations(beta1_alpha2)
    prepared = []
    for z_mask, rhs in g_rows:
        y_mask = transpose_apply(l_columns, z_mask)
        bits = []
        value = y_mask
        while value:
            low = value & -value
            bits.append(low.bit_length() - 1)
            value ^= low
        prepared.append((z_mask, rhs, y_mask, bits))

    rng = random.Random(seed)
    if row_order == "small":
        prepared.sort(key=lambda item: len(ymask_sboxes(item[2])))
    elif row_order == "large":
        prepared.sort(key=lambda item: len(ymask_sboxes(item[2])), reverse=True)
    elif row_order == "random":
        rng.shuffle(prepared)
    elif row_order == "mixed":
        prepared.sort(key=lambda item: (len(ymask_sboxes(item[2])), rng.random()))
    else:
        raise ValueError(f"unknown row_order: {row_order}")

    system = GF2System(STATE_BITS)
    system.add_equations(fixed_initial_bit_equations(rate, padding_bits, linv_columns))
    assigned: dict[int, AssignedModel] = {}
    added_g_rows = 0

    for _z_mask, rhs, _y_mask, bits in prepared:
        success = False
        for _ in range(row_retries):
            trial_system = system.copy()
            trial_assigned = dict(assigned)
            equation_coeff = 0
            equation_const = 0
            bit_order = list(bits)
            rng.shuffle(bit_order)
            ok = True
            for bit in bit_order:
                sbox_index = bit_to_sbox(bit)
                local_bit = bit_local_index(bit)
                assigned_model = trial_assigned.get(sbox_index)
                if assigned_model is None:
                    assigned_model = assign_sbox_model(
                        sbox_index,
                        candidates[sbox_index],
                        trial_system,
                        rng,
                    )
                    if assigned_model is None:
                        ok = False
                        break
                    trial_assigned[sbox_index] = assigned_model
                coeff = assigned_model.output_masks[local_bit]
                const = assigned_model.output_constants[local_bit]
                equation_coeff ^= coeff
                equation_const ^= const
            if not ok:
                continue
            rc_const = (_z_mask & apply_l_int(round_constant_state(0))).bit_count() & 1
            final_rhs = rhs ^ equation_const ^ rc_const
            if trial_system.add_equation(equation_coeff, final_rhs):
                system = trial_system
                assigned = trial_assigned
                added_g_rows += 1
                success = True
                break
        if not success:
            system.inconsistent = True
            break

    # Materialize y coefficients for any assigned S-boxes, mostly for debugging.
    y_coeffs = [0] * STATE_BITS
    y_consts = [0] * STATE_BITS
    for sbox_index, assigned_model in assigned.items():
        lifted = lift_model(sbox_index, assigned_model.model)
        positions = [(sbox_index // 64) * 5 * 64 + x * 64 + (sbox_index % 64) for x in range(5)]
        for local_bit, out_bit in enumerate(positions):
            y_coeffs[out_bit] = lifted.output_masks[local_bit]
            y_consts[out_bit] = lifted.output_constants[local_bit]

    return IncrementalConnector(
        system=system,
        y_coeffs=tuple(y_coeffs),
        y_consts=tuple(y_consts),
        assigned_sboxes=len(assigned),
        added_g_rows=added_g_rows,
        total_g_rows=len(g_rows),
    )


def build_bitwise_prefix_connector(
    beta0_alpha1: list[SBoxTransition],
    beta1_alpha2: list[SBoxTransition],
    rate: int,
    padding_bits: int,
    seed: int = 1,
    row_order: str = "mixed",
    row_retries: int = 100,
    stop_after_rows: int | None = None,
    prepared_rows: list[PreparedBitRow] | None = None,
    preassign_sbox_order: tuple[int, ...] = (),
    forced_model_choices: dict[int, int] | None = None,
) -> tuple[IncrementalConnector, dict[int, AssignedModel], list[PreparedBitRow]]:
    """Build a greedy bitwise prefix and expose assignments for repair search."""
    l_columns, linv_columns = load_or_build_matrices()
    candidates = lifted_candidates_by_sbox(beta0_alpha1)
    rng = random.Random(seed)
    if prepared_rows is None:
        candidate_counts = tuple(len(options) for options in candidates)
        rows = prepared_bit_rows(
            second_round_g_equations(beta1_alpha2),
            l_columns,
            row_order,
            rng,
            candidate_counts=candidate_counts,
        )
    else:
        rows = prepared_rows
    forced_model_choices = forced_model_choices or {}

    system = GF2System(STATE_BITS)
    system.add_equations(fixed_initial_bit_equations(rate, padding_bits, linv_columns))
    assigned: dict[int, AssignedModel] = {}
    added_g_rows = 0
    for sbox_index in preassign_sbox_order:
        if sbox_index in assigned:
            continue
        assigned_model = assign_sbox_model(
            sbox_index,
            candidates[sbox_index],
            system,
            rng,
            forced_index=forced_model_choices.get(sbox_index),
        )
        if assigned_model is None:
            system.inconsistent = True
            break
        assigned[sbox_index] = assigned_model

    if not system.inconsistent:
        for row in rows:
            if stop_after_rows is not None and added_g_rows >= stop_after_rows:
                break
            success = False
            for _ in range(row_retries):
                trial_system = system.copy()
                trial_assigned = dict(assigned)
                equation_coeff = 0
                equation_const = 0
                bit_order = list(row.bits)
                rng.shuffle(bit_order)
                ok = True
                for bit in bit_order:
                    sbox_index = bit_to_sbox(bit)
                    local_bit = bit_local_index(bit)
                    assigned_model = trial_assigned.get(sbox_index)
                    if assigned_model is None:
                        assigned_model = assign_sbox_model(
                            sbox_index,
                            candidates[sbox_index],
                            trial_system,
                            rng,
                            forced_index=forced_model_choices.get(sbox_index),
                        )
                        if assigned_model is None:
                            ok = False
                            break
                        trial_assigned[sbox_index] = assigned_model
                    equation_coeff ^= assigned_model.output_masks[local_bit]
                    equation_const ^= assigned_model.output_constants[local_bit]
                if not ok:
                    continue
                rc_const = (row.z_mask & apply_l_int(round_constant_state(0))).bit_count() & 1
                final_rhs = row.rhs ^ equation_const ^ rc_const
                if trial_system.add_equation(equation_coeff, final_rhs):
                    system = trial_system
                    assigned = trial_assigned
                    added_g_rows += 1
                    success = True
                    break
            if not success:
                system.inconsistent = True
                break

    y_coeffs, y_consts = materialize_y_affine_from_assignments(assigned)
    failed_row = rows[added_g_rows].original_index if added_g_rows < len(rows) else None
    failed_sboxes = rows[added_g_rows].deps if added_g_rows < len(rows) else tuple()
    connector = IncrementalConnector(
        system=system,
        y_coeffs=y_coeffs,
        y_consts=y_consts,
        assigned_sboxes=len(assigned),
        added_g_rows=added_g_rows,
        total_g_rows=len(rows),
        failed_row=failed_row,
        failed_sboxes=failed_sboxes,
        failure_reason="greedy prefix stopped" if system.inconsistent else "",
    )
    return connector, assigned, rows


def build_backtracking_connector(
    beta0_alpha1: list[SBoxTransition],
    beta1_alpha2: list[SBoxTransition],
    rate: int,
    padding_bits: int,
    seed: int = 1,
    row_order: str = "mixed",
    max_nodes: int = 5000,
    options_per_row: int = 40,
) -> IncrementalConnector:
    """Row-level backtracking version of the connector builder."""
    l_columns, linv_columns = load_or_build_matrices()
    candidates = model_candidates_by_sbox(beta0_alpha1)
    g_rows = second_round_g_equations(beta1_alpha2)
    prepared = []
    for z_mask, rhs in g_rows:
        y_mask = transpose_apply(l_columns, z_mask)
        prepared.append((z_mask, rhs, ymask_sboxes(y_mask)))

    rng = random.Random(seed)
    if row_order == "small":
        prepared.sort(key=lambda item: len(item[2]))
    elif row_order == "large":
        prepared.sort(key=lambda item: len(item[2]), reverse=True)
    elif row_order == "random":
        rng.shuffle(prepared)
    elif row_order == "mixed":
        prepared.sort(key=lambda item: (len(item[2]), rng.random()))
    else:
        raise ValueError(f"unknown row_order: {row_order}")

    base = GF2System(STATE_BITS)
    base.add_equations(fixed_initial_bit_equations(rate, padding_bits, linv_columns))
    zero_coeffs = [0] * STATE_BITS
    zero_consts = [0] * STATE_BITS

    # Stack entries: row index, state before trying that row, attempts used.
    stack: list[tuple[int, GF2System, list[int], list[int], set[int], int]] = [
        (0, base, zero_coeffs, zero_consts, set(), 0)
    ]
    nodes = 0
    best: tuple[int, GF2System, list[int], list[int], set[int]] = (
        0,
        base,
        zero_coeffs,
        zero_consts,
        set(),
    )

    while stack and nodes < max_nodes:
        row_index, system, y_coeffs, y_consts, assigned, attempts_used = stack.pop()
        if row_index > best[0]:
            best = (row_index, system, y_coeffs, y_consts, assigned)
        if row_index == len(prepared):
            return IncrementalConnector(
                system=system,
                y_coeffs=tuple(y_coeffs),
                y_consts=tuple(y_consts),
                assigned_sboxes=len(assigned),
                added_g_rows=row_index,
                total_g_rows=len(prepared),
            )
        if attempts_used >= options_per_row:
            continue

        # Re-push this frame with one more used attempt, so later failures can
        # return here and try another local assignment.
        stack.append((row_index, system, y_coeffs, y_consts, assigned, attempts_used + 1))

        z_mask, rhs, deps = prepared[row_index]
        missing = [sbox for sbox in deps if sbox not in assigned]
        rng.shuffle(missing)
        trial_system = system.copy()
        trial_coeffs = list(y_coeffs)
        trial_consts = list(y_consts)
        trial_assigned = set(assigned)

        ok = True
        for sbox in missing:
            if not assign_sbox(
                sbox,
                candidates[sbox],
                trial_system,
                trial_coeffs,
                trial_consts,
                rng,
            ):
                ok = False
                break
            trial_assigned.add(sbox)
        if not ok:
            nodes += 1
            continue

        equation = substitute_second_round_equation(
            z_mask,
            rhs,
            trial_coeffs,
            trial_consts,
            l_columns,
            round_constant_state(0),
        )
        if not trial_system.add_equation(*equation):
            nodes += 1
            continue

        nodes += 1
        stack.append((row_index + 1, trial_system, trial_coeffs, trial_consts, trial_assigned, 0))

    best_rows, best_system, best_coeffs, best_consts, best_assigned = best
    best_system = best_system.copy()
    best_system.inconsistent = True
    return IncrementalConnector(
        system=best_system,
        y_coeffs=tuple(best_coeffs),
        y_consts=tuple(best_consts),
        assigned_sboxes=len(best_assigned),
        added_g_rows=best_rows,
        total_g_rows=len(prepared),
    )


def demo() -> None:
    from connector_runner import SBoxTransition as T

    connector = build_incremental_connector(
        beta0_alpha1=[T(0, 0x03, 0x02)],
        beta1_alpha2=[T(0, 0x03, 0x02)],
        rate=1440,
        padding_bits=1,
        seed=1,
    )
    print("incremental connector checks passed")
    print(f"  inconsistent: {connector.system.inconsistent}")
    print(f"  assigned S-boxes: {connector.assigned_sboxes}")
    print(f"  added G rows: {connector.added_g_rows}")


if __name__ == "__main__":
    demo()
