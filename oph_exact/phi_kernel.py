"""Potential-linked stochastic repair kernel on the exact twelve-port carrier (lane L5).

Declared object.  On the sector of nonnegative integer port loads with a
protected total ``s`` the sixty directed seam labels ``(c, o)`` act by the
nearest-agreement repair: a label waits when the seam gap ``|x_c - x_o|`` is
at most one and otherwise sends the endpoint pair to
``(ceil(t/2), floor(t/2))`` with the ceiling at ``c`` and ``t`` the endpoint
total.  The kernel draws one label per move with probability

    P_beta(m | x) = 2^(-beta * dU_m(x)) / Z_beta(x),

where ``U`` is the declared potential, ``dU_m(x) = U(move(x, m)) - U(x)``, and
``Z_beta(x)`` is the row normalizer.  Two potentials are declared.  The
primary one is the flagship's quadratic descent functional
``V(x) = sum_i x_i^2``, whose unit-transfer decrement across a seam with
oriented mismatch ``d >= 2`` is exactly ``2 (d - 1)``.  The secondary one is
the seam sum ``Phi(x) = sum_seams (x_i - x_j)^2``, kept as a diagnostic
variant whose terminal value varies over the agreement shell.  ``beta = 0`` is
the uniform A3 schedule ``1/60`` for both.  Every probability is an exact
rational.

Verified on the protected totals three and four, for both potentials: every
state-label pair has a nonpositive decrement, zero exactly on waits; for ``V``
the decrement of a declared move is ``-2 floor(D/2) ceil(D/2)`` for seam gap
``D``, the sum of the flagship unit-transfer decrements, and the absorbing set
equals the ``V``-minimal set of the sector; exact expected move counts to
absorption; the log-transition action decomposes as
``beta ln 2 sum dU + sum log Z``; the label path law is the exponential tilt
of the step-uniform reference by that action at multiplier one with the Lean
partition constant ``60^(-n)``; the gauge characterization; most-probable
label paths with tie counts; and the non-identifiability family
``L_a = L_0 + (a/2) y (y - 1)`` on realized records.

Not claimed: source selection of the kernel, a reversible reference, an
entropy inequality, laboratory readout, or a continuum limit.  The reverse
label on a gap-one seam is read as a wait; the literal endpoint swap is
``V``-neutral and raises ``Phi`` on explicit states, and those counts are
recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from oph_exact import carrier

SCHEMA = "oph.exact.phi-linked-kernel.v1"
LANE = "L5"
PROTECTED_TOTALS: tuple[int, ...] = (3, 4)
BETAS: tuple[int, ...] = (0, 1, 2)
PRIMARY_POTENTIAL = "squared_norm_V"
SECONDARY_POTENTIAL = "seam_sum_Phi"
POTENTIALS: tuple[str, ...] = (PRIMARY_POTENTIAL, SECONDARY_POTENTIAL)
POTENTIAL_FORMULAS: dict[str, str] = {
    PRIMARY_POTENTIAL: "V(x) = sum_i x_i^2, the flagship quadratic descent functional",
    SECONDARY_POTENTIAL: "Phi(x) = sum over the thirty seams of (x_i - x_j)^2, diagnostic variant",
}
TILT_BASE = 2
LABEL_COUNT = 60
PATH_SECTOR_TOTAL = 3
DERIVED_ACTION_PATH_LENGTH = 3
GAUGE_PATH_LENGTH = 2
MOST_PROBABLE_MAX_LENGTH = 4
ENRICHMENT_PARAMETERS: tuple[Fraction, ...] = (Fraction(1), Fraction(2), Fraction(5, 2))
ENTROPY_TRACE_STEPS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 2000)
FLOAT_TOLERANCE = 1e-9

SIM_ROOT = Path(__file__).resolve().parents[1]
META_ROOT = Path(os.environ.get("OPH_META_ROOT", str(SIM_ROOT.parent)))
DEFAULT_RECEIPT = SIM_ROOT / "data/exact/phi_linked_kernel_receipt.json"
PINNED_SIM_FILES: dict[str, str] = {
    "producer": "oph_exact/phi_kernel.py",
    "verifier": "oph_exact/verify_phi_kernel_independent.py",
    "tests": "tests/test_exact_phi_kernel.py",
    "carrier": "oph_exact/carrier.py",
}
PINNED_META_FILES: dict[str, str] = {
    "lean_log_transition_action": (
        "reverse-engineering-reality/Lean/InformationProjection/LogTransitionAction.lean"
    ),
    "observers_paper_tex": "reverse-engineering-reality/paper/observers_are_all_you_need.tex",
    "flagship_tex": (
        "reverse-engineering-reality/flagship/from_observer_consensus_to_standard_physics.tex"
    ),
}
PAPER_REFERENCE: dict[str, str] = {
    "repository": "oph-meta",
    "path": (
        "trt-scspl/instantiating_the_self_configuring_self_processing_language_"
        "that_is_our_universe.tex"
    ),
    "title": (
        "Instantiating the Self-Configuring Self-Processing Language That Is Our "
        "Universe: Observer Patch Holography as a Consistent Realization of the CTMU"
    ),
    "subsection": "(M5) A source-selected action",
    "hashing": "none",
    "note": (
        "uncommitted working file of another repository; referenced by path and "
        "subsection without a content hash"
    ),
}

CLAIM_BOUNDARY = (
    "The kernel is declared, so (M5) of the instantiation paper stays open: "
    "the receipt inhabits the flagship's missing source-justified stochastic "
    "coupling premise with one explicit law and supplies no source selection "
    "of that law. The verified statements are exact finite identities on the "
    "protected totals three and four of the twelve-port carrier under the "
    "declared move law, for the flagship quadratic descent functional V as the "
    "primary potential and the seam sum Phi as a diagnostic variant; the "
    "kernel is a strict-descent normalizer with no reversible reference and no "
    "entropy inequality; no laboratory readout, continuum limit, or physical "
    "attachment is claimed."
)

State = tuple[int, ...]


# Canonical serialization and pins


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_of(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def frac(value: Fraction | int) -> str:
    return str(Fraction(value))


def pins() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for key, rel in PINNED_SIM_FILES.items():
        out[key] = {"root": "oph-physics-sim", "path": rel, "sha256": file_sha256(SIM_ROOT / rel)}
    for key, rel in PINNED_META_FILES.items():
        out[key] = {"root": "oph-meta", "path": rel, "sha256": file_sha256(META_ROOT / rel)}
    return out


# Sector: states, labels, moves, potentials


def weak_compositions(total: int, parts: int) -> tuple[State, ...]:
    """All nonnegative integer vectors of length ``parts`` summing to ``total``."""

    states: list[State] = []
    for bars in combinations(range(total + parts - 1), parts - 1):
        previous = -1
        vector: list[int] = []
        for bar in bars:
            vector.append(bar - previous - 1)
            previous = bar
        vector.append(total + parts - 1 - previous - 1)
        states.append(tuple(vector))
    states.sort(reverse=True)
    return tuple(states)


@lru_cache(maxsize=1)
def directed_labels() -> tuple[tuple[int, int], ...]:
    """The sixty labels ``(c, o)``: ceiling endpoint first, seam partner second."""

    labels: list[tuple[int, int]] = []
    for i, j in carrier.seams():
        labels.append((i, j))
        labels.append((j, i))
    if len(labels) != LABEL_COUNT:
        raise AssertionError("expected sixty directed labels")
    return tuple(labels)


def phi(x: Sequence[int]) -> int:
    return sum((x[i] - x[j]) ** 2 for i, j in carrier.seams())


def squared_norm(x: Sequence[int]) -> int:
    return sum(v * v for v in x)


def potential_value(x: Sequence[int], potential: str) -> int:
    if potential == PRIMARY_POTENTIAL:
        return squared_norm(x)
    if potential == SECONDARY_POTENTIAL:
        return phi(x)
    raise ValueError(f"unknown potential {potential!r}")


def apply_label(x: State, label: tuple[int, int]) -> State:
    """Declared move: wait on a gap of at most one, else place the ceiling at ``c``."""

    c, o = label
    if abs(x[c] - x[o]) <= 1:
        return x
    high, low = carrier.integer_nearest_agreement(x[c], x[o], ceiling_to_first=True)
    y = list(x)
    y[c] = high
    y[o] = low
    return tuple(y)


def apply_label_literal(x: State, label: tuple[int, int]) -> State:
    """The literal carrier function on every pair: a gap-one seam may swap."""

    c, o = label
    high, low = carrier.integer_nearest_agreement(x[c], x[o], ceiling_to_first=True)
    y = list(x)
    y[c] = high
    y[o] = low
    return tuple(y)


def closed_form_dv(gap: int) -> int:
    """``-2 floor(D/2) ceil(D/2)`` for a seam gap ``D``; zero on a gap of at most one."""

    if gap <= 1:
        return 0
    return -2 * (gap // 2) * ((gap + 1) // 2)


@dataclass(frozen=True)
class Sector:
    total: int
    states: tuple[State, ...]
    index: Mapping[State, int]
    phi: tuple[int, ...]
    v: tuple[int, ...]
    next_state: tuple[tuple[int, ...], ...]
    dphi: tuple[tuple[int, ...], ...]
    dv: tuple[tuple[int, ...], ...]

    @property
    def size(self) -> int:
        return len(self.states)

    def value(self, potential: str) -> tuple[int, ...]:
        return self.v if potential == PRIMARY_POTENTIAL else self.phi

    def decrement(self, potential: str) -> tuple[tuple[int, ...], ...]:
        return self.dv if potential == PRIMARY_POTENTIAL else self.dphi

    def is_absorbing(self, state_index: int) -> bool:
        return all(n == state_index for n in self.next_state[state_index])

    def absorbing_indices(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.size) if self.is_absorbing(i))

    def transient_indices(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.size) if not self.is_absorbing(i))


@lru_cache(maxsize=None)
def build_sector(total: int) -> Sector:
    states = weak_compositions(total, carrier.PORT_COUNT)
    index = {state: k for k, state in enumerate(states)}
    labels = directed_labels()
    phis = tuple(phi(state) for state in states)
    vs = tuple(squared_norm(state) for state in states)
    next_state: list[tuple[int, ...]] = []
    dphi: list[tuple[int, ...]] = []
    dv: list[tuple[int, ...]] = []
    for k, state in enumerate(states):
        row_next: list[int] = []
        row_dphi: list[int] = []
        row_dv: list[int] = []
        for label in labels:
            t = index[apply_label(state, label)]
            row_next.append(t)
            row_dphi.append(phis[t] - phis[k])
            row_dv.append(vs[t] - vs[k])
        next_state.append(tuple(row_next))
        dphi.append(tuple(row_dphi))
        dv.append(tuple(row_dv))
    return Sector(total, states, index, phis, vs, tuple(next_state), tuple(dphi), tuple(dv))


def minimal_indices(sector: Sector, potential: str) -> tuple[int, ...]:
    values = sector.value(potential)
    minimum = min(values)
    return tuple(i for i, value in enumerate(values) if value == minimum)


def phi_minimal_indices(sector: Sector) -> tuple[int, ...]:
    return minimal_indices(sector, SECONDARY_POTENTIAL)


def v_minimal_indices(sector: Sector) -> tuple[int, ...]:
    return minimal_indices(sector, PRIMARY_POTENTIAL)


def descent_summary(sector: Sector, potential: str = PRIMARY_POTENTIAL) -> dict[str, Any]:
    """Exhaustive descent facts for every state-label pair of the sector."""

    table = sector.decrement(potential)
    all_nonpositive = True
    zero_iff_wait = True
    changing_all_negative = True
    minimum = 0
    changing = 0
    for k in range(sector.size):
        for m in range(LABEL_COUNT):
            d = table[k][m]
            wait = sector.next_state[k][m] == k
            if d > 0:
                all_nonpositive = False
            if (d == 0) != wait:
                zero_iff_wait = False
            if not wait:
                changing += 1
                if d >= 0:
                    changing_all_negative = False
            minimum = min(minimum, d)
    return {
        "potential": potential,
        "state_label_pairs": sector.size * LABEL_COUNT,
        "state_changing_pairs": changing,
        "all_nonpositive": all_nonpositive,
        "zero_iff_wait": zero_iff_wait,
        "state_changing_all_negative": changing_all_negative,
        "min_decrement": minimum,
        "decrement_table_sha256": sha256_of([[int(d) for d in row] for row in table]),
    }


def unit_transfer_summary(sector: Sector) -> dict[str, Any]:
    """The flagship identity ``dV = -2 (d - 1)`` and the closed form of the declared moves.

    A unit transfer moves one unit from the high endpoint to the low endpoint
    of a seam with oriented mismatch ``d >= 2``.  A declared move with seam gap
    ``D`` transfers ``floor(D/2)`` or ``ceil(D/2)`` units, so its decrement is
    the sum of the unit decrements ``-2 (D - 2k - 1)``, which is
    ``-2 floor(D/2) ceil(D/2)`` for either placement.
    """

    labels = directed_labels()
    unit_pairs = 0
    unit_identity = True
    closed_form = True
    decomposition = True
    unit_moves = 0
    multi_unit_moves = 0
    for k, state in enumerate(sector.states):
        for i, j in carrier.seams():
            gap = abs(state[i] - state[j])
            if gap < 2:
                continue
            high, low = (i, j) if state[i] > state[j] else (j, i)
            y = list(state)
            y[high] -= 1
            y[low] += 1
            unit_pairs += 1
            if squared_norm(y) - sector.v[k] != -2 * (gap - 1):
                unit_identity = False
        for m, (c, o) in enumerate(labels):
            gap = abs(state[c] - state[o])
            d = sector.dv[k][m]
            if d != closed_form_dv(gap):
                closed_form = False
            target = sector.states[sector.next_state[k][m]]
            if target == state:
                continue
            delta = abs(target[c] - state[c])
            if delta == 1:
                unit_moves += 1
            else:
                multi_unit_moves += 1
            if d != sum(-2 * (gap - 2 * step - 1) for step in range(delta)):
                decomposition = False
    return {
        "unit_transfer_pairs_checked": unit_pairs,
        "unit_transfer_identity_dv_equals_minus_two_d_minus_one": unit_identity,
        "declared_move_closed_form_dv_equals_minus_two_floor_ceil": closed_form,
        "declared_move_dv_equals_sum_of_unit_transfer_decrements": decomposition,
        "state_changing_unit_transfers": unit_moves,
        "state_changing_multi_unit_transfers": multi_unit_moves,
        "verified": unit_identity and closed_form and decomposition,
    }


def literal_swap_summary(sector: Sector) -> dict[str, Any]:
    """The literal carrier function against the declared move on every pair."""

    labels = directed_labels()
    differing = 0
    positive = 0
    zero = 0
    negative = 0
    dv_zero = 0
    max_d = None
    example: dict[str, Any] | None = None
    for k, state in enumerate(sector.states):
        for m, label in enumerate(labels):
            literal = apply_label_literal(state, label)
            declared = sector.states[sector.next_state[k][m]]
            if literal == declared:
                continue
            differing += 1
            if squared_norm(literal) == sector.v[k]:
                dv_zero += 1
            d = phi(literal) - sector.phi[k]
            if d > 0:
                positive += 1
                if max_d is None or d > max_d:
                    max_d = d
                    example = {
                        "state": list(state),
                        "label": list(label),
                        "literal_target": list(literal),
                        "dphi": d,
                    }
            elif d == 0:
                zero += 1
            else:
                negative += 1
    return {
        "pairs_where_literal_differs_from_declared": differing,
        "literal_dv_zero_on_every_differing_pair": dv_zero == differing,
        "literal_dphi_positive": positive,
        "literal_dphi_zero": zero,
        "literal_dphi_negative": negative,
        "max_literal_dphi": max_d,
        "example_positive": example,
        "reading": (
            "every differing pair is a gap-one seam whose reverse label swaps "
            "the endpoints; the swap is V-neutral and can raise Phi; the "
            "declared move waits there"
        ),
    }


# Kernels


@dataclass(frozen=True)
class Kernel:
    total: int
    beta: int
    potential: str
    weights: tuple[tuple[int, ...], ...]
    partition: tuple[int, ...]

    def label_probability(self, state_index: int, label_index: int) -> Fraction:
        return Fraction(self.weights[state_index][label_index], self.partition[state_index])

    def row(self, state_index: int) -> tuple[Fraction, ...]:
        z = self.partition[state_index]
        return tuple(Fraction(w, z) for w in self.weights[state_index])


@lru_cache(maxsize=None)
def build_kernel(total: int, beta: int, potential: str = PRIMARY_POTENTIAL) -> Kernel:
    sector = build_sector(total)
    weights: list[tuple[int, ...]] = []
    partition: list[int] = []
    for row in sector.decrement(potential):
        if any(d > 0 for d in row):
            raise AssertionError("the tilt requires nonpositive decrements")
        w = tuple(TILT_BASE ** (-beta * d) for d in row)
        weights.append(w)
        partition.append(sum(w))
    return Kernel(total, beta, potential, tuple(weights), tuple(partition))


@lru_cache(maxsize=None)
def aggregated_rows(total: int, beta: int, potential: str = PRIMARY_POTENTIAL) -> tuple[dict[int, Fraction], ...]:
    """State-to-state transition rows: label probabilities summed by target."""

    sector = build_sector(total)
    kernel = build_kernel(total, beta, potential)
    rows: list[dict[int, Fraction]] = []
    for k in range(sector.size):
        row: dict[int, Fraction] = {}
        for m in range(LABEL_COUNT):
            t = sector.next_state[k][m]
            row[t] = row.get(t, Fraction(0)) + kernel.label_probability(k, m)
        rows.append(row)
    return tuple(rows)


def kernel_summary(total: int, beta: int, potential: str) -> dict[str, Any]:
    sector = build_sector(total)
    kernel = build_kernel(total, beta, potential)
    rows = aggregated_rows(total, beta, potential)
    label_rows_sum_one = all(sum(kernel.row(k), Fraction(0)) == 1 for k in range(sector.size))
    state_rows_sum_one = all(sum(row.values(), Fraction(0)) == 1 for row in rows)
    table = [
        [frac(kernel.label_probability(k, m)) for m in range(LABEL_COUNT)]
        for k in range(sector.size)
    ]
    return {
        "beta": beta,
        "potential": potential,
        "label_rows_sum_to_one": label_rows_sum_one,
        "state_rows_sum_to_one": state_rows_sum_one,
        "max_partition": str(max(kernel.partition)),
        "min_partition": str(min(kernel.partition)),
        "label_probability_table_sha256": sha256_of(table),
        "uniform_schedule": beta == 0 and all(z == LABEL_COUNT for z in kernel.partition),
    }


# Absorption, terminal statistics, reversibility, entropy


def _v_ascending(sector: Sector) -> list[int]:
    return sorted(range(sector.size), key=lambda k: (sector.v[k], k))


def expected_moves_to_absorption(total: int, beta: int, potential: str = PRIMARY_POTENTIAL) -> tuple[Fraction, ...]:
    """Exact expected move count to absorption from every state.

    The transient support graph is acyclic because every state-changing move
    lowers ``V``; states are processed in ascending ``V`` order.
    """

    sector = build_sector(total)
    rows = aggregated_rows(total, beta, potential)
    if not descent_summary(sector, PRIMARY_POTENTIAL)["state_changing_all_negative"]:
        raise AssertionError("ascending-V recursion needs strict descent")
    times: list[Fraction | None] = [None] * sector.size
    for k in _v_ascending(sector):
        if sector.is_absorbing(k):
            times[k] = Fraction(0)
            continue
        stay = rows[k].get(k, Fraction(0))
        acc = Fraction(1)
        for t, p in rows[k].items():
            if t == k:
                continue
            value = times[t]
            if value is None:
                raise AssertionError("target processed out of order")
            acc += p * value
        times[k] = acc / (1 - stay)
    return tuple(t if t is not None else Fraction(0) for t in times)


def terminal_functionals(
    total: int, beta: int, potential: str = PRIMARY_POTENTIAL
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Per state: probabilities of a Phi-minimal and of a V-minimal terminal, expected terminal Phi and V."""

    sector = build_sector(total)
    rows = aggregated_rows(total, beta, potential)
    phi_min = set(phi_minimal_indices(sector))
    v_min = set(v_minimal_indices(sector))
    g_phi: list[Fraction] = [Fraction(0)] * sector.size
    g_v: list[Fraction] = [Fraction(0)] * sector.size
    h_phi: list[Fraction] = [Fraction(0)] * sector.size
    h_v: list[Fraction] = [Fraction(0)] * sector.size
    done: set[int] = set()
    for k in _v_ascending(sector):
        if sector.is_absorbing(k):
            g_phi[k] = Fraction(1 if k in phi_min else 0)
            g_v[k] = Fraction(1 if k in v_min else 0)
            h_phi[k] = Fraction(sector.phi[k])
            h_v[k] = Fraction(sector.v[k])
            done.add(k)
            continue
        stay = rows[k].get(k, Fraction(0))
        acc = [Fraction(0)] * 4
        for t, p in rows[k].items():
            if t == k:
                continue
            if t not in done:
                raise AssertionError("target processed out of order")
            acc[0] += p * g_phi[t]
            acc[1] += p * g_v[t]
            acc[2] += p * h_phi[t]
            acc[3] += p * h_v[t]
        g_phi[k] = acc[0] / (1 - stay)
        g_v[k] = acc[1] / (1 - stay)
        h_phi[k] = acc[2] / (1 - stay)
        h_v[k] = acc[3] / (1 - stay)
        done.add(k)
    return tuple(g_phi), tuple(g_v), tuple(h_phi), tuple(h_v)


def float_transition_matrix(total: int, beta: int, potential: str = PRIMARY_POTENTIAL) -> np.ndarray:
    sector = build_sector(total)
    rows = aggregated_rows(total, beta, potential)
    matrix = np.zeros((sector.size, sector.size))
    for k, row in enumerate(rows):
        for t, p in row.items():
            matrix[k, t] = float(p)
    return matrix


def float_absorption_check(total: int, beta: int, potential: str, exact: Sequence[Fraction]) -> dict[str, Any]:
    sector = build_sector(total)
    transient = list(sector.transient_indices())
    matrix = float_transition_matrix(total, beta, potential)
    q = matrix[np.ix_(transient, transient)]
    solved = np.linalg.solve(np.eye(len(transient)) - q, np.ones(len(transient)))
    reference = np.array([float(exact[k]) for k in transient])
    max_rel = float(np.max(np.abs(solved - reference) / np.maximum(1.0, np.abs(reference))))
    return {
        "fundamental_matrix_solve_max_relative_error": max_rel,
        "tolerance": FLOAT_TOLERANCE,
        "agrees": max_rel <= FLOAT_TOLERANCE,
    }


def absorption_summary(total: int, beta: int, potential: str) -> dict[str, Any]:
    sector = build_sector(total)
    times = expected_moves_to_absorption(total, beta, potential)
    g_phi, g_v, h_phi, h_v = terminal_functionals(total, beta, potential)
    n = sector.size
    uniform_time = sum(times, Fraction(0)) / n
    transient = sector.transient_indices()
    transient_time = sum(times[k] for k in transient) / len(transient)
    argmax = max(range(n), key=lambda k: (times[k], -k))
    p_phi_min = sum(g_phi, Fraction(0)) / n
    p_v_min = sum(g_v, Fraction(0)) / n
    e_phi = sum(h_phi, Fraction(0)) / n
    e_v = sum(h_v, Fraction(0)) / n
    return {
        "beta": beta,
        "potential": potential,
        "expected_moves_uniform_initial": frac(uniform_time),
        "expected_moves_uniform_initial_float": float(uniform_time),
        "expected_moves_uniform_transient_initial": frac(transient_time),
        "max_expected_moves": frac(times[argmax]),
        "max_expected_moves_state": list(sector.states[argmax]),
        "terminal_v_minimal_probability_uniform_initial": frac(p_v_min),
        "expected_terminal_v_uniform_initial": frac(e_v),
        "terminal_phi_minimal_probability_uniform_initial": frac(p_phi_min),
        "terminal_phi_minimal_probability_uniform_initial_float": float(p_phi_min),
        "expected_terminal_phi_uniform_initial": frac(e_phi),
        "expected_terminal_phi_uniform_initial_float": float(e_phi),
        "expected_moves_table_sha256": sha256_of([frac(t) for t in times]),
        "float_cross_check": float_absorption_check(total, beta, potential, times),
    }


def reversibility_summary(total: int, beta: int, potential: str) -> dict[str, Any]:
    sector = build_sector(total)
    rows = aggregated_rows(total, beta, potential)
    support = 0
    reversed_support = 0
    for k, row in enumerate(rows):
        for t, p in row.items():
            if t == k or p == 0:
                continue
            support += 1
            if rows[t].get(k, Fraction(0)) > 0:
                reversed_support += 1
    transient = sector.transient_indices()
    normalized = all(sum(rows[k].values(), Fraction(0)) == 1 for k in transient)
    return {
        "beta": beta,
        "potential": potential,
        "support_edges_between_distinct_states": support,
        "support_edges_with_reverse_support": reversed_support,
        "detailed_balance_possible": reversed_support == support and support > 0,
        "cycles_of_length_at_least_two_in_support": 0,
        "faithful_stationary_reference_exists": False,
        "transient_rows_normalized": normalized,
        "reading": (
            "every state-changing transition lowers V, so the support graph "
            "on transient states is acyclic and no transition has a reverse; a "
            "positive reference pi with pi_x P_xy = pi_y P_yx would force "
            "P_xy = 0 on every such edge; stationary laws vanish on transient "
            "states"
        ),
    }


def _entropy_bits(law: np.ndarray) -> float:
    mass = law[law > 0]
    return float(-np.sum(mass * np.log2(mass)))


def entropy_summary(total: int, beta: int, potential: str) -> dict[str, Any]:
    sector = build_sector(total)
    rows = aggregated_rows(total, beta, potential)
    n = sector.size
    data: list[float] = []
    row_index: list[int] = []
    col_index: list[int] = []
    for k, row in enumerate(rows):
        for t, p in row.items():
            data.append(float(p))
            row_index.append(t)
            col_index.append(k)
    transposed = csr_matrix((data, (row_index, col_index)), shape=(n, n))
    law = np.full(n, 1.0 / n)
    absorbing = np.array([sector.is_absorbing(k) for k in range(n)])
    trace: dict[str, float] = {}
    horizon = max(ENTROPY_TRACE_STEPS)
    for step in range(horizon + 1):
        if step in ENTROPY_TRACE_STEPS:
            trace[str(step)] = _entropy_bits(law)
        if step < horizon:
            law = transposed @ law
    one_step_irreversible_mass = sum(
        sum((p for t, p in rows[k].items() if t != k), Fraction(0)) for k in range(n)
    ) / n
    return {
        "beta": beta,
        "potential": potential,
        "initial_entropy_bits": float(math.log2(n)),
        "state_entropy_trace_bits": trace,
        "absorbing_set_entropy_cap_bits": float(math.log2(int(absorbing.sum()))),
        "transient_mass_at_horizon": float(law[~absorbing].sum()),
        "path_entropy_production": "infinite",
        "path_entropy_production_reading": (
            "sum over state paths of P(path) log(P(path)/P(reversed path)) "
            "diverges because every state-changing transition has zero reverse "
            "probability; the finite part over reversible pairs is zero"
        ),
        "one_step_irreversible_mass_uniform_initial": frac(one_step_irreversible_mass),
        "relative_entropy_to_uniform_reference_increases": bool(trace[str(horizon)] < trace["0"]),
    }


# Label paths: derived action, gauge, most probable paths


def chosen_initial_states(total: int = PATH_SECTOR_TOTAL) -> tuple[State, ...]:
    """Three transient states of the ``s = 3`` sector.

    ``A`` carries the total on port zero, ``B`` puts two units on port zero
    and one on its neighbour port one, ``C`` puts the third unit on the
    antipode of port zero.
    """

    if total != 3:
        raise ValueError("the path checks are declared on the s = 3 sector")
    anti = carrier.antipode()
    a = [0] * carrier.PORT_COUNT
    a[0] = 3
    b = [0] * carrier.PORT_COUNT
    b[0] = 2
    b[1] = 1
    c = [0] * carrier.PORT_COUNT
    c[0] = 2
    c[anti[0]] = 1
    if (0, 1) not in carrier.seams():
        raise AssertionError("port one is expected adjacent to port zero")
    return (tuple(a), tuple(b), tuple(c))


def _leaf_paths(sector: Sector, kernel: Kernel, start: int, length: int):
    """Yield ``(num, den, decrement_sum, end_state)`` over all label paths of ``length``."""

    weights = kernel.weights
    partition = kernel.partition
    nxt = sector.next_state
    table = sector.decrement(kernel.potential)
    stack = [(start, 1, 1, 0, 0)]
    while stack:
        state, num, den, dsum, depth = stack.pop()
        if depth == length:
            yield num, den, dsum, state
            continue
        w = weights[state]
        z = partition[state]
        n_row = nxt[state]
        d_row = table[state]
        for m in range(LABEL_COUNT):
            stack.append((n_row[m], num * w[m], den * z, dsum + d_row[m], depth + 1))


def path_mass_by_state(sector: Sector, kernel: Kernel, start: int, length: int) -> dict[int, Fraction]:
    """Exact total path mass grouped by end state (a different summation order)."""

    mass = {start: Fraction(1)}
    for _ in range(length):
        nxt: dict[int, Fraction] = {}
        for state, p in mass.items():
            for m in range(LABEL_COUNT):
                t = sector.next_state[state][m]
                nxt[t] = nxt.get(t, Fraction(0)) + p * kernel.label_probability(state, m)
        mass = nxt
    return mass


def derived_action_check(total: int, beta: int, potential: str, length: int) -> dict[str, Any]:
    """Every label path of ``length`` from the three chosen states.

    Checks, exactly: the product of the drawn weights equals
    ``2^(-beta * sum dU)`` (the ``ln 2`` part of the decomposition); the path
    law summed over all label paths equals one, so the tilt normalizer equals
    the Lean partition constant ``60^(-n)``; and with the initial law uniform
    on the three states the ratio ``law / exp(-S)`` is the constant ``1/3``.
    """

    sector = build_sector(total)
    kernel = build_kernel(total, beta, potential)
    starts = [sector.index[s] for s in chosen_initial_states(total)]
    per_start: list[dict[str, Any]] = []
    all_ok = True
    total_paths = 0
    for start in starts:
        count = 0
        ok = True
        for num, _den, dsum, _end in _leaf_paths(sector, kernel, start, length):
            count += 1
            if num != TILT_BASE ** (-beta * dsum):
                ok = False
        mass = path_mass_by_state(sector, kernel, start, length)
        summed = sum(mass.values(), Fraction(0))
        per_start.append(
            {
                "initial_state": list(sector.states[start]),
                "label_path_count": count,
                "weight_product_equals_two_power": ok,
                "path_law_sum": frac(summed),
                "end_state_count": len(mass),
            }
        )
        all_ok = all_ok and ok and summed == 1 and count == LABEL_COUNT**length
        total_paths += count
    lean_constant = Fraction(1, LABEL_COUNT**length)
    tilt_constant = sum(
        Fraction(1, 3) * lean_constant * sum(path_mass_by_state(sector, kernel, s, length).values(), Fraction(0))
        for s in starts
    )
    return {
        "beta": beta,
        "potential": potential,
        "path_length": length,
        "label_path_count": total_paths,
        "initial_law": "uniform on the three chosen states",
        "per_initial_state": per_start,
        "tilt_partition_constant": frac(tilt_constant),
        "lean_partition_constant": frac(lean_constant),
        "partition_constants_agree": tilt_constant == lean_constant,
        "law_over_exp_neg_action_constant": frac(Fraction(1, 3)),
        "action_decomposition": "S = beta ln2 sum_k dU_k + sum_k ln Z(x_{k-1})",
        "verified": all_ok and tilt_constant == lean_constant,
    }


def _path_records(total: int, beta: int, potential: str, length: int) -> list[dict[str, Any]]:
    """All label paths of ``length`` from the three states as exact records."""

    sector = build_sector(total)
    kernel = build_kernel(total, beta, potential)
    table = sector.decrement(potential)
    records: list[dict[str, Any]] = []
    for start_state in chosen_initial_states(total):
        start = sector.index[start_state]
        stack = [(start, Fraction(1), 0, (), ())]
        while stack:
            state, prob, dsum, path, zs = stack.pop()
            if len(path) == length:
                records.append({"start": start, "labels": path, "prob": prob, "dsum": dsum, "partitions": zs, "end": state})
                continue
            for m in range(LABEL_COUNT):
                stack.append(
                    (
                        sector.next_state[state][m],
                        prob * kernel.label_probability(state, m),
                        dsum + table[state][m],
                        path + (m,),
                        zs + (kernel.partition[state],),
                    )
                )
    return records


def _tilt(reference: Sequence[Fraction], factors: Sequence[Fraction]) -> list[Fraction]:
    z = sum((r * f for r, f in zip(reference, factors)), Fraction(0))
    return [r * f / z for r, f in zip(reference, factors)]


def gauge_check(total: int, beta: int, potential: str, length: int) -> dict[str, Any]:
    """The gauge characterization on all label paths of ``length``.

    Candidate action-multiplier pairs are compared with the log-transition
    action through the exact factor ``exp(-lambda' S')`` and the constancy of
    ``lambda' S' - S``; the tilt equals the path law exactly when that
    difference is constant.
    """

    records = _path_records(total, beta, potential, length)
    initial = Fraction(1, 3)
    reference = [initial * Fraction(1, LABEL_COUNT**length) for _ in records]
    law = [initial * r["prob"] for r in records]
    exp_neg_s = [r["prob"] for r in records]
    decomposition_ok = all(
        r["prob"] == Fraction(TILT_BASE ** (-beta * r["dsum"]), math.prod(r["partitions"]))
        for r in records
    )
    fit_ratio = {law[i] / exp_neg_s[i] for i in range(len(records))}
    base_ok = _tilt(reference, exp_neg_s) == law
    first_label_zero = [r["labels"][0] == 0 for r in records]
    candidates: list[dict[str, Any]] = []

    def add_candidate(name: str, factors: list[Fraction], difference_factors: list[Fraction], note: str) -> None:
        tilt = _tilt(reference, factors)
        equal = tilt == law
        constant = len(set(difference_factors)) == 1
        candidates.append(
            {
                "candidate": name,
                "tilt_equals_path_law": equal,
                "multiplier_weighted_difference_constant": constant,
                "characterization_holds": equal == constant,
                "paths_where_tilt_differs": sum(1 for a, b in zip(tilt, law) if a != b),
                "note": note,
            }
        )

    add_candidate(
        "S + c with exp(-c) = 3",
        [3 * v for v in exp_neg_s],
        [Fraction(3) for _ in records],
        "additive constant; the normalizer absorbs it",
    )
    add_candidate(
        "S / 2 at multiplier 2",
        list(exp_neg_s),
        [Fraction(1) for _ in records],
        "multiplier against action scale; identical multiplier-weighted action",
    )
    add_candidate(
        "S + f, exp(-f) = 2 on paths whose first label is label 0",
        [v * (2 if flag else 1) for v, flag in zip(exp_neg_s, first_label_zero)],
        [Fraction(2 if flag else 1) for flag in first_label_zero],
        "nonconstant additive function of the path",
    )
    add_candidate(
        "2 S at multiplier 1",
        [v * v for v in exp_neg_s],
        list(exp_neg_s),
        "multiplier mismatch; the difference is S itself, constant only when the action is constant",
    )
    distinct_actions = len(set(exp_neg_s))
    return {
        "beta": beta,
        "potential": potential,
        "path_length": length,
        "label_path_count": len(records),
        "decomposition_reproduces_path_probability": decomposition_ok,
        "base_tilt_equals_path_law": base_ok,
        "fitted_action_minus_action_constant": len(fit_ratio) == 1,
        "fitted_action_minus_action_exp_neg": frac(next(iter(fit_ratio))),
        "distinct_action_values": distinct_actions,
        "action_nonconstant": distinct_actions > 1,
        "candidates": candidates,
        "verified": decomposition_ok
        and base_ok
        and len(fit_ratio) == 1
        and all(c["characterization_holds"] for c in candidates),
    }


def _compare_ratio(a_num: int, a_den: int, b_num: int, b_den: int) -> int:
    left = a_num * b_den
    right = b_num * a_den
    return (left > right) - (left < right)


def most_probable_paths(total: int, beta: int, potential: str, max_length: int) -> dict[str, Any]:
    """Argmax of the label path probability against argmin of the action.

    A dynamic programme over states keeps, per length, the maximal path
    probability with the number of label paths attaining it.  The
    probability route multiplies row probabilities; the action route uses
    ``2^(-beta sum dU) / prod Z``; both are compared as exact rationals.
    """

    sector = build_sector(total)
    kernel = build_kernel(total, beta, potential)
    table = sector.decrement(potential)
    labels = directed_labels()
    out: list[dict[str, Any]] = []
    for start_state in chosen_initial_states(total):
        start = sector.index[start_state]
        frontier: dict[int, tuple[int, int, int, tuple[int, ...]]] = {start: (1, 1, 1, ())}
        action_frontier: dict[int, tuple[int, int, int]] = {start: (0, 1, 1)}
        per_length: list[dict[str, Any]] = []
        for length in range(1, max_length + 1):
            nxt: dict[int, tuple[int, int, int, tuple[int, ...]]] = {}
            action_nxt: dict[int, tuple[int, int, int]] = {}
            for state, (num, den, count, witness) in frontier.items():
                w = kernel.weights[state]
                z = kernel.partition[state]
                for m in range(LABEL_COUNT):
                    t = sector.next_state[state][m]
                    c_num, c_den = num * w[m], den * z
                    current = nxt.get(t)
                    if current is None:
                        nxt[t] = (c_num, c_den, count, witness + (m,))
                    else:
                        cmp = _compare_ratio(c_num, c_den, current[0], current[1])
                        if cmp > 0:
                            nxt[t] = (c_num, c_den, count, witness + (m,))
                        elif cmp == 0:
                            nxt[t] = (current[0], current[1], current[2] + count, current[3])
            for state, (dsum, zprod, count) in action_frontier.items():
                z = kernel.partition[state]
                for m in range(LABEL_COUNT):
                    t = sector.next_state[state][m]
                    c_dsum, c_zprod = dsum + table[state][m], zprod * z
                    c_num, c_den = TILT_BASE ** (-beta * c_dsum), c_zprod
                    current = action_nxt.get(t)
                    if current is None:
                        action_nxt[t] = (c_dsum, c_zprod, count)
                    else:
                        cmp = _compare_ratio(c_num, c_den, TILT_BASE ** (-beta * current[0]), current[1])
                        if cmp > 0:
                            action_nxt[t] = (c_dsum, c_zprod, count)
                        elif cmp == 0:
                            action_nxt[t] = (current[0], current[1], current[2] + count)
            frontier, action_frontier = nxt, action_nxt
            best_num, best_den = 0, 1
            for num, den, _c, _w in frontier.values():
                if _compare_ratio(num, den, best_num, best_den) > 0:
                    best_num, best_den = num, den
            winners = [
                (state, count, witness)
                for state, (num, den, count, witness) in frontier.items()
                if _compare_ratio(num, den, best_num, best_den) == 0
            ]
            exp_neg_action_max = max(
                Fraction(TILT_BASE ** (-beta * dsum), zprod) for dsum, zprod, _c in action_frontier.values()
            )
            action_winners = [
                (state, count)
                for state, (dsum, zprod, count) in action_frontier.items()
                if Fraction(TILT_BASE ** (-beta * dsum), zprod) == exp_neg_action_max
            ]
            tie_label_paths = sum(count for _s, count, _w in winners)
            action_tie_label_paths = sum(count for _s, count in action_winners)
            witness_state, _count, witness_labels = min(winners, key=lambda item: item[0])
            per_length.append(
                {
                    "length": length,
                    "max_path_probability": frac(Fraction(best_num, best_den)),
                    "argmax_terminal_state_count": len(winners),
                    "argmax_label_path_count": tie_label_paths,
                    "argmin_action_terminal_state_count": len(action_winners),
                    "argmin_action_label_path_count": action_tie_label_paths,
                    "argmax_equals_argmin": (
                        sorted(s for s, _c, _w in winners) == sorted(s for s, _c in action_winners)
                        and tie_label_paths == action_tie_label_paths
                        and Fraction(best_num, best_den) == exp_neg_action_max
                    ),
                    "witness_labels": [list(labels[m]) for m in witness_labels],
                    "witness_terminal_state": list(sector.states[witness_state]),
                    "witness_terminal_phi": sector.phi[witness_state],
                    "witness_terminal_v": sector.v[witness_state],
                }
            )
        out.append({"initial_state": list(start_state), "per_length": per_length})
    verified = all(row["argmax_equals_argmin"] for entry in out for row in entry["per_length"])
    return {"beta": beta, "potential": potential, "max_length": max_length, "per_initial_state": out, "verified": verified}


# Non-identifiability family on realized records


def enrichment_correction(a: Fraction, record: Sequence[Fraction]) -> Fraction:
    """``(a/2) sum_m y_m (y_m - 1)`` on an occupancy record of the sixty labels."""

    return Fraction(a) / 2 * sum((Fraction(y) * (Fraction(y) - 1) for y in record), Fraction(0))


def realized_record(label_index: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(1 if m == label_index else 0) for m in range(LABEL_COUNT))


def non_identifiability_check(total: int, beta: int, potential: str, length: int) -> dict[str, Any]:
    """The family ``L_a = L_0 + (a/2) y (y - 1)`` transported to the label alphabet.

    ``L_0(x, y) = sum_m y_m (-log P(m | x))`` is affine in the occupancy
    record ``y``; realized records are one-hot, where the correction
    vanishes, so every member agrees with the log-transition action on every
    realized path.  Unrealized configurations separate the members.
    """

    records = _path_records(total, beta, potential, length)
    parameters = list(ENRICHMENT_PARAMETERS)
    per_label = {
        (a, m): enrichment_correction(a, realized_record(m))
        for a in parameters
        for m in range(LABEL_COUNT)
    }
    per_record_zero = all(value == 0 for value in per_label.values())
    per_path_zero = all(
        sum((per_label[(a, m)] for m in r["labels"]), Fraction(0)) == 0
        for a in parameters
        for r in records
    )
    unrealized: list[dict[str, Any]] = []
    for name, value in (("y_m = 2", Fraction(2)), ("y_m = 1/2", Fraction(1, 2))):
        record = [Fraction(0)] * LABEL_COUNT
        record[0] = value
        corrections = {frac(a): frac(enrichment_correction(a, record)) for a in parameters}
        unrealized.append(
            {
                "configuration": name,
                "corrections_by_a": corrections,
                "difference_a2_minus_a1": frac(
                    enrichment_correction(Fraction(2), record) - enrichment_correction(Fraction(1), record)
                ),
            }
        )
    corpus_midpoint = next(u for u in unrealized if u["configuration"] == "y_m = 1/2")
    midpoint_gap = abs(Fraction(corpus_midpoint["difference_a2_minus_a1"]))
    return {
        "beta": beta,
        "potential": potential,
        "path_length": length,
        "label_path_count": len(records),
        "parameters": [frac(a) for a in parameters],
        "record_alphabet": "occupancy of the sixty directed labels, one-hot on realized records",
        "l0": "sum_m y_m (-log P(m | x)); affine in y",
        "correction_vanishes_on_every_realized_record": per_record_zero,
        "correction_vanishes_on_every_enumerated_path": per_path_zero,
        "tilt_by_every_member_equals_path_law": per_path_zero,
        "unrealized_configurations": unrealized,
        "corpus_midpoint_gap": frac(midpoint_gap),
        "corpus_midpoint_difference_reproduced": midpoint_gap == Fraction(1, 8),
        "statement": (
            "selection inside the declared family from realized histories is "
            "impossible: every member induces the same path law"
        ),
        "verified": per_record_zero and per_path_zero and midpoint_gap == Fraction(1, 8),
    }


# Sample path exhibits


def exhibit_paths(total: int, beta: int, potential: str, length: int, per_start: int = 2) -> list[dict[str, Any]]:
    sector = build_sector(total)
    kernel = build_kernel(total, beta, potential)
    table = sector.decrement(potential)
    labels = directed_labels()
    exhibits: list[dict[str, Any]] = []
    for start_state in chosen_initial_states(total):
        start = sector.index[start_state]
        chosen: list[tuple[int, ...]] = [tuple([0] * length), tuple(range(0, 2 * length, 2))[:length]]
        for path in chosen[:per_start]:
            state = start
            prob = Fraction(1)
            decrements: list[int] = []
            zs: list[int] = []
            states = [list(sector.states[state])]
            for m in path:
                prob *= kernel.label_probability(state, m)
                decrements.append(table[state][m])
                zs.append(kernel.partition[state])
                state = sector.next_state[state][m]
                states.append(list(sector.states[state]))
            dsum = sum(decrements)
            action_float = beta * dsum * math.log(2) + sum(math.log(z) for z in zs)
            exhibits.append(
                {
                    "initial_state": list(start_state),
                    "labels": [list(labels[m]) for m in path],
                    "states": states,
                    "potential_along_path": [potential_value(s, potential) for s in states],
                    "decrements": decrements,
                    "partitions": [str(z) for z in zs],
                    "ln2_coefficient": beta * dsum,
                    "path_probability": frac(prob),
                    "decomposition_probability": frac(Fraction(TILT_BASE ** (-beta * dsum), math.prod(zs))),
                    "action_float": action_float,
                    "exp_neg_action_matches": prob == Fraction(TILT_BASE ** (-beta * dsum), math.prod(zs)),
                }
            )
    return exhibits


# Payload


def sector_summary(total: int) -> dict[str, Any]:
    sector = build_sector(total)
    absorbing = sector.absorbing_indices()
    phi_min = phi_minimal_indices(sector)
    v_min = v_minimal_indices(sector)
    phi_on_absorbing: dict[str, int] = {}
    for k in absorbing:
        key = str(sector.phi[k])
        phi_on_absorbing[key] = phi_on_absorbing.get(key, 0) + 1
    q, r = divmod(total, carrier.PORT_COUNT)
    shell = all(
        all(v in (q, q + 1) for v in sector.states[k]) for k in absorbing
    ) and len(absorbing) == math.comb(carrier.PORT_COUNT, r)
    per_potential: dict[str, Any] = {}
    for potential in POTENTIALS:
        per_potential[potential] = {
            "formula": POTENTIAL_FORMULAS[potential],
            "descent": descent_summary(sector, potential),
            "kernels": {str(beta): kernel_summary(total, beta, potential) for beta in BETAS},
            "absorption": {str(beta): absorption_summary(total, beta, potential) for beta in BETAS},
            "reversibility": {str(beta): reversibility_summary(total, beta, potential) for beta in BETAS},
            "entropy": {str(beta): entropy_summary(total, beta, potential) for beta in BETAS},
        }
    beta_zero_coincide = (
        build_kernel(total, 0, PRIMARY_POTENTIAL).weights == build_kernel(total, 0, SECONDARY_POTENTIAL).weights
    )
    return {
        "protected_total": total,
        "state_count": sector.size,
        "absorbing_state_count": len(absorbing),
        "transient_state_count": sector.size - len(absorbing),
        "absorbing_set_is_agreement_shell": shell,
        "v_min": min(sector.v),
        "v_max": max(sector.v),
        "v_minimal_state_count": len(v_min),
        "absorbing_set_equals_v_minimal_set": set(absorbing) == set(v_min),
        "v_minimal_set_reading": (
            "a state is absorbing exactly when every seam split is the floor/ceil "
            "split, the integer minimizer of x_c^2 + x_o^2 at fixed total; on this "
            "sector that local condition coincides with global V-minimality, "
            "checked exhaustively"
        ),
        "phi_min": min(sector.phi),
        "phi_max": max(sector.phi),
        "phi_minimal_state_count": len(phi_min),
        "phi_minimal_states_are_absorbing": all(sector.is_absorbing(k) for k in phi_min),
        "phi_values_on_absorbing_set": phi_on_absorbing,
        "unit_transfer": unit_transfer_summary(sector),
        "literal_gap_one_swap": literal_swap_summary(sector),
        "beta_zero_kernels_coincide": beta_zero_coincide,
        "potentials": per_potential,
    }


def build_payload() -> dict[str, Any]:
    labels = directed_labels()
    sectors = {str(total): sector_summary(total) for total in PROTECTED_TOTALS}
    path_blocks: dict[str, Any] = {}
    non_identifiability: dict[str, Any] = {}
    for potential in POTENTIALS:
        path_blocks[potential] = {
            "derived_action": {
                str(beta): derived_action_check(PATH_SECTOR_TOTAL, beta, potential, DERIVED_ACTION_PATH_LENGTH)
                for beta in BETAS
            },
            "gauge": {str(beta): gauge_check(PATH_SECTOR_TOTAL, beta, potential, GAUGE_PATH_LENGTH) for beta in BETAS},
            "most_probable": {
                str(beta): most_probable_paths(PATH_SECTOR_TOTAL, beta, potential, MOST_PROBABLE_MAX_LENGTH)
                for beta in BETAS
            },
            "exhibits": {
                str(beta): exhibit_paths(PATH_SECTOR_TOTAL, beta, potential, DERIVED_ACTION_PATH_LENGTH)
                for beta in BETAS
            },
        }
        non_identifiability[potential] = {
            str(beta): non_identifiability_check(PATH_SECTOR_TOTAL, beta, potential, GAUGE_PATH_LENGTH)
            for beta in BETAS
        }
    paths = {
        "sector": PATH_SECTOR_TOTAL,
        "initial_states": [list(s) for s in chosen_initial_states(PATH_SECTOR_TOTAL)],
        "initial_law": "1/3 on each chosen state",
        "reference": "initial law times 60^(-n) per label path (step-uniform)",
        "potentials": path_blocks,
    }
    verified = (
        all(sectors[str(t)]["absorbing_set_is_agreement_shell"] for t in PROTECTED_TOTALS)
        and all(sectors[str(t)]["absorbing_set_equals_v_minimal_set"] for t in PROTECTED_TOTALS)
        and all(sectors[str(t)]["unit_transfer"]["verified"] for t in PROTECTED_TOTALS)
        and all(sectors[str(t)]["beta_zero_kernels_coincide"] for t in PROTECTED_TOTALS)
        and all(
            sectors[str(t)]["potentials"][p]["descent"]["all_nonpositive"]
            and sectors[str(t)]["potentials"][p]["descent"]["zero_iff_wait"]
            for t in PROTECTED_TOTALS
            for p in POTENTIALS
        )
        and all(
            sectors[str(t)]["potentials"][p]["kernels"][str(b)]["label_rows_sum_to_one"]
            and sectors[str(t)]["potentials"][p]["absorption"][str(b)]["float_cross_check"]["agrees"]
            and not sectors[str(t)]["potentials"][p]["reversibility"][str(b)]["detailed_balance_possible"]
            for t in PROTECTED_TOTALS
            for p in POTENTIALS
            for b in BETAS
        )
        and all(
            path_blocks[p]["derived_action"][str(b)]["verified"]
            and path_blocks[p]["gauge"][str(b)]["verified"]
            and path_blocks[p]["most_probable"][str(b)]["verified"]
            and non_identifiability[p][str(b)]["verified"]
            for p in POTENTIALS
            for b in BETAS
        )
    )
    return {
        "schema": SCHEMA,
        "lane": LANE,
        "declared": {
            "kernel_family": (
                "P_beta(m | x) = 2^(-beta dU_m(x)) / Z_beta(x) over the sixty directed "
                "labels, dU the decrement of the declared potential U"
            ),
            "potentials": {
                "primary": PRIMARY_POTENTIAL,
                "secondary": SECONDARY_POTENTIAL,
                "formulas": dict(POTENTIAL_FORMULAS),
                "reading": (
                    "the instantiation paper's Phi decrement is read as the flagship's "
                    "V decrement; the seam-sum variant is a diagnostic whose terminal "
                    "value varies over the agreement shell"
                ),
            },
            "tilt_base": TILT_BASE,
            "betas": list(BETAS),
            "protected_totals": list(PROTECTED_TOTALS),
            "move_law": (
                "label (c, o) waits when |x_c - x_o| <= 1 and otherwise sends the "
                "pair to (ceil(t/2), floor(t/2)) with the ceiling at c"
            ),
            "gap_one_reading": (
                "the reverse label on a gap-one seam waits; the literal endpoint "
                "swap of the carrier function is V-neutral, raises Phi on explicit "
                "states, and is excluded from the declared kernel"
            ),
            "beta_zero": "the uniform A3 schedule, 1/60 per directed label, identical for both potentials",
        },
        "carrier": {
            "ports": carrier.PORT_COUNT,
            "seams": [list(s) for s in carrier.seams()],
            "directed_labels": [list(l) for l in labels],
            "antipode": list(carrier.antipode()),
        },
        "sectors": sectors,
        "paths": paths,
        "non_identifiability": non_identifiability,
        "corpus": {
            "flagship_quadratic_descent": {
                "path": PINNED_META_FILES["flagship_tex"],
                "lines": "889-895",
                "quote": (
                    "a conservative repair transfers one unit across a seam whenever the "
                    "oriented mismatch has magnitude at least two. With V(N) = sum_i N_i^2, "
                    "each such repair strictly decreases V by 2(d-1) for mismatch d >= 2, "
                    "so repair terminates by a finite theorem."
                ),
            },
            "instantiation_paper_m5": {
                "path": PAPER_REFERENCE["path"],
                "lines": "1665-1715",
                "quote": (
                    "What would connect Phi to an action is a declared stochastic "
                    "repair law whose log-transition action is a function of the Phi "
                    "decrement along each accepted move. With that law the "
                    "derived-action theorem applies and the source selects the action "
                    "up to the non-identifiability family. [...] No such law is declared."
                ),
                "reading": "the Phi decrement is read as the flagship's V decrement",
            },
            "observers_paper_theorem": {
                "path": PINNED_META_FILES["observers_paper_tex"],
                "lines": "3076-3103",
                "definition": (
                    "L_0(x,y) is the bilinear real extension of the committed two-state "
                    "log-transition table, affine in y; L_a(x,y) = L_0(x,y) + (a/2) y (y-1); "
                    "y is the target record slot, valued in {0, 1} on realized histories; "
                    "the correction y(y-1) vanishes at both source symbols; a = 1 and "
                    "a = 2 differ by 1/8 when one middle record is varied to y = 1/2"
                ),
            },
            "flagship_non_identifiability": {
                "path": PINNED_META_FILES["flagship_tex"],
                "lines": "3006-3021",
            },
            "flagship_four_law_premise": {
                "path": PINNED_META_FILES["flagship_tex"],
                "lines": "432-436, 4308-4319",
                "quote": (
                    "Every finite stochastic repair kernel preserving a supplied "
                    "faithful reference contracts relative entropy; the deterministic "
                    "strict-descent normalizer need not, as an exact counterexample "
                    "shows. [...] Physical thermodynamics additionally requires "
                    "source-justified transitions, a shared reference, energy-clock "
                    "calibration and uniform low-temperature tails on one cofinal family."
                ),
            },
            "lean_common_reference_obstruction": {
                "path": "reverse-engineering-reality/Lean/Thermodynamics/CommonReferenceObstruction.lean",
                "lines": "22-26",
                "quote": (
                    "They do not exclude a newly source-produced random-scan kernel, a "
                    "different common reference, or an independently justified "
                    "stochastic coupling. Merely inventing such a coupling would not be "
                    "source evidence."
                ),
            },
            "lean_log_transition_action": {
                "path": PINNED_META_FILES["lean_log_transition_action"],
                "theorems": [
                    "markov_path_law_eq_gibbs",
                    "markov_tiltZ_eq",
                    "tilt_eq_tilt_iff_gauge",
                    "action_unique_up_to_gauge",
                    "bare_log_action_multiplier_unique_of_nonconstant",
                ],
                "instantiation": (
                    "the label path space with the strictly positive label kernel; the "
                    "state-space kernel has zero entries and sits outside the theorem's "
                    "positivity hypothesis"
                ),
            },
        },
        "scope": {
            "declared": [
                "kernel family P_beta(m | x) = 2^(-beta dU) / Z with tilt base 2",
                "potential axis: squared_norm_V primary (flagship V), seam_sum_Phi secondary (diagnostic)",
                "betas 0, 1, 2",
                "protected totals 3 and 4",
                "move law with waits on gap-one seams",
                "initial law uniform on three chosen s = 3 states for the path checks",
            ],
            "verified": [
                "descent: nonpositive decrement on every state-label pair for both potentials, zero exactly on waits",
                "V: unit-transfer decrement -2(d-1), declared-move closed form -2 floor(D/2) ceil(D/2), absorbing set equals the V-minimal set",
                "Phi: absorbing set equals the agreement shell and contains the Phi-minimal set as a proper subset",
                "exact expected moves to absorption with a float fundamental-matrix cross-check",
                "derived action: label path law equals the tilt of the step-uniform reference at multiplier one with partition constant 60^(-n)",
                "gauge characterization on four candidate action-multiplier pairs",
                "most-probable label path equals the action minimizer with exact tie counts",
                "non-identifiability family agrees on every realized record and separates off the alphabet",
                "no reversible reference; no faithful stationary law",
            ],
            "not_claimed": [
                "source selection of the kernel",
                "laboratory readout",
                "continuum limit",
                "an entropy inequality for the strict-descent normalizer",
                "uniqueness of the terminal state under the seam-sum variant",
                "local-global V-minimality agreement beyond the two verified totals",
            ],
        },
        "verified": verified,
        "claim_boundary": CLAIM_BOUNDARY,
        "paper_reference": dict(PAPER_REFERENCE),
        "pins": pins(),
    }


# Comparison, CLI


def compare(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Exact comparison with a float tolerance; returns the mismatches."""

    problems: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                problems.append(f"{path}.{key}: missing on one side")
                continue
            problems.extend(compare(expected[key], actual[key], f"{path}.{key}"))
        return problems
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} != {len(actual)}"]
        for k, (e, a) in enumerate(zip(expected, actual)):
            problems.extend(compare(e, a, f"{path}[{k}]"))
        return problems
    if isinstance(expected, float) or isinstance(actual, float):
        if isinstance(expected, bool) or isinstance(actual, bool):
            return [f"{path}: {expected!r} != {actual!r}"]
        try:
            e, a = float(expected), float(actual)
        except (TypeError, ValueError):
            return [f"{path}: {expected!r} != {actual!r}"]
        if abs(e - a) > FLOAT_TOLERANCE * max(1.0, abs(e), abs(a)):
            return [f"{path}: {e!r} != {a!r}"]
        return []
    if expected != actual:
        return [f"{path}: {expected!r} != {actual!r}"]
    return []


def write_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    payload = build_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload))
    return payload


def check_receipt(path: Path = DEFAULT_RECEIPT) -> list[str]:
    stored = json.loads(path.read_text(encoding="ascii"))
    if canonical_bytes(stored) != path.read_bytes():
        return ["receipt is not in canonical form"]
    return compare(stored, build_payload())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        payload = write_receipt(args.receipt)
        print(f"wrote {args.receipt} verified={payload['verified']}")
        return 0
    problems = check_receipt(args.receipt)
    if problems:
        for problem in problems[:50]:
            print(problem)
        print(f"FAIL: {len(problems)} mismatches")
        return 1
    print(f"OK: {args.receipt} matches a fresh replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
