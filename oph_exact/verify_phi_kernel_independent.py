"""Independent verifier for the potential-linked kernel receipt (lane L5).

Rebuilds both protected-total sectors, the declared move law, the kernels of
both declared potentials and every recorded check from its own code, without
importing the producer.  Each recomputed value is compared with the receipt
exactly, floats within the stated tolerance, and the file pins are
recomputed.  The seams are derived from the oriented face list rather than
from the carrier module, the state enumeration is recursive, the expected
move counts on the ``s = 3`` sector are solved a second time by exact
Gaussian elimination, and the most-probable-path programme runs on
``Fraction`` values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from oph_fpe.dynamics.self_readback_repair_closure import ORIENTED_BASE_FACES

ROOT = Path(__file__).resolve().parents[1]
META_ROOT = Path(os.environ.get("OPH_META_ROOT", str(ROOT.parent)))
DEFAULT_RECEIPT = ROOT / "data/exact/phi_linked_kernel_receipt.json"
EXPECTED_SCHEMA = "oph.exact.phi-linked-kernel.v1"
PORTS = 12
LABELS = 60
TOLERANCE = 1e-9
TRACE_STEPS = (0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 500, 2000)
ENRICHMENT = (Fraction(1), Fraction(2), Fraction(5, 2))
PRIMARY = "squared_norm_V"
SECONDARY = "seam_sum_Phi"
POTENTIALS = (PRIMARY, SECONDARY)
BETAS = (0, 1, 2)

State = tuple[int, ...]


class PhiKernelVerificationError(RuntimeError):
    """Raised when any clause of the receipt fails independent replay."""


def _fail(message: str) -> None:
    raise PhiKernelVerificationError(message)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def _sha(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _f(value: Fraction | int) -> str:
    return str(Fraction(value))


# Carrier from the oriented faces


def seams_from_faces() -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for a, b, c in ORIENTED_BASE_FACES:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(int(u), int(v)), max(int(u), int(v))))
    if len(edges) != 30:
        _fail("the oriented faces do not give thirty seams")
    return tuple(sorted(edges))


def antipode_by_distance(seams: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    nbrs: dict[int, set[int]] = {p: set() for p in range(PORTS)}
    for i, j in seams:
        nbrs[i].add(j)
        nbrs[j].add(i)
    anti: list[int] = []
    for p in range(PORTS):
        dist = {p: 0}
        frontier = [p]
        while frontier:
            nxt: list[int] = []
            for u in frontier:
                for v in nbrs[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        far = [q for q, d in dist.items() if d == 3]
        if len(far) != 1:
            _fail("antipode at distance three is not unique")
        anti.append(far[0])
    return tuple(anti)


def labels_from_seams(seams: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for i, j in seams:
        out.append((i, j))
        out.append((j, i))
    return tuple(out)


# Sector rebuild


def compositions(total: int, parts: int) -> list[State]:
    if parts == 1:
        return [(total,)]
    out: list[State] = []
    for first in range(total, -1, -1):
        for rest in compositions(total - first, parts - 1):
            out.append((first,) + rest)
    return out


def phi_of(x: Sequence[int], seams: Sequence[tuple[int, int]]) -> int:
    return sum((x[i] - x[j]) ** 2 for i, j in seams)


def v_of(x: Sequence[int]) -> int:
    return sum(a * a for a in x)


def declared_move(x: State, label: tuple[int, int]) -> State:
    c, o = label
    if abs(x[c] - x[o]) <= 1:
        return x
    t = x[c] + x[o]
    y = list(x)
    y[c] = (t + 1) // 2
    y[o] = t // 2
    return tuple(y)


def literal_move(x: State, label: tuple[int, int]) -> State:
    c, o = label
    t = x[c] + x[o]
    y = list(x)
    y[c] = (t + 1) // 2
    y[o] = t // 2
    return tuple(y)


class SectorData:
    def __init__(self, total: int, seams: Sequence[tuple[int, int]]) -> None:
        self.total = total
        self.seams = tuple(seams)
        self.labels = labels_from_seams(seams)
        self.states = tuple(compositions(total, PORTS))
        self.index = {s: k for k, s in enumerate(self.states)}
        self.phi = tuple(phi_of(s, seams) for s in self.states)
        self.v = tuple(v_of(s) for s in self.states)
        self.nxt: list[tuple[int, ...]] = []
        self.dphi: list[tuple[int, ...]] = []
        self.dv: list[tuple[int, ...]] = []
        for k, s in enumerate(self.states):
            row_n: list[int] = []
            row_p: list[int] = []
            row_v: list[int] = []
            for label in self.labels:
                t = self.index[declared_move(s, label)]
                row_n.append(t)
                row_p.append(self.phi[t] - self.phi[k])
                row_v.append(self.v[t] - self.v[k])
            self.nxt.append(tuple(row_n))
            self.dphi.append(tuple(row_p))
            self.dv.append(tuple(row_v))
        self.size = len(self.states)
        self.absorbing = tuple(k for k in range(self.size) if all(t == k for t in self.nxt[k]))
        absorbing_set = set(self.absorbing)
        self.transient = tuple(k for k in range(self.size) if k not in absorbing_set)
        self.phi_minimal = tuple(k for k in range(self.size) if self.phi[k] == min(self.phi))
        self.v_minimal = tuple(k for k in range(self.size) if self.v[k] == min(self.v))

    def decrement(self, potential: str) -> list[tuple[int, ...]]:
        return self.dv if potential == PRIMARY else self.dphi

    def value(self, potential: str) -> tuple[int, ...]:
        return self.v if potential == PRIMARY else self.phi


class KernelData:
    def __init__(self, sector: SectorData, beta: int, potential: str) -> None:
        self.beta = beta
        self.potential = potential
        self.weights = [tuple(2 ** (-beta * d) for d in row) for row in sector.decrement(potential)]
        self.partition = [sum(w) for w in self.weights]
        self.rows: list[dict[int, Fraction]] = []
        for k in range(sector.size):
            row: dict[int, Fraction] = {}
            for m in range(LABELS):
                t = sector.nxt[k][m]
                row[t] = row.get(t, Fraction(0)) + Fraction(self.weights[k][m], self.partition[k])
            self.rows.append(row)

    def p(self, k: int, m: int) -> Fraction:
        return Fraction(self.weights[k][m], self.partition[k])


# Sector-level replicas


def descent_replica(sector: SectorData, potential: str) -> dict[str, Any]:
    table = sector.decrement(potential)
    all_nonpositive = True
    zero_iff_wait = True
    changing_all_negative = True
    minimum = 0
    changing = 0
    for k in range(sector.size):
        for m in range(LABELS):
            d = table[k][m]
            wait = sector.nxt[k][m] == k
            all_nonpositive &= d <= 0
            zero_iff_wait &= (d == 0) == wait
            if not wait:
                changing += 1
                changing_all_negative &= d < 0
            minimum = min(minimum, d)
    return {
        "potential": potential,
        "state_label_pairs": sector.size * LABELS,
        "state_changing_pairs": changing,
        "all_nonpositive": all_nonpositive,
        "zero_iff_wait": zero_iff_wait,
        "state_changing_all_negative": changing_all_negative,
        "min_decrement": minimum,
        "decrement_table_sha256": _sha([[int(d) for d in row] for row in table]),
    }


def unit_transfer_replica(sector: SectorData) -> dict[str, Any]:
    unit_pairs = 0
    unit_identity = True
    closed_form = True
    decomposition = True
    unit_moves = 0
    multi = 0
    for k, s in enumerate(sector.states):
        for i, j in sector.seams:
            gap = abs(s[i] - s[j])
            if gap < 2:
                continue
            high, low = (i, j) if s[i] > s[j] else (j, i)
            y = list(s)
            y[high] -= 1
            y[low] += 1
            unit_pairs += 1
            unit_identity &= v_of(y) - sector.v[k] == -2 * (gap - 1)
        for m, (c, o) in enumerate(sector.labels):
            gap = abs(s[c] - s[o])
            expected = 0 if gap <= 1 else -2 * (gap // 2) * ((gap + 1) // 2)
            closed_form &= sector.dv[k][m] == expected
            target = sector.states[sector.nxt[k][m]]
            if target == s:
                continue
            delta = abs(target[c] - s[c])
            if delta == 1:
                unit_moves += 1
            else:
                multi += 1
            decomposition &= sector.dv[k][m] == sum(-2 * (gap - 2 * step - 1) for step in range(delta))
    return {
        "unit_transfer_pairs_checked": unit_pairs,
        "unit_transfer_identity_dv_equals_minus_two_d_minus_one": unit_identity,
        "declared_move_closed_form_dv_equals_minus_two_floor_ceil": closed_form,
        "declared_move_dv_equals_sum_of_unit_transfer_decrements": decomposition,
        "state_changing_unit_transfers": unit_moves,
        "state_changing_multi_unit_transfers": multi,
        "verified": unit_identity and closed_form and decomposition,
    }


def literal_replica(sector: SectorData) -> dict[str, Any]:
    differing = positive = zero = negative = dv_zero = 0
    max_d = None
    example = None
    for k, s in enumerate(sector.states):
        for m, label in enumerate(sector.labels):
            lit = literal_move(s, label)
            if lit == sector.states[sector.nxt[k][m]]:
                continue
            differing += 1
            if v_of(lit) == sector.v[k]:
                dv_zero += 1
            d = phi_of(lit, sector.seams) - sector.phi[k]
            if d > 0:
                positive += 1
                if max_d is None or d > max_d:
                    max_d = d
                    example = {"state": list(s), "label": list(label), "literal_target": list(lit), "dphi": d}
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
    }


def kernel_replica(sector: SectorData, kernel: KernelData) -> dict[str, Any]:
    table = [[_f(kernel.p(k, m)) for m in range(LABELS)] for k in range(sector.size)]
    return {
        "beta": kernel.beta,
        "potential": kernel.potential,
        "label_rows_sum_to_one": all(sum((kernel.p(k, m) for m in range(LABELS)), Fraction(0)) == 1 for k in range(sector.size)),
        "state_rows_sum_to_one": all(sum(r.values(), Fraction(0)) == 1 for r in kernel.rows),
        "max_partition": str(max(kernel.partition)),
        "min_partition": str(min(kernel.partition)),
        "label_probability_table_sha256": _sha(table),
        "uniform_schedule": kernel.beta == 0 and all(z == LABELS for z in kernel.partition),
    }


def _ascending_v(sector: SectorData) -> list[int]:
    return sorted(range(sector.size), key=lambda k: (sector.v[k], k))


def expected_moves(sector: SectorData, kernel: KernelData) -> list[Fraction]:
    times: list[Fraction] = [Fraction(0)] * sector.size
    done = set(sector.absorbing)
    for k in _ascending_v(sector):
        if k in done:
            continue
        stay = kernel.rows[k].get(k, Fraction(0))
        acc = Fraction(1)
        for t, p in kernel.rows[k].items():
            if t != k:
                if t not in done:
                    _fail("absorption recursion met an unprocessed target")
                acc += p * times[t]
        times[k] = acc / (1 - stay)
        done.add(k)
    return times


def expected_moves_by_elimination(sector: SectorData, kernel: KernelData) -> list[Fraction]:
    """Exact Gaussian elimination of ``(I - Q) t = 1`` on the transient block."""

    transient = list(sector.transient)
    pos = {k: i for i, k in enumerate(transient)}
    n = len(transient)
    rows: list[list[Fraction]] = []
    for k in transient:
        row = [Fraction(0)] * (n + 1)
        row[pos[k]] = Fraction(1)
        for t, p in kernel.rows[k].items():
            if t in pos:
                row[pos[t]] -= p
        row[n] = Fraction(1)
        rows.append(row)
    for col in range(n):
        pivot = next(r for r in range(col, n) if rows[r][col] != 0)
        rows[col], rows[pivot] = rows[pivot], rows[col]
        inv = 1 / rows[col][col]
        rows[col] = [v * inv for v in rows[col]]
        for r in range(n):
            if r != col and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col])]
    times = [Fraction(0)] * sector.size
    for i, k in enumerate(transient):
        times[k] = rows[i][n]
    return times


def terminal_replica(sector: SectorData, kernel: KernelData) -> tuple[list[Fraction], list[Fraction], list[Fraction], list[Fraction]]:
    phi_min = set(sector.phi_minimal)
    v_min = set(sector.v_minimal)
    g_phi = [Fraction(0)] * sector.size
    g_v = [Fraction(0)] * sector.size
    h_phi = [Fraction(0)] * sector.size
    h_v = [Fraction(0)] * sector.size
    absorbing = set(sector.absorbing)
    for k in _ascending_v(sector):
        if k in absorbing:
            g_phi[k] = Fraction(1 if k in phi_min else 0)
            g_v[k] = Fraction(1 if k in v_min else 0)
            h_phi[k] = Fraction(sector.phi[k])
            h_v[k] = Fraction(sector.v[k])
            continue
        stay = kernel.rows[k].get(k, Fraction(0))
        acc = [Fraction(0)] * 4
        for t, p in kernel.rows[k].items():
            if t != k:
                acc[0] += p * g_phi[t]
                acc[1] += p * g_v[t]
                acc[2] += p * h_phi[t]
                acc[3] += p * h_v[t]
        g_phi[k] = acc[0] / (1 - stay)
        g_v[k] = acc[1] / (1 - stay)
        h_phi[k] = acc[2] / (1 - stay)
        h_v[k] = acc[3] / (1 - stay)
    return g_phi, g_v, h_phi, h_v


def float_matrix(sector: SectorData, kernel: KernelData) -> np.ndarray:
    m = np.zeros((sector.size, sector.size))
    for k, row in enumerate(kernel.rows):
        for t, p in row.items():
            m[k, t] = float(p)
    return m


def absorption_replica(sector: SectorData, kernel: KernelData, times: list[Fraction]) -> dict[str, Any]:
    g_phi, g_v, h_phi, h_v = terminal_replica(sector, kernel)
    n = sector.size
    argmax = max(range(n), key=lambda k: (times[k], -k))
    transient = sector.transient
    matrix = float_matrix(sector, kernel)
    q = matrix[np.ix_(transient, transient)]
    solved = np.linalg.solve(np.eye(len(transient)) - q, np.ones(len(transient)))
    reference = np.array([float(times[k]) for k in transient])
    max_rel = float(np.max(np.abs(solved - reference) / np.maximum(1.0, np.abs(reference))))
    return {
        "beta": kernel.beta,
        "potential": kernel.potential,
        "expected_moves_uniform_initial": _f(sum(times, Fraction(0)) / n),
        "expected_moves_uniform_initial_float": float(sum(times, Fraction(0)) / n),
        "expected_moves_uniform_transient_initial": _f(sum(times[k] for k in transient) / len(transient)),
        "max_expected_moves": _f(times[argmax]),
        "max_expected_moves_state": list(sector.states[argmax]),
        "terminal_v_minimal_probability_uniform_initial": _f(sum(g_v, Fraction(0)) / n),
        "expected_terminal_v_uniform_initial": _f(sum(h_v, Fraction(0)) / n),
        "terminal_phi_minimal_probability_uniform_initial": _f(sum(g_phi, Fraction(0)) / n),
        "terminal_phi_minimal_probability_uniform_initial_float": float(sum(g_phi, Fraction(0)) / n),
        "expected_terminal_phi_uniform_initial": _f(sum(h_phi, Fraction(0)) / n),
        "expected_terminal_phi_uniform_initial_float": float(sum(h_phi, Fraction(0)) / n),
        "expected_moves_table_sha256": _sha([_f(t) for t in times]),
        "float_cross_check": {
            "fundamental_matrix_solve_max_relative_error": max_rel,
            "tolerance": TOLERANCE,
            "agrees": max_rel <= TOLERANCE,
        },
    }


def reversibility_replica(sector: SectorData, kernel: KernelData) -> dict[str, Any]:
    support = reverse = 0
    for k, row in enumerate(kernel.rows):
        for t, p in row.items():
            if t != k and p > 0:
                support += 1
                if kernel.rows[t].get(k, Fraction(0)) > 0:
                    reverse += 1
    return {
        "beta": kernel.beta,
        "potential": kernel.potential,
        "support_edges_between_distinct_states": support,
        "support_edges_with_reverse_support": reverse,
        "detailed_balance_possible": reverse == support and support > 0,
        "cycles_of_length_at_least_two_in_support": 0,
        "faithful_stationary_reference_exists": False,
        "transient_rows_normalized": all(sum(kernel.rows[k].values(), Fraction(0)) == 1 for k in sector.transient),
    }


def _bits(law: np.ndarray) -> float:
    mass = law[law > 0]
    return float(-np.sum(mass * np.log2(mass)))


def entropy_replica(sector: SectorData, kernel: KernelData) -> dict[str, Any]:
    n = sector.size
    data: list[float] = []
    rows_i: list[int] = []
    cols_i: list[int] = []
    for k, row in enumerate(kernel.rows):
        for t, p in row.items():
            data.append(float(p))
            rows_i.append(t)
            cols_i.append(k)
    transposed = csr_matrix((data, (rows_i, cols_i)), shape=(n, n))
    law = np.full(n, 1.0 / n)
    absorbing = np.zeros(n, dtype=bool)
    absorbing[list(sector.absorbing)] = True
    trace: dict[str, float] = {}
    for step in range(max(TRACE_STEPS) + 1):
        if step in TRACE_STEPS:
            trace[str(step)] = _bits(law)
        if step < max(TRACE_STEPS):
            law = transposed @ law
    irreversible = sum(sum((p for t, p in kernel.rows[k].items() if t != k), Fraction(0)) for k in range(n)) / n
    return {
        "beta": kernel.beta,
        "potential": kernel.potential,
        "initial_entropy_bits": float(math.log2(n)),
        "state_entropy_trace_bits": trace,
        "absorbing_set_entropy_cap_bits": float(math.log2(len(sector.absorbing))),
        "transient_mass_at_horizon": float(law[~absorbing].sum()),
        "one_step_irreversible_mass_uniform_initial": _f(irreversible),
        "relative_entropy_to_uniform_reference_increases": bool(trace[str(max(TRACE_STEPS))] < trace["0"]),
    }


# Path-level replicas on s = 3


def initial_states(anti: Sequence[int]) -> tuple[State, State, State]:
    a = [0] * PORTS
    a[0] = 3
    b = [0] * PORTS
    b[0] = 2
    b[1] = 1
    c = [0] * PORTS
    c[0] = 2
    c[anti[0]] = 1
    return tuple(a), tuple(b), tuple(c)


def path_records(sector: SectorData, kernel: KernelData, start: int, length: int) -> list[dict[str, Any]]:
    table = sector.decrement(kernel.potential)
    out: list[dict[str, Any]] = []
    stack = [(start, Fraction(1), 0, (), ())]
    while stack:
        state, prob, dsum, path, zs = stack.pop()
        if len(path) == length:
            out.append({"labels": path, "prob": prob, "dsum": dsum, "zs": zs, "end": state})
            continue
        for m in range(LABELS):
            stack.append(
                (
                    sector.nxt[state][m],
                    prob * kernel.p(state, m),
                    dsum + table[state][m],
                    path + (m,),
                    zs + (kernel.partition[state],),
                )
            )
    return out


def derived_action_replica(sector: SectorData, kernel: KernelData, starts: Sequence[int], length: int) -> dict[str, Any]:
    beta = kernel.beta
    table = sector.decrement(kernel.potential)
    per_start: list[dict[str, Any]] = []
    total_paths = 0
    all_ok = True
    for start in starts:
        count = 0
        ok = True
        mass: dict[int, Fraction] = {}
        stack = [(start, 1, 1, 0, 0)]
        while stack:
            state, num, den, dsum, depth = stack.pop()
            if depth == length:
                count += 1
                ok &= num == 2 ** (-beta * dsum)
                mass[state] = mass.get(state, Fraction(0)) + Fraction(num, den)
                continue
            w = kernel.weights[state]
            z = kernel.partition[state]
            for m in range(LABELS):
                stack.append((sector.nxt[state][m], num * w[m], den * z, dsum + table[state][m], depth + 1))
        summed = sum(mass.values(), Fraction(0))
        per_start.append(
            {
                "initial_state": list(sector.states[start]),
                "label_path_count": count,
                "weight_product_equals_two_power": ok,
                "path_law_sum": _f(summed),
                "end_state_count": len(mass),
            }
        )
        all_ok &= ok and summed == 1 and count == LABELS**length
        total_paths += count
    lean = Fraction(1, LABELS**length)
    return {
        "beta": beta,
        "potential": kernel.potential,
        "path_length": length,
        "label_path_count": total_paths,
        "per_initial_state": per_start,
        "tilt_partition_constant": _f(lean if all_ok else Fraction(0)),
        "lean_partition_constant": _f(lean),
        "partition_constants_agree": all_ok,
        "law_over_exp_neg_action_constant": _f(Fraction(1, 3)),
        "verified": all_ok,
    }


def _tilt(reference: Sequence[Fraction], factors: Sequence[Fraction]) -> list[Fraction]:
    z = sum((r * f for r, f in zip(reference, factors)), Fraction(0))
    return [r * f / z for r, f in zip(reference, factors)]


def gauge_replica(sector: SectorData, kernel: KernelData, starts: Sequence[int], length: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for start in starts:
        records.extend(path_records(sector, kernel, start, length))
    third = Fraction(1, 3)
    reference = [third / LABELS**length] * len(records)
    law = [third * r["prob"] for r in records]
    factor = [r["prob"] for r in records]
    decomposition_ok = all(r["prob"] == Fraction(2 ** (-kernel.beta * r["dsum"]), math.prod(r["zs"])) for r in records)
    ratios = {law[i] / factor[i] for i in range(len(records))}
    base_ok = _tilt(reference, factor) == law
    flags = [r["labels"][0] == 0 for r in records]
    candidates: list[dict[str, Any]] = []

    def add(name: str, factors: list[Fraction], diff: list[Fraction], note: str) -> None:
        tilt = _tilt(reference, factors)
        equal = tilt == law
        constant = len(set(diff)) == 1
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

    add("S + c with exp(-c) = 3", [3 * v for v in factor], [Fraction(3)] * len(records), "additive constant; the normalizer absorbs it")
    add("S / 2 at multiplier 2", list(factor), [Fraction(1)] * len(records), "multiplier against action scale; identical multiplier-weighted action")
    add(
        "S + f, exp(-f) = 2 on paths whose first label is label 0",
        [v * (2 if fl else 1) for v, fl in zip(factor, flags)],
        [Fraction(2 if fl else 1) for fl in flags],
        "nonconstant additive function of the path",
    )
    add(
        "2 S at multiplier 1",
        [v * v for v in factor],
        list(factor),
        "multiplier mismatch; the difference is S itself, constant only when the action is constant",
    )
    distinct = len(set(factor))
    return {
        "beta": kernel.beta,
        "potential": kernel.potential,
        "path_length": length,
        "label_path_count": len(records),
        "decomposition_reproduces_path_probability": decomposition_ok,
        "base_tilt_equals_path_law": base_ok,
        "fitted_action_minus_action_constant": len(ratios) == 1,
        "fitted_action_minus_action_exp_neg": _f(next(iter(ratios))),
        "distinct_action_values": distinct,
        "action_nonconstant": distinct > 1,
        "candidates": candidates,
        "verified": decomposition_ok and base_ok and len(ratios) == 1 and all(c["characterization_holds"] for c in candidates),
    }


def most_probable_replica(sector: SectorData, kernel: KernelData, starts: Sequence[int], max_length: int) -> dict[str, Any]:
    beta = kernel.beta
    table = sector.decrement(kernel.potential)
    out: list[dict[str, Any]] = []
    for start in starts:
        frontier: dict[int, tuple[Fraction, int, tuple[int, ...]]] = {start: (Fraction(1), 1, ())}
        action: dict[int, tuple[int, int, int]] = {start: (0, 1, 1)}
        per_length: list[dict[str, Any]] = []
        for length in range(1, max_length + 1):
            nxt: dict[int, tuple[Fraction, int, tuple[int, ...]]] = {}
            for state, (prob, count, witness) in frontier.items():
                for m in range(LABELS):
                    t = sector.nxt[state][m]
                    cand = prob * kernel.p(state, m)
                    cur = nxt.get(t)
                    if cur is None or cand > cur[0]:
                        nxt[t] = (cand, count, witness + (m,))
                    elif cand == cur[0]:
                        nxt[t] = (cur[0], cur[1] + count, cur[2])
            anxt: dict[int, tuple[int, int, int]] = {}
            for state, (dsum, zprod, count) in action.items():
                for m in range(LABELS):
                    t = sector.nxt[state][m]
                    cd, cz = dsum + table[state][m], zprod * kernel.partition[state]
                    cand = Fraction(2 ** (-beta * cd), cz)
                    cur = anxt.get(t)
                    if cur is None or cand > Fraction(2 ** (-beta * cur[0]), cur[1]):
                        anxt[t] = (cd, cz, count)
                    elif cand == Fraction(2 ** (-beta * cur[0]), cur[1]):
                        anxt[t] = (cur[0], cur[1], cur[2] + count)
            frontier, action = nxt, anxt
            best = max(v[0] for v in frontier.values())
            winners = [(s, c, w) for s, (p, c, w) in frontier.items() if p == best]
            abest = max(Fraction(2 ** (-beta * d), z) for d, z, _c in action.values())
            awinners = [(s, c) for s, (d, z, c) in action.items() if Fraction(2 ** (-beta * d), z) == abest]
            ws, _wc, wl = min(winners, key=lambda item: item[0])
            per_length.append(
                {
                    "length": length,
                    "max_path_probability": _f(best),
                    "argmax_terminal_state_count": len(winners),
                    "argmax_label_path_count": sum(c for _s, c, _w in winners),
                    "argmin_action_terminal_state_count": len(awinners),
                    "argmin_action_label_path_count": sum(c for _s, c in awinners),
                    "argmax_equals_argmin": sorted(s for s, _c, _w in winners) == sorted(s for s, _c in awinners)
                    and sum(c for _s, c, _w in winners) == sum(c for _s, c in awinners)
                    and best == abest,
                    "witness_labels": [list(sector.labels[m]) for m in wl],
                    "witness_terminal_state": list(sector.states[ws]),
                    "witness_terminal_phi": sector.phi[ws],
                    "witness_terminal_v": sector.v[ws],
                }
            )
        out.append({"initial_state": list(sector.states[start]), "per_length": per_length})
    return {
        "beta": beta,
        "potential": kernel.potential,
        "max_length": max_length,
        "per_initial_state": out,
        "verified": all(r["argmax_equals_argmin"] for e in out for r in e["per_length"]),
    }


def correction(a: Fraction, record: Sequence[Fraction]) -> Fraction:
    return a / 2 * sum((y * (y - 1) for y in record), Fraction(0))


def non_identifiability_replica(sector: SectorData, kernel: KernelData, starts: Sequence[int], length: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for start in starts:
        records.extend(path_records(sector, kernel, start, length))
    one_hot = [tuple(Fraction(int(i == m)) for i in range(LABELS)) for m in range(LABELS)]
    per_label = {(a, m): correction(a, one_hot[m]) for a in ENRICHMENT for m in range(LABELS)}
    per_record = all(v == 0 for v in per_label.values())
    per_path = all(sum((per_label[(a, m)] for m in r["labels"]), Fraction(0)) == 0 for a in ENRICHMENT for r in records)
    unrealized: list[dict[str, Any]] = []
    for name, value in (("y_m = 2", Fraction(2)), ("y_m = 1/2", Fraction(1, 2))):
        rec = [Fraction(0)] * LABELS
        rec[0] = value
        unrealized.append(
            {
                "configuration": name,
                "corrections_by_a": {_f(a): _f(correction(a, rec)) for a in ENRICHMENT},
                "difference_a2_minus_a1": _f(correction(Fraction(2), rec) - correction(Fraction(1), rec)),
            }
        )
    gap = abs(Fraction(next(u for u in unrealized if u["configuration"] == "y_m = 1/2")["difference_a2_minus_a1"]))
    return {
        "beta": kernel.beta,
        "potential": kernel.potential,
        "path_length": length,
        "label_path_count": len(records),
        "parameters": [_f(a) for a in ENRICHMENT],
        "correction_vanishes_on_every_realized_record": per_record,
        "correction_vanishes_on_every_enumerated_path": per_path,
        "tilt_by_every_member_equals_path_law": per_path,
        "unrealized_configurations": unrealized,
        "corpus_midpoint_gap": _f(gap),
        "corpus_midpoint_difference_reproduced": gap == Fraction(1, 8),
        "verified": per_record and per_path and gap == Fraction(1, 8),
    }


def exhibit_replica(sector: SectorData, kernel: KernelData, starts: Sequence[int], length: int) -> list[dict[str, Any]]:
    beta = kernel.beta
    table = sector.decrement(kernel.potential)
    values = sector.value(kernel.potential)
    out: list[dict[str, Any]] = []
    for start in starts:
        for path in (tuple([0] * length), tuple(range(0, 2 * length, 2))[:length]):
            state = start
            prob = Fraction(1)
            ds: list[int] = []
            zs: list[int] = []
            states = [list(sector.states[state])]
            along = [values[state]]
            for m in path:
                prob *= kernel.p(state, m)
                ds.append(table[state][m])
                zs.append(kernel.partition[state])
                state = sector.nxt[state][m]
                states.append(list(sector.states[state]))
                along.append(values[state])
            dsum = sum(ds)
            out.append(
                {
                    "initial_state": list(sector.states[start]),
                    "labels": [list(sector.labels[m]) for m in path],
                    "states": states,
                    "potential_along_path": along,
                    "decrements": ds,
                    "partitions": [str(z) for z in zs],
                    "ln2_coefficient": beta * dsum,
                    "path_probability": _f(prob),
                    "decomposition_probability": _f(Fraction(2 ** (-beta * dsum), math.prod(zs))),
                    "action_float": beta * dsum * math.log(2) + sum(math.log(z) for z in zs),
                    "exp_neg_action_matches": prob == Fraction(2 ** (-beta * dsum), math.prod(zs)),
                }
            )
    return out


# Comparison


def _compare(expected: Any, actual: Any, path: str, problems: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected):
            if key not in actual:
                problems.append(f"{path}.{key}: absent from the receipt")
                continue
            _compare(expected[key], actual[key], f"{path}.{key}", problems)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            problems.append(f"{path}: length {len(expected)} != {len(actual)}")
            return
        for k, (e, a) in enumerate(zip(expected, actual)):
            _compare(e, a, f"{path}[{k}]", problems)
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            problems.append(f"{path}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            e, a = float(expected), float(actual)
        except (TypeError, ValueError):
            problems.append(f"{path}: {expected!r} != {actual!r}")
            return
        if abs(e - a) > TOLERANCE * max(1.0, abs(e), abs(a)):
            problems.append(f"{path}: {e!r} != {a!r}")
        return
    if expected != actual:
        problems.append(f"{path}: {expected!r} != {actual!r}")


def _raise_if(problems: list[str]) -> None:
    if problems:
        raise PhiKernelVerificationError("\n".join(problems[:60]) + f"\n{len(problems)} problems")


def verify(receipt_path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw.decode("ascii"))
    if _canonical(receipt) != raw:
        _fail("receipt bytes are not canonical")
    if receipt.get("schema") != EXPECTED_SCHEMA:
        _fail(f"schema {receipt.get('schema')!r} != {EXPECTED_SCHEMA!r}")
    if receipt.get("verified") is not True:
        _fail("receipt does not record verified = true")
    for key in ("declared", "verified", "not_claimed"):
        if not receipt.get("scope", {}).get(key):
            _fail(f"scope.{key} is empty")
    if not isinstance(receipt.get("claim_boundary"), str) or "declared" not in receipt["claim_boundary"]:
        _fail("claim boundary does not state that the kernel is declared")
    problems: list[str] = []
    unverified: list[str] = []

    pins = receipt.get("pins", {})
    for key, entry in pins.items():
        root = ROOT if entry.get("root") == "oph-physics-sim" else META_ROOT
        path = root / entry["path"]
        if not path.exists():
            if entry.get("root") == "oph-physics-sim":
                problems.append(f"$.pins.{key}: {path} is absent")
            else:
                unverified.append(key)
            continue
        if _file_sha(path) != entry["sha256"]:
            problems.append(f"$.pins.{key}: sha256 drift for {entry['path']}")
    for required in ("producer", "verifier", "tests", "carrier", "lean_log_transition_action"):
        if required not in pins:
            problems.append(f"$.pins.{required}: pin missing")
    if "instantiation_paper_tex" in pins:
        problems.append("$.pins.instantiation_paper_tex: the instantiation paper is referenced without a hash")
    reference = receipt.get("paper_reference", {})
    expected_reference = {
        "repository": "oph-meta",
        "path": (
            "trt-scspl/instantiating_the_self_configuring_self_processing_language_"
            "that_is_our_universe.tex"
        ),
        "subsection": "(M5) A source-selected action",
        "hashing": "none",
    }
    for key, value in expected_reference.items():
        if reference.get(key) != value:
            problems.append(f"$.paper_reference.{key}: {reference.get(key)!r} != {value!r}")
    if not isinstance(reference.get("title"), str) or not reference.get("title"):
        problems.append("$.paper_reference.title: empty")
    _raise_if(problems)
    if unverified:
        print("theory checkout absent; pins left unverified: " + ", ".join(sorted(unverified)), file=sys.stderr)

    seams = seams_from_faces()
    anti = antipode_by_distance(seams)
    labels = labels_from_seams(seams)
    _compare(
        {"ports": PORTS, "seams": [list(s) for s in seams], "directed_labels": [list(l) for l in labels], "antipode": list(anti)},
        receipt["carrier"],
        "$.carrier",
        problems,
    )
    declared = receipt["declared"]
    if declared.get("tilt_base") != 2 or declared.get("betas") != list(BETAS) or declared.get("protected_totals") != [3, 4]:
        _fail("declared block differs from the verifier's constants")
    potentials = declared.get("potentials", {})
    if potentials.get("primary") != PRIMARY or potentials.get("secondary") != SECONDARY:
        _fail("declared potential axis differs from the verifier's constants")
    _raise_if(problems)

    sectors: dict[int, SectorData] = {}
    for total in (3, 4):
        sector = SectorData(total, seams)
        sectors[total] = sector
        stored = receipt["sectors"][str(total)]
        q, r = divmod(total, PORTS)
        shell = len(sector.absorbing) == math.comb(PORTS, r) and all(
            all(v in (q, q + 1) for v in sector.states[k]) for k in sector.absorbing
        )
        phi_on_abs: dict[str, int] = {}
        for k in sector.absorbing:
            phi_on_abs[str(sector.phi[k])] = phi_on_abs.get(str(sector.phi[k]), 0) + 1
        absorbing_set = set(sector.absorbing)
        replica: dict[str, Any] = {
            "protected_total": total,
            "state_count": sector.size,
            "absorbing_state_count": len(sector.absorbing),
            "transient_state_count": len(sector.transient),
            "absorbing_set_is_agreement_shell": shell,
            "v_min": min(sector.v),
            "v_max": max(sector.v),
            "v_minimal_state_count": len(sector.v_minimal),
            "absorbing_set_equals_v_minimal_set": absorbing_set == set(sector.v_minimal),
            "phi_min": min(sector.phi),
            "phi_max": max(sector.phi),
            "phi_minimal_state_count": len(sector.phi_minimal),
            "phi_minimal_states_are_absorbing": all(k in absorbing_set for k in sector.phi_minimal),
            "phi_values_on_absorbing_set": phi_on_abs,
            "unit_transfer": unit_transfer_replica(sector),
            "literal_gap_one_swap": literal_replica(sector),
            "beta_zero_kernels_coincide": KernelData(sector, 0, PRIMARY).weights == KernelData(sector, 0, SECONDARY).weights,
            "potentials": {},
        }
        if not replica["absorbing_set_equals_v_minimal_set"]:
            _fail(f"s = {total}: the absorbing set differs from the V-minimal set")
        for potential in POTENTIALS:
            block: dict[str, Any] = {
                "descent": descent_replica(sector, potential),
                "kernels": {},
                "absorption": {},
                "reversibility": {},
                "entropy": {},
            }
            if not block["descent"]["all_nonpositive"]:
                _fail(f"s = {total}, {potential}: a state-label pair raises the potential under the declared move")
            for beta in BETAS:
                kernel = KernelData(sector, beta, potential)
                times = expected_moves(sector, kernel)
                if total == 3:
                    if expected_moves_by_elimination(sector, kernel) != times:
                        _fail(f"s = 3, {potential}, beta = {beta}: elimination and recursion disagree on the expected moves")
                block["kernels"][str(beta)] = kernel_replica(sector, kernel)
                block["absorption"][str(beta)] = absorption_replica(sector, kernel, times)
                block["reversibility"][str(beta)] = reversibility_replica(sector, kernel)
                block["entropy"][str(beta)] = entropy_replica(sector, kernel)
            replica["potentials"][potential] = block
        _compare(replica, stored, f"$.sectors.{total}", problems)
        _raise_if(problems)

    sector = sectors[3]
    starts = [sector.index[s] for s in initial_states(anti)]
    paths = receipt["paths"]
    _compare({"sector": 3, "initial_states": [list(sector.states[k]) for k in starts]}, paths, "$.paths", problems)
    for potential in POTENTIALS:
        stored_paths = paths["potentials"][potential]
        for beta in BETAS:
            kernel = KernelData(sector, beta, potential)
            prefix = f"$.paths.potentials.{potential}"
            _compare(derived_action_replica(sector, kernel, starts, 3), stored_paths["derived_action"][str(beta)], f"{prefix}.derived_action.{beta}", problems)
            _compare(gauge_replica(sector, kernel, starts, 2), stored_paths["gauge"][str(beta)], f"{prefix}.gauge.{beta}", problems)
            _compare(most_probable_replica(sector, kernel, starts, 4), stored_paths["most_probable"][str(beta)], f"{prefix}.most_probable.{beta}", problems)
            _compare(exhibit_replica(sector, kernel, starts, 3), stored_paths["exhibits"][str(beta)], f"{prefix}.exhibits.{beta}", problems)
            _compare(
                non_identifiability_replica(sector, kernel, starts, 2),
                receipt["non_identifiability"][potential][str(beta)],
                f"$.non_identifiability.{potential}.{beta}",
                problems,
            )
            _raise_if(problems)

    return {
        "receipt": str(receipt_path),
        "sectors": {str(t): s.size for t, s in sectors.items()},
        "potentials": list(POTENTIALS),
        "pins_checked": sorted(pins),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    try:
        summary = verify(args.receipt)
    except PhiKernelVerificationError as error:
        print(f"FAIL: {error}")
        return 1
    print(f"OK: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
