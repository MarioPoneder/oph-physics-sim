"""Closure loop at carrier scale: recovered specification, constructed realization, inhabited structure.

The closure hypothesis of the flagship states that the specification observers
recover from their records, the realization they construct from that
specification, and the structure they inhabit return the same invariant
quantities.  This module runs that condition on one exact carrier.

Three stages with a firewall between them:

* Stage A (generator).  ``run_source`` runs a carrier under a repair law with
  the uniform A3 seam schedule from seeded integer loads and emits only an
  event log: the port count, the initial loads, one record per completed event
  listing the two endpoint readings before and after, the final loads, and a
  probe log (the readback after ``2n`` synchronous expectation steps from each
  one-hot state, at declared resolution).  No seam list, group, or law
  description enters the log.
* Stage B (recovery).  ``recover_specification`` is a fixed algorithm that
  reads only the log.  It recovers the port count, the seam set, the inverse
  port pairing, the incidence automorphism group and its orientation
  preserving subgroup, the repair rule class, the schedule law, the
  normalized response Gram with its rank, the Lie type dimension count of the
  recovered rotation group, the descent law and the terminal law.  Its output
  is the recovered specification with an invariant vector.
* Stage C (instantiation).  ``instantiate`` builds a carrier from the
  recovered specification alone and Stage A runs it with a fresh seed.

Stage B applied to the source log yields the inhabited-structure invariants;
Stage B applied to the instantiated run yields the constructed-realization
invariants.  Equality of the two vectors, iterated once more, is the receipt
``CLOSURE_FIXED_POINT_AT_CARRIER_SCALE``.  Negative controls (tetrahedron,
octahedron, integer law, overwrite law) and the isolated federation of twenty
carriers run through the same loop.

Exactness: loads and event readings are rationals (dyadic under the seam
mean law), the probe readback is the exact expectation readback rounded to a
declared number of significant digits, the schedule statistic is an exact
rational, the band characters and determinants live in ``Q(sqrt 5)``.

Descent potential: the seam Laplacian form ``sum_seams (x_i - x_j)^2`` is
not monotone under a single seam-mean event (an endpoint moved toward its
partner can move away from its other neighbours).  The declared descent
potential is the centered squared norm ``V(x) = |x - mean(x) 1|^2``, which
every conservative seam retraction decreases strictly whenever the multiset
of readings changes.  Increases of the Laplacian form are reported as an
informational count.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations, combinations_with_replacement
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import chi2 as _chi2

from oph_exact import carrier as _carrier

SCHEMA = "oph.exact.closure-loop.v1"
LOG_SCHEMA = "oph.exact.closure-loop.event-log.v1"
ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "data" / "exact" / "closure_loop_receipt.json"
LOG_DIR = ROOT / "data" / "exact" / "closure_loop_logs"
PRODUCER_PATH = Path(__file__).resolve()
VERIFIER_PATH = ROOT / "oph_exact" / "verify_closure_loop_independent.py"
TEST_PATH = ROOT / "tests" / "test_exact_closure_loop.py"
CARRIER_PATH = ROOT / "oph_exact" / "carrier.py"

FIXED_POINT_RECEIPT = "CLOSURE_FIXED_POINT_AT_CARRIER_SCALE"
STATUS = (
    "CLOSURE_FIXED_POINT_AT_CARRIER_SCALE_ATTAINED_FOR_THE_CANONICAL_SOURCE__"
    "NEGATIVE_CONTROLS_DISTINGUISHED__GLUED_FEDERATION_WORK_IN_PROGRESS"
)

# Declared parameters (generator side).
PROBE_STEPS: tuple[int, ...] = (1, 5, 30, 100, 300)
READBACK_DIGITS = 120
EVENTS_PER_SEAM = 12
FEDERATION_EVENTS_PER_SEAM = 20
LOAD_RANGE = 100
FEDERATION_CARRIERS = 20

# Declared parameters (recovery side; part of the fixed algorithm).
REPLAY_SCHEDULES = 8
REPLAY_EVENTS_PER_SEAM = 32
REPLAY_SEED_BASE = 7919
CHI_SQUARE_SIGNIFICANCE = 1e-3
TIE_Z_THRESHOLD = 3.29  # two-sided normal threshold at significance 1e-3
GRAM_RANK_TOLERANCE = 1e-9
GRAM_RESOLUTION_DIGITS = 12  # the centered kernel must carry this many significant digits
TERMINAL_TOLERANCE = Fraction(1, 10**9)
PROBE_PREDICTION_TOLERANCE_MARGIN = 3  # relative tolerance 10^-(readback_digits - margin)
LIE_SIMPLE_DIMENSIONS: tuple[int, ...] = (3, 8, 10)
EIGENVALUE_CLUSTER_TOLERANCE = 1e-6

RULE_SEAM_MEAN = "seam_mean"
RULE_INTEGER = "integer_nearest_agreement"
RULE_OVERWRITE = "overwrite"
RULE_UNCLASSIFIED = "unclassified"

POTENTIAL_REFERENCE = (
    "flagship from_observer_consensus_to_standard_physics.tex, integer record keeping paragraph "
    "(lines 889-895 at RER r2039): V(N) = sum_i N_i^2 decreases by exactly 2(d-1) per conservative "
    "unit transfer across a seam with oriented mismatch d >= 2, and the confluence theorem's exact "
    "quadratic descent is this V. On a conservative law the centered squared norm equals V minus the "
    "conserved (sum_i x_i)^2 / ports, so both potentials have the same decrements; the seam-mean event "
    "with mismatch d decreases both by d^2 / 2."
)

LOOP_SEEDS: dict[str, int] = {
    "icosahedron_seam_mean": 20260909,
    "icosahedron_integer_nearest_agreement": 20260300,
    "icosahedron_overwrite": 20260400,
    "tetrahedron_seam_mean": 20260100,
    "octahedron_seam_mean": 20260200,
    "federation_isolated_20": 20260500,
}
LOOP_ITERATIONS: dict[str, int] = {
    "icosahedron_seam_mean": 2,
    "icosahedron_integer_nearest_agreement": 2,
    "icosahedron_overwrite": 2,
    "tetrahedron_seam_mean": 2,
    "octahedron_seam_mean": 2,
    "federation_isolated_20": 1,
}


# ==========================================================================
# Canonical JSON and hashing
# ==========================================================================


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def round_floats(value: Any) -> Any:
    """Round every float to 1e-9 (and normalize negative zero) before writing."""

    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite float in receipt")
        return round(value, 9) + 0.0
    if isinstance(value, dict):
        return {str(k): round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v) for v in value]
    return value


# ==========================================================================
# Exact value encoding shared by generator and recovery
# ==========================================================================


def encode_value(value: Fraction) -> str:
    return str(Fraction(value))


def parse_value(text: str) -> Fraction:
    return Fraction(text)


def readback_string(numerator: int, denominator: int, digits: int) -> str:
    """Round ``numerator/denominator`` to ``digits`` significant decimal digits.

    Integer arithmetic only; the result ``<mantissa>E<exponent>`` parses back
    through ``fractions.Fraction``.
    """

    if numerator == 0:
        return "0"
    sign = "-" if numerator < 0 else ""
    num = abs(numerator)
    den = denominator
    estimate = int((num.bit_length() - den.bit_length()) * 0.30102999566398)
    shift = digits - 1 - estimate
    lower = 10 ** (digits - 1)
    upper = 10**digits
    while True:
        if shift >= 0:
            divisor = den
            q, r = divmod(num * 10**shift, divisor)
        else:
            divisor = den * 10 ** (-shift)
            q, r = divmod(num, divisor)
        if q >= upper:
            shift -= 1
            continue
        if q < lower:
            shift += 1
            continue
        break
    twice = 2 * r
    if twice > divisor or (twice == divisor and q % 2 == 1):
        q += 1
        if q == upper:
            q //= 10
            shift -= 1
    return f"{sign}{q}E{-shift}"


# ==========================================================================
# Q(sqrt 5) arithmetic used by the recovery (self-contained)
# ==========================================================================

Q5 = tuple[Fraction, Fraction]
Q5_ZERO: Q5 = (Fraction(0), Fraction(0))
Q5_ONE: Q5 = (Fraction(1), Fraction(0))


def q5(a: int | Fraction, b: int | Fraction = 0) -> Q5:
    return (Fraction(a), Fraction(b))


def q5_add(x: Q5, y: Q5) -> Q5:
    return (x[0] + y[0], x[1] + y[1])


def q5_sub(x: Q5, y: Q5) -> Q5:
    return (x[0] - y[0], x[1] - y[1])


def q5_mul(x: Q5, y: Q5) -> Q5:
    return (x[0] * y[0] + 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def q5_inv(x: Q5) -> Q5:
    norm = x[0] * x[0] - 5 * x[1] * x[1]
    if norm == 0:
        raise ZeroDivisionError("zero element of Q(sqrt5)")
    return (x[0] / norm, -x[1] / norm)


def q5_float(x: Q5) -> float:
    return float(x[0]) + float(x[1]) * 5.0**0.5


def q5_str(x: Q5) -> list[str]:
    return [str(x[0]), str(x[1])]


def q5_matmul(x: Sequence[Sequence[Q5]], y: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    n, k, m = len(x), len(y), len(y[0])
    out = [[Q5_ZERO for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for t in range(k):
            xit = x[i][t]
            if xit == Q5_ZERO:
                continue
            row = y[t]
            for j in range(m):
                out[i][j] = q5_add(out[i][j], q5_mul(xit, row[j]))
    return out


def q5_nullity(matrix: Sequence[Sequence[Q5]]) -> int:
    """Exact nullity over Q(sqrt 5) by Gaussian elimination."""

    rows = [list(r) for r in matrix]
    n_rows, n_cols = len(rows), len(rows[0])
    rank = 0
    col = 0
    while rank < n_rows and col < n_cols:
        pivot = next((r for r in range(rank, n_rows) if rows[r][col] != Q5_ZERO), None)
        if pivot is None:
            col += 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inv = q5_inv(rows[rank][col])
        rows[rank] = [q5_mul(v, inv) for v in rows[rank]]
        for r in range(n_rows):
            if r != rank and rows[r][col] != Q5_ZERO:
                factor = rows[r][col]
                rows[r] = [q5_sub(rows[r][c], q5_mul(factor, rows[rank][c])) for c in range(n_cols)]
        rank += 1
        col += 1
    return n_cols - rank


# ==========================================================================
# Carrier specifications (Stage A objects)
# ==========================================================================


@dataclass(frozen=True)
class CarrierSpec:
    """A carrier as the generator runs it: ports, seams, rule, schedule."""

    name: str
    ports: int
    seams: tuple[tuple[int, int], ...]
    rule: str
    schedule: str = "uniform"
    carriers: int = 1

    def __post_init__(self) -> None:
        if self.rule not in (RULE_SEAM_MEAN, RULE_INTEGER, RULE_OVERWRITE):
            raise ValueError(f"unknown rule {self.rule!r}")
        if self.schedule != "uniform":
            raise ValueError("only the uniform schedule is declared")
        if any(not (0 <= i < j < self.ports) for i, j in self.seams):
            raise ValueError("seams must be pairs i < j inside the port range")
        if len(set(self.seams)) != len(self.seams):
            raise ValueError("duplicate seam")


def seams_from_oriented_faces(faces: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """Unoriented seams of an oriented triangulation; every edge must appear once per direction."""

    directed: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if (u, v) in directed:
                raise ValueError("directed edge repeated: orientation inconsistent")
            directed.add((u, v))
    for u, v in directed:
        if (v, u) not in directed:
            raise ValueError("directed edge without its reverse: surface not closed")
    return tuple(sorted({(min(u, v), max(u, v)) for u, v in directed}))


TETRAHEDRON_ORIENTED_FACES: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2))

# Octahedron ports: +x 0, -x 1, +y 2, -y 3, +z 4, -z 5; faces oriented outward.
OCTAHEDRON_ORIENTED_FACES: tuple[tuple[int, int, int], ...] = (
    (0, 2, 4),
    (2, 1, 4),
    (1, 3, 4),
    (3, 0, 4),
    (2, 0, 5),
    (1, 2, 5),
    (3, 1, 5),
    (0, 3, 5),
)


def icosahedron_spec(rule: str = RULE_SEAM_MEAN, name: str | None = None) -> CarrierSpec:
    return CarrierSpec(
        name=name or f"icosahedron_{rule}",
        ports=_carrier.PORT_COUNT,
        seams=tuple(sorted(_carrier.seams())),
        rule=rule,
    )


def tetrahedron_spec() -> CarrierSpec:
    return CarrierSpec(
        name="tetrahedron_seam_mean",
        ports=4,
        seams=seams_from_oriented_faces(TETRAHEDRON_ORIENTED_FACES),
        rule=RULE_SEAM_MEAN,
    )


def octahedron_spec() -> CarrierSpec:
    return CarrierSpec(
        name="octahedron_seam_mean",
        ports=6,
        seams=seams_from_oriented_faces(OCTAHEDRON_ORIENTED_FACES),
        rule=RULE_SEAM_MEAN,
    )


def federation_spec(carriers: int = FEDERATION_CARRIERS) -> CarrierSpec:
    """Isolated federation: ``carriers`` disjoint icosahedral carriers, seam mean law."""

    base = tuple(sorted(_carrier.seams()))
    ports = _carrier.PORT_COUNT
    seams = tuple(
        (i + c * ports, j + c * ports) for c in range(carriers) for i, j in base
    )
    return CarrierSpec(
        name=f"federation_isolated_{carriers}",
        ports=ports * carriers,
        seams=seams,
        rule=RULE_SEAM_MEAN,
        carriers=carriers,
    )


SOURCE_SPECS: dict[str, Any] = {
    "icosahedron_seam_mean": lambda: icosahedron_spec(RULE_SEAM_MEAN),
    "icosahedron_integer_nearest_agreement": lambda: icosahedron_spec(RULE_INTEGER),
    "icosahedron_overwrite": lambda: icosahedron_spec(RULE_OVERWRITE),
    "tetrahedron_seam_mean": tetrahedron_spec,
    "octahedron_seam_mean": octahedron_spec,
    "federation_isolated_20": federation_spec,
}


# ==========================================================================
# Stage A: generator
# ==========================================================================


def apply_rule(rule: str, xi: Fraction, xj: Fraction, direction: int) -> tuple[Fraction, Fraction]:
    """One completed event on a seam; ``direction`` is the uniformly drawn directed label."""

    if rule == RULE_SEAM_MEAN:
        m = (xi + xj) / 2
        return m, m
    if rule == RULE_INTEGER:
        s = int(xi) + int(xj)
        low, high = s // 2, -((-s) // 2)
        return (Fraction(high), Fraction(low)) if direction == 0 else (Fraction(low), Fraction(high))
    if rule == RULE_OVERWRITE:
        return (xi, xi) if direction == 0 else (xj, xj)
    raise ValueError(rule)


def _components(ports: int, seams: Sequence[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(ports))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in seams:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for p in range(ports):
        groups.setdefault(find(p), []).append(p)
    return [sorted(g) for g in sorted(groups.values(), key=lambda g: min(g))]


def _int_matmul(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> list[list[int]]:
    n = len(a)
    bt = [[b[k][j] for k in range(n)] for j in range(n)]
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def integer_matrix_power_naive(matrix: tuple[tuple[int, ...], ...], exponent: int) -> tuple[tuple[int, ...], ...]:
    """Repeated squaring on the matrix itself (cross-check for the Cayley-Hamilton route)."""

    n = len(matrix)
    result = [[int(i == j) for j in range(n)] for i in range(n)]
    base = [list(r) for r in matrix]
    e = exponent
    while e:
        if e & 1:
            result = _int_matmul(result, base)
        e >>= 1
        if e:
            base = _int_matmul(base, base)
    return tuple(tuple(r) for r in result)


def characteristic_polynomial(matrix: Sequence[Sequence[int]]) -> list[int]:
    """Coefficients ``c_0 .. c_n`` of ``det(xI - M)`` (monic) by Faddeev-LeVerrier, exact."""

    n = len(matrix)
    a = [[Fraction(v) for v in row] for row in matrix]
    c = [Fraction(0)] * (n + 1)
    c[n] = Fraction(1)
    mk = [[Fraction(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        prod = [[sum(a[i][t] * mk[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
        for i in range(n):
            prod[i][i] += c[n - k + 1]
        mk = prod
        trace = sum(a[i][t] * mk[t][i] for i in range(n) for t in range(n))
        c[n - k] = -trace / k
    if any(v.denominator != 1 for v in c):
        raise AssertionError("characteristic polynomial coefficients must be integers")
    return [int(v) for v in c]


def _poly_mulmod(p: Sequence[int], q: Sequence[int], c: Sequence[int], n: int) -> list[int]:
    prod = [0] * (2 * n - 1)
    for i, pi in enumerate(p):
        if pi:
            for j, qj in enumerate(q):
                if qj:
                    prod[i + j] += pi * qj
    for i in range(2 * n - 2, n - 1, -1):
        coef = prod[i]
        if coef:
            prod[i] = 0
            for j in range(n):
                prod[i - n + j] -= coef * c[j]
    return prod[:n]


@lru_cache(maxsize=64)
def integer_matrix_power(matrix: tuple[tuple[int, ...], ...], exponent: int) -> tuple[tuple[int, ...], ...]:
    """``M^exponent`` exactly: ``x^exponent`` modulo the characteristic polynomial, evaluated at ``M``."""

    n = len(matrix)
    if n == 1:
        return ((matrix[0][0] ** exponent,),)
    c = characteristic_polynomial(matrix)
    x = [0] * n
    x[1] = 1
    result = [1] + [0] * (n - 1)
    base = x
    e = exponent
    while e:
        if e & 1:
            result = _poly_mulmod(result, base, c, n)
        e >>= 1
        if e:
            base = _poly_mulmod(base, base, c, n)
    powers = [[[int(i == j) for j in range(n)] for i in range(n)]]
    for _ in range(1, n):
        powers.append(_int_matmul(powers[-1], matrix))
    out = [[0] * n for _ in range(n)]
    for k, coef in enumerate(result):
        if coef:
            pk = powers[k]
            for i in range(n):
                row = pk[i]
                out_row = out[i]
                for j in range(n):
                    if row[j]:
                        out_row[j] += coef * row[j]
    return tuple(tuple(r) for r in out)


def expectation_power_numerators(
    ports: int, seams: Sequence[tuple[int, int]], total_seams: int, steps: int
) -> tuple[list[list[int]], int]:
    """``M^steps`` with ``M = 2S I - L`` on one component; the value is ``M^steps / (2S)^steps``.

    ``T = I - L/(2S)`` is the synchronous expectation of one uniformly
    scheduled conservative side-symmetric seam event.
    """

    two_s = 2 * total_seams
    lap = [[0] * ports for _ in range(ports)]
    for i, j in seams:
        lap[i][i] += 1
        lap[j][j] += 1
        lap[i][j] -= 1
        lap[j][i] -= 1
    m = tuple(tuple((two_s if r == c else 0) - lap[r][c] for c in range(ports)) for r in range(ports))
    power = integer_matrix_power(m, steps)
    return [list(r) for r in power], two_s**steps


def probe_log(spec: CarrierSpec, steps: Sequence[int], digits: int) -> dict[str, Any]:
    """Readback after ``2n`` synchronous expectation steps from every one-hot state."""

    components = _components(spec.ports, spec.seams)
    readback: dict[str, list[dict[str, str]]] = {}
    for n in steps:
        vectors: list[dict[str, str]] = [dict() for _ in range(spec.ports)]
        for comp in components:
            local = {p: k for k, p in enumerate(comp)}
            comp_seams = [(local[i], local[j]) for i, j in spec.seams if i in local and j in local]
            power, den = expectation_power_numerators(len(comp), comp_seams, len(spec.seams), 2 * n)
            for p in comp:
                column = local[p]
                entries = {}
                for q in comp:
                    num = power[local[q]][column]
                    if num != 0:
                        entries[str(q)] = readback_string(num, den, digits)
                vectors[p] = entries
        readback[str(n)] = vectors
    return {"steps": list(steps), "readback_digits": digits, "readback": readback}


def run_source(
    spec: CarrierSpec,
    seed: int,
    events: int | None = None,
    probe_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Stage A.  Run the carrier and emit the event log only."""

    rng = random.Random(seed)
    state = [Fraction(rng.randrange(LOAD_RANGE)) for _ in range(spec.ports)]
    initial = [encode_value(v) for v in state]
    if events is None:
        per_seam = FEDERATION_EVENTS_PER_SEAM if spec.carriers > 1 else EVENTS_PER_SEAM
        events = per_seam * len(spec.seams)
    if probe_steps is None:
        probe_steps = [n * spec.carriers for n in PROBE_STEPS]
    ports_per_carrier = spec.ports // spec.carriers
    log_events: list[dict[str, Any]] = []
    labels = 2 * len(spec.seams)
    for step in range(events):
        label = rng.randrange(labels)
        i, j = spec.seams[label // 2]
        direction = label % 2
        before_i, before_j = state[i], state[j]
        after_i, after_j = apply_rule(spec.rule, before_i, before_j, direction)
        state[i], state[j] = after_i, after_j
        event: dict[str, Any] = {
            "step": step,
            "changed": {
                str(i): [encode_value(before_i), encode_value(after_i)],
                str(j): [encode_value(before_j), encode_value(after_j)],
            },
        }
        if spec.carriers > 1:
            event["carrier"] = i // ports_per_carrier
        log_events.append(event)
    log: dict[str, Any] = {
        "schema": LOG_SCHEMA,
        "ports": spec.ports,
        "carriers": spec.carriers,
        "ports_per_carrier": ports_per_carrier,
        "initial": initial,
        "events": log_events,
        "final": [encode_value(v) for v in state],
        "probe": probe_log(spec, probe_steps, READBACK_DIGITS),
    }
    return log


# ==========================================================================
# Stage B: recovery.  Every function below reads the log or values derived
# from it; none reads a CarrierSpec.
# ==========================================================================


def _validate_log(log: Mapping[str, Any]) -> None:
    required = {"schema", "ports", "carriers", "ports_per_carrier", "initial", "events", "final", "probe"}
    if set(log) != required:
        raise ValueError(f"log keys {sorted(log)} differ from {sorted(required)}")
    if log["schema"] != LOG_SCHEMA:
        raise ValueError("unexpected log schema")
    ports = int(log["ports"])
    if len(log["initial"]) != ports or len(log["final"]) != ports:
        raise ValueError("initial/final length differs from the port count")


def _events_table(log: Mapping[str, Any]) -> list[tuple[int, int, Fraction, Fraction, Fraction, Fraction]]:
    """Per event ``(i, j, before_i, after_i, before_j, after_j)`` with ``i < j``."""

    ports = int(log["ports"])
    table = []
    for event in log["events"]:
        changed = event["changed"]
        if len(changed) != 2:
            raise ValueError(f"event {event.get('step')} touches {len(changed)} ports")
        (a, va), (b, vb) = sorted(((int(k), v) for k, v in changed.items()), key=lambda kv: kv[0])
        if not (0 <= a < b < ports):
            raise ValueError("event port outside the port range")
        table.append((a, b, parse_value(va[0]), parse_value(va[1]), parse_value(vb[0]), parse_value(vb[1])))
    return table


def recover_seams(log: Mapping[str, Any]) -> dict[str, Any]:
    ports = int(log["ports"])
    table = _events_table(log)
    seams = sorted({(a, b) for a, b, *_ in table})
    degree = [0] * ports
    for a, b in seams:
        degree[a] += 1
        degree[b] += 1
    ports_per_carrier = int(log["ports_per_carrier"])
    carriers_respected = True
    for event, (a, b, *_) in zip(log["events"], table):
        if "carrier" in event:
            c = int(event["carrier"])
            if not (c * ports_per_carrier <= a < (c + 1) * ports_per_carrier) or not (
                c * ports_per_carrier <= b < (c + 1) * ports_per_carrier
            ):
                carriers_respected = False
    return {
        "ports": ports,
        "seams": [list(s) for s in seams],
        "seam_count": len(seams),
        "degree_sequence": sorted(degree),
        "event_count": len(table),
        "all_events_touch_exactly_two_ports": True,
        "ports_without_events": sum(1 for d in degree if d == 0),
        "carrier_index_consistent_with_blocks": carriers_respected,
    }


def _adjacency_sets(ports: int, seams: Sequence[tuple[int, int]]) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(ports)]
    for i, j in seams:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _distances(ports: int, adj: Sequence[set[int]]) -> list[list[int]]:
    dist = [[-1] * ports for _ in range(ports)]
    for s in range(ports):
        dist[s][s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if dist[s][v] < 0:
                        dist[s][v] = d
                        nxt.append(v)
            frontier = nxt
    return dist


def recover_pairing(ports: int, seams: Sequence[tuple[int, int]]) -> dict[str, Any]:
    adj = _adjacency_sets(ports, seams)
    dist = _distances(ports, adj)

    def involution(candidates: list[list[int]]) -> list[int] | None:
        if any(len(c) != 1 for c in candidates):
            return None
        pairing = [c[0] for c in candidates]
        if any(pairing[p] == p or pairing[pairing[p]] != p for p in range(ports)):
            return None
        return pairing

    three = involution([[q for q in range(ports) if dist[p][q] == 3] for p in range(ports)])
    eccentricity = [max(dist[p]) for p in range(ports)]
    farthest = None
    if len(set(eccentricity)) == 1 and eccentricity[0] > 0:
        pairing = involution([[q for q in range(ports) if dist[p][q] == eccentricity[0]] for p in range(ports)])
        if pairing is not None:
            farthest = {"distance": eccentricity[0], "pairing": pairing}
    return {
        "type": "distance_three_involution" if three is not None else "none",
        "distance_three_pairing": three,
        "farthest_port_pairing": farthest,
        "diameter": max(eccentricity),
    }


def graph_automorphisms(ports: int, seams: Sequence[tuple[int, int]]) -> list[tuple[int, ...]]:
    """All port permutations preserving the seam set (backtracking)."""

    adj = _adjacency_sets(ports, seams)
    degree = [len(a) for a in adj]
    found: list[tuple[int, ...]] = []

    def extend(assignment: list[int], used: set[int]) -> None:
        p = len(assignment)
        if p == ports:
            found.append(tuple(assignment))
            return
        for image in range(ports):
            if image in used or degree[image] != degree[p]:
                continue
            if all((q in adj[p]) == (assignment[q] in adj[image]) for q in range(p)):
                assignment.append(image)
                used.add(image)
                extend(assignment, used)
                assignment.pop()
                used.discard(image)

    extend([], set())
    return found


def _cyclic(face: Sequence[int]) -> tuple[int, ...]:
    face = tuple(face)
    k = face.index(min(face))
    return face[k:] + face[:k]


def recover_orientation(ports: int, seams: Sequence[tuple[int, int]], automorphisms: Sequence[tuple[int, ...]]) -> dict[str, Any]:
    """Triangles as 3-cycles, one consistent orientation up to global reversal, proper subgroup."""

    adj = _adjacency_sets(ports, seams)
    triangles = sorted({tuple(sorted((a, b, c))) for a in range(ports) for b in adj[a] if b > a for c in adj[a] & adj[b] if c > b})
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for t, (a, b, c) in enumerate(triangles):
        for e in ((a, b), (a, c), (b, c)):
            edge_faces.setdefault(e, []).append(t)
    surface = len(triangles) > 0 and all(len(v) == 2 for v in edge_faces.values()) and set(edge_faces) == set(seams)
    result: dict[str, Any] = {
        "triangle_count": len(triangles),
        "every_seam_in_exactly_two_triangles": surface,
        "orientable": None,
        "orientation_classes": None,
        "rotation_order": None,
        "rotation_order_by_class": None,
        "rotation_group_identical_for_both_classes": None,
        "orientation_declared_by_a1": True,
    }
    if not surface:
        return result
    oriented: dict[int, tuple[int, int, int]] = {0: triangles[0]}
    stack = [0]
    orientable = True
    while stack and orientable:
        t = stack.pop()
        a, b, c = oriented[t]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            other = next(s for s in edge_faces[key] if s != t)
            w = next(x for x in triangles[other] if x not in (u, v))
            want = (v, u, w)  # the neighbour traverses the shared edge in reverse
            if other in oriented:
                if _cyclic(oriented[other]) != _cyclic(want):
                    orientable = False
                    break
            else:
                oriented[other] = want
                stack.append(other)
    result["orientable"] = orientable and len(oriented) == len(triangles)
    if not result["orientable"]:
        return result
    class_a = {_cyclic(f) for f in oriented.values()}
    class_b = {_cyclic((f[0], f[2], f[1])) for f in oriented.values()}
    proper: list[list[tuple[int, ...]]] = []
    for cls in (class_a, class_b):
        proper.append([g for g in automorphisms if all(_cyclic((g[a], g[b], g[c])) in cls for a, b, c in cls)])
    result["orientation_classes"] = 2
    result["rotation_order_by_class"] = [len(proper[0]), len(proper[1])]
    result["rotation_group_identical_for_both_classes"] = set(proper[0]) == set(proper[1])
    result["rotation_order"] = len(proper[0])
    result["_rotations"] = proper[0]
    return result


def recover_rule(table: Sequence[tuple[int, int, Fraction, Fraction, Fraction, Fraction]]) -> dict[str, Any]:
    conservative = 0
    symmetric = 0
    discriminating = 0
    mean_form = 0
    shell_form = 0
    overwrite_form = 0
    odd_total = 0
    ceiling_to_lower = 0
    overwrite_lower_side = 0
    integer_values = True
    for a, b, bi, ai, bj, aj in table:
        if ai + aj == bi + bj:
            conservative += 1
        if ai == aj:
            symmetric += 1
        if any(v.denominator != 1 for v in (bi, ai, bj, aj)):
            integer_values = False
        if bi == bj:
            continue
        discriminating += 1
        s = bi + bj
        if ai == aj == s / 2:
            mean_form += 1
        if all(v.denominator == 1 for v in (bi, bj, ai, aj)):
            si = int(s)
            low, high = si // 2, -((-si) // 2)
            if {int(ai), int(aj)} == {low, high} and int(ai) + int(aj) == si:
                shell_form += 1
                if si % 2 == 1:
                    odd_total += 1
                    if int(ai) == high:
                        ceiling_to_lower += 1
        if (ai == bi and aj == bi) or (aj == bj and ai == bj):
            overwrite_form += 1
            if ai == bj:
                overwrite_lower_side += 1
    n = len(table)
    nonconservative = n - conservative
    if discriminating == 0:
        rule = RULE_UNCLASSIFIED
    elif nonconservative == 0 and mean_form == discriminating:
        rule = RULE_SEAM_MEAN
    elif nonconservative == 0 and integer_values and shell_form == discriminating:
        rule = RULE_INTEGER
    elif overwrite_form == discriminating:
        rule = RULE_OVERWRITE
    else:
        rule = RULE_UNCLASSIFIED
    tie_frequency = None
    tie_law = None
    if rule == RULE_INTEGER and odd_total > 0:
        tie_frequency = Fraction(ceiling_to_lower, odd_total)
        z = abs(float(tie_frequency) - 0.5) * 2.0 * odd_total**0.5
        tie_law = "uniform" if z <= TIE_Z_THRESHOLD else "biased"
    side_frequency = None
    side_law = None
    if rule == RULE_OVERWRITE and discriminating > 0:
        side_frequency = Fraction(overwrite_lower_side, discriminating)
        z = abs(float(side_frequency) - 0.5) * 2.0 * discriminating**0.5
        side_law = "uniform" if z <= TIE_Z_THRESHOLD else "biased"
    return {
        "class": rule,
        "event_count": n,
        "conservative_events": conservative,
        "nonconservative_events": nonconservative,
        "symmetric_events": symmetric,
        "discriminating_events": discriminating,
        "mean_form_events": mean_form,
        "nearest_agreement_shell_events": shell_form,
        "overwrite_form_events": overwrite_form,
        "odd_total_events": odd_total,
        "all_values_integer": integer_values,
        "tie_ceiling_to_lower_index_frequency": None if tie_frequency is None else str(tie_frequency),
        "tie_ceiling_to_lower_index_frequency_float": None if tie_frequency is None else float(tie_frequency),
        "tie_law": tie_law,
        "overwrite_lower_index_side_frequency": None if side_frequency is None else str(side_frequency),
        "overwrite_side_law": side_law,
    }


def recover_schedule(table: Sequence[tuple[int, int, Fraction, Fraction, Fraction, Fraction]], seams: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Exact chi-square statistic of the seam event counts against the uniform law."""

    counts = {tuple(s): 0 for s in seams}
    for a, b, *_ in table:
        counts[(a, b)] += 1
    n = len(table)
    s = len(seams)
    statistic = Fraction(s, n) * sum(Fraction(c * c) for c in counts.values()) - n if n else Fraction(0)
    df = s - 1
    threshold = float(_chi2.ppf(1.0 - CHI_SQUARE_SIGNIFICANCE, df)) if df > 0 else 0.0
    not_rejected = float(statistic) <= threshold
    return {
        "law": "uniform" if not_rejected else "nonuniform",
        "chi_square": str(statistic),
        "chi_square_float": float(statistic),
        "degrees_of_freedom": df,
        "significance": CHI_SQUARE_SIGNIFICANCE,
        "threshold": threshold,
        "uniform_not_rejected": not_rejected,
        "min_seam_count": min(counts.values()),
        "max_seam_count": max(counts.values()),
    }


@dataclass(frozen=True)
class ProbeData:
    steps: tuple[int, ...]
    digits: int
    readback: dict[int, list[dict[int, Fraction]]]


def _parse_probe(log: Mapping[str, Any]) -> ProbeData:
    probe = log["probe"]
    steps = tuple(int(n) for n in probe["steps"])
    parsed: dict[int, list[dict[int, Fraction]]] = {}
    for n in steps:
        vectors = probe["readback"][str(n)]
        parsed[n] = [{int(q): Fraction(v) for q, v in vec.items()} for vec in vectors]
    return ProbeData(steps=steps, digits=int(probe["readback_digits"]), readback=parsed)


def recover_gram(
    comp: Sequence[int],
    comp_seams: Sequence[tuple[int, int]],
    total_seams: int,
    probe: ProbeData,
) -> dict[str, Any]:
    """Centre and normalize the probe readback into ``K_n = m C_n / tr C_n`` on one component.

    The readback carries a declared number of significant digits.  A step is
    resolved when the centered kernel keeps at least ``GRAM_RESOLUTION_DIGITS``
    of them; the Gram, its eigenvalues and its rank are read at the largest
    resolved step.  The probe is also predicted from the recovered seam set
    with ``T_rec = I - L_rec / (2 S_rec)`` and compared at the readback
    tolerance using integer arithmetic only.
    """

    m = len(comp)
    local = {p: k for k, p in enumerate(comp)}
    steps = probe.steps
    support_ok = True
    symmetric = True
    ranks: dict[str, int | None] = {}
    eigen_by_step: dict[str, list[float] | None] = {}
    kernel_by_step: dict[str, list[list[float]] | None] = {}
    resolved_digits: dict[str, float | None] = {}
    max_dev = 0.0
    prediction_ok = True
    tolerance_exponent = probe.digits - PROBE_PREDICTION_TOLERANCE_MARGIN
    tol_scale = 10**tolerance_exponent
    for n in steps:
        r = [[Fraction(0)] * m for _ in range(m)]
        for p in comp:
            for q, v in probe.readback[n][p].items():
                if q not in local:
                    support_ok = False
                    continue
                r[local[q]][local[p]] = v
        if any(r[a][b] != r[b][a] for a in range(m) for b in range(a + 1, m)):
            symmetric = False
        row_mean = [sum(row) / m for row in r]
        col_mean = [sum(r[a][b] for a in range(m)) / m for b in range(m)]
        total = sum(row_mean) / m
        c = [[r[a][b] - row_mean[a] - col_mean[b] + total for b in range(m)] for a in range(m)]
        raw_scale = max(abs(v) for row in r for v in row)
        centered_scale = max(abs(v) for row in c for v in row)
        trace = sum(c[a][a] for a in range(m))
        if centered_scale == 0 or raw_scale == 0 or trace == 0:
            resolved_digits[str(n)] = None
            ranks[str(n)] = None
            eigen_by_step[str(n)] = None
            kernel_by_step[str(n)] = None
        else:
            digits = probe.digits + float(np.log10(float(centered_scale / raw_scale)))
            resolved_digits[str(n)] = digits
            if digits < GRAM_RESOLUTION_DIGITS:
                ranks[str(n)] = None
                eigen_by_step[str(n)] = None
                kernel_by_step[str(n)] = None
            else:
                k = np.array([[float(c[a][b] * m / trace) for b in range(m)] for a in range(m)])
                eig = sorted(np.linalg.eigvalsh(0.5 * (k + k.T)).tolist(), reverse=True)
                top = max(abs(e) for e in eig)
                ranks[str(n)] = int(sum(1 for e in eig if e > GRAM_RANK_TOLERANCE * top))
                eigen_by_step[str(n)] = eig
                kernel_by_step[str(n)] = k.tolist()
        # Prediction of the probe from the recovered seam set: T_rec = I - L_rec / (2 S_rec).
        power, den = expectation_power_numerators(m, [(local[i], local[j]) for i, j in comp_seams], total_seams, 2 * n)
        for p in comp:
            column = local[p]
            observed = probe.readback[n][p]
            for q in comp:
                pred_num = power[local[q]][column]
                obs = observed.get(q, Fraction(0))
                if pred_num == 0:
                    ok = obs == 0
                    dev = float(abs(obs))
                else:
                    lhs = abs(obs.numerator * den - pred_num * obs.denominator)
                    rhs = abs(pred_num * obs.denominator)
                    ok = lhs * tol_scale <= rhs
                    dev = lhs / rhs
                max_dev = max(max_dev, dev)
                prediction_ok = prediction_ok and ok
    resolved_steps = [n for n in steps if ranks[str(n)] is not None]
    chosen = max(resolved_steps) if resolved_steps else None
    return {
        "probe_steps": list(steps),
        "readback_digits": probe.digits,
        "resolution_digits_required": GRAM_RESOLUTION_DIGITS,
        "resolved_digits_by_step": resolved_digits,
        "gram_step": chosen,
        "gram_step_index": None if chosen is None else steps.index(chosen),
        "readback_support_within_component": support_ok,
        "readback_symmetric": symmetric,
        "eigenvalues": None if chosen is None else eigen_by_step[str(chosen)],
        "kernel": None if chosen is None else kernel_by_step[str(chosen)],
        "rank": None if chosen is None else ranks[str(chosen)],
        "rank_by_step": ranks,
        "rank_tolerance": GRAM_RANK_TOLERANCE,
        "prediction_relative_tolerance_exponent": -tolerance_exponent,
        "prediction_max_relative_deviation": max_dev,
        "prediction_within_readback_tolerance": prediction_ok,
    }


def _identify_q5(value: float, ports: int, lap: Sequence[Sequence[Q5]], multiplicity: int) -> Q5 | None:
    bound = 4 * ports
    candidates: list[Q5] = []
    for kb in range(0, bound + 1):
        for sign_b in ((1,) if kb == 0 else (1, -1)):
            b = Fraction(sign_b * kb, 2)
            for ka in range(-bound, bound + 1):
                a = Fraction(ka, 2)
                if abs(float(a) + float(b) * 5.0**0.5 - value) < EIGENVALUE_CLUSTER_TOLERANCE:
                    candidates.append((a, b))
    for mu in candidates:
        shifted = [[q5_sub(lap[r][c], mu) if r == c else lap[r][c] for c in range(ports)] for r in range(ports)]
        if q5_nullity(shifted) == multiplicity:
            return mu
    return None


def _compose(g: Sequence[int], h: Sequence[int]) -> tuple[int, ...]:
    return tuple(g[h[p]] for p in range(len(g)))


def recover_lie_type(ports: int, seams: Sequence[tuple[int, int]], rotations: Sequence[tuple[int, ...]] | None) -> dict[str, Any]:
    """Character test of the Lie type dimension count on the recovered rotation group."""

    result: dict[str, Any] = {
        "simple_dimension_table": list(LIE_SIMPLE_DIMENSIONS),
        "centre_dimension_at_most": 1,
        "candidate_splits": [],
        "surviving_splits": [],
        "split": "unidentified",
        "assignment_count": 0,
        "trivial_isotypic_dimension": None,
        "bands": [],
        "commutant_dimension": None,
        "permutation_character_by_order": None,
    }
    if not rotations:
        result["split"] = "no_rotation_group"
        return result
    lap_int = [[0] * ports for _ in range(ports)]
    for i, j in seams:
        lap_int[i][i] += 1
        lap_int[j][j] += 1
        lap_int[i][j] -= 1
        lap_int[j][i] -= 1
    lap = [[q5(lap_int[r][c]) for c in range(ports)] for r in range(ports)]
    eig = np.linalg.eigvalsh(np.array(lap_int, dtype=float))
    clusters: list[list[float]] = []
    for e in sorted(eig.tolist()):
        if clusters and abs(e - clusters[-1][0]) < EIGENVALUE_CLUSTER_TOLERANCE:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    exact: list[tuple[Q5, int]] = []
    for cl in clusters:
        mu = _identify_q5(sum(cl) / len(cl), ports, lap, len(cl))
        if mu is None:
            result["split"] = "band_not_in_q_sqrt5"
            return result
        exact.append((mu, len(cl)))
    projectors: list[list[list[Q5]]] = []
    identity = [[q5(int(r == c)) for c in range(ports)] for r in range(ports)]
    for k, (mu_k, _) in enumerate(exact):
        acc = [row[:] for row in identity]
        for m_, (mu_m, _) in enumerate(exact):
            if m_ == k:
                continue
            factor = [[q5_sub(lap[r][c], mu_m) if r == c else lap[r][c] for c in range(ports)] for r in range(ports)]
            acc = q5_matmul(acc, factor)
            scale = q5_inv(q5_sub(mu_k, mu_m))
            acc = [[q5_mul(v, scale) for v in row] for row in acc]
        projectors.append(acc)
    group = [tuple(g) for g in rotations]
    order = len(group)
    orders: dict[tuple[int, ...], int] = {}
    identity_perm = tuple(range(ports))
    for g in group:
        k = 1
        h = g
        while h != identity_perm:
            h = _compose(g, h)
            k += 1
        orders[g] = k
    perm_char = {g: sum(int(g[p] == p) for p in range(ports)) for g in group}
    trivial_dim = Fraction(sum(perm_char.values()), order)
    commutant = Fraction(sum(v * v for v in perm_char.values()), order)
    band_chars: list[dict[tuple[int, ...], Q5]] = []
    bands_out = []
    for (mu, mult), proj in zip(exact, projectors):
        chars = {g: _q5_sum(proj[q][g[q]] for q in range(ports)) for g in group}
        norm = _q5_sum(q5_mul(v, v) for v in chars.values())
        irreducible = norm == (Fraction(order), Fraction(0))
        trivial = all(v == Q5_ONE for v in chars.values())
        by_order: dict[str, list[list[str]]] = {}
        for g, v in chars.items():
            by_order.setdefault(str(orders[g]), [])
            if q5_str(v) not in by_order[str(orders[g])]:
                by_order[str(orders[g])].append(q5_str(v))
        band_chars.append(chars)
        bands_out.append(
            {
                "laplacian_eigenvalue": q5_str(mu),
                "laplacian_eigenvalue_float": q5_float(mu),
                "dimension": mult,
                "irreducible": irreducible,
                "trivial": trivial,
                "character_values_by_element_order": {k: sorted(v) for k, v in by_order.items()},
            }
        )
    result["trivial_isotypic_dimension"] = int(trivial_dim) if trivial_dim.denominator == 1 else str(trivial_dim)
    result["commutant_dimension"] = int(commutant) if commutant.denominator == 1 else str(commutant)
    result["bands"] = bands_out
    result["permutation_character_by_order"] = {
        str(o): sorted({perm_char[g] for g in group if orders[g] == o}) for o in sorted(set(orders.values()))
    }
    if not all(b["irreducible"] for b in bands_out):
        result["split"] = "bands_not_irreducible"
        return result

    def union_char(bands: frozenset[int], g: tuple[int, ...]) -> Q5:
        return _q5_sum(band_chars[b][g] for b in bands)

    def union_dim(bands: frozenset[int]) -> int:
        return sum(exact[b][1] for b in bands)

    power_cache: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {}

    def perm_power(g: tuple[int, ...], k: int) -> tuple[int, ...]:
        key = (g, k)
        if key not in power_cache:
            h = identity_perm
            for _ in range(k):
                h = _compose(g, h)
            power_cache[key] = h
        return power_cache[key]

    def determinant(bands: frozenset[int], g: tuple[int, ...]) -> Q5:
        d = union_dim(bands)
        p = [None] + [union_char(bands, perm_power(g, k)) for k in range(1, d + 1)]
        e: list[Q5] = [Q5_ONE]
        for k in range(1, d + 1):
            acc = Q5_ZERO
            for i in range(1, k + 1):
                term = q5_mul(e[k - i], p[i])
                acc = q5_add(acc, term) if i % 2 == 1 else q5_sub(acc, term)
            e.append((acc[0] / k, acc[1] / k))
        return e[d]

    so_type_cache: dict[frozenset[int], bool] = {}

    def special_orthogonal(bands: frozenset[int]) -> bool:
        if bands not in so_type_cache:
            so_type_cache[bands] = all(determinant(bands, g) == Q5_ONE for g in group)
        return so_type_cache[bands]

    n_bands = len(exact)
    all_unions = [frozenset(c) for k in range(1, n_bands + 1) for c in combinations(range(n_bands), k)]
    three_reps = [u for u in all_unions if union_dim(u) == 3 and special_orthogonal(u)]
    five_reps = [u for u in all_unions if union_dim(u) == 5 and special_orthogonal(u)]

    def part_passes(bands: frozenset[int], dim: int) -> list[str]:
        witnesses: list[str] = []
        if dim == 3:
            if special_orthogonal(bands):
                witnesses.append("so3")
        elif dim == 8:
            if all(union_char(bands, g) == q5(8) for g in group):
                witnesses.append("trivial")
            for rho in three_reps:
                if all(union_char(bands, g) == q5_sub(q5_mul(union_char(rho, g), union_char(rho, g)), Q5_ONE) for g in group):
                    witnesses.append("adjoint_su3_via_so3_rep_" + "+".join(str(b) for b in sorted(rho)))
        elif dim == 10:
            if all(union_char(bands, g) == q5(10) for g in group):
                witnesses.append("trivial")
            for sigma in five_reps:
                if all(
                    union_char(bands, g)
                    == _q5_half(q5_sub(q5_mul(union_char(sigma, g), union_char(sigma, g)), union_char(sigma, perm_power(g, 2))))
                    for g in group
                ):
                    witnesses.append("adjoint_so5_via_so5_rep_" + "+".join(str(b) for b in sorted(sigma)))
        return witnesses

    candidate_splits: list[tuple[int, tuple[int, ...]]] = []
    for centre in (0, 1):
        remaining = ports - centre
        for k in range(1, remaining // min(LIE_SIMPLE_DIMENSIONS) + 1):
            for parts in combinations_with_replacement(LIE_SIMPLE_DIMENSIONS, k):
                if sum(parts) == remaining:
                    candidate_splits.append((centre, parts))
    result["candidate_splits"] = [
        {"centre": c, "simple": list(p), "label": "+".join([str(c)] * (c > 0) + [str(x) for x in p])} for c, p in candidate_splits
    ]
    trivial_bands = [b for b in range(n_bands) if bands_out[b]["trivial"] and exact[b][1] == 1]
    survivors = []
    for centre, parts in candidate_splits:
        if centre > int(trivial_dim):
            continue
        slots = list(parts)
        seen: set[tuple[Any, ...]] = set()
        assignments = []

        def assign(idx: int, free: frozenset[int], chosen: list[frozenset[int]]) -> None:
            if idx == len(slots):
                if free:
                    return
                key = tuple(sorted((slots[i], tuple(sorted(chosen[i]))) for i in range(len(slots))))
                if key in seen:
                    return
                seen.add(key)
                witnesses = [part_passes(chosen[i], slots[i]) for i in range(len(slots))]
                if all(witnesses):
                    assignments.append(
                        [
                            {"dimension": slots[i], "bands": sorted(chosen[i]), "witness": witnesses[i]}
                            for i in range(len(slots))
                        ]
                    )
                return
            for k in range(1, len(free) + 1):
                for c in combinations(sorted(free), k):
                    if union_dim(frozenset(c)) == slots[idx]:
                        assign(idx + 1, free - frozenset(c), chosen + [frozenset(c)])

        centre_choices = [frozenset()] if centre == 0 else [frozenset([b]) for b in trivial_bands]
        for cb in centre_choices:
            assign(0, frozenset(range(n_bands)) - cb, [])
        if assignments:
            survivors.append(
                {
                    "centre": centre,
                    "simple": list(parts),
                    "label": "+".join([str(centre)] * (centre > 0) + [str(x) for x in parts]),
                    "assignments": assignments,
                }
            )
    result["surviving_splits"] = survivors
    if len(survivors) == 1:
        result["split"] = survivors[0]["label"]
        result["assignment_count"] = len(survivors[0]["assignments"])
    elif not survivors:
        result["split"] = "none"
    else:
        result["split"] = "ambiguous:" + ",".join(s["label"] for s in survivors)
        result["assignment_count"] = sum(len(s["assignments"]) for s in survivors)
    return result


def _q5_sum(values: Any) -> Q5:
    acc = Q5_ZERO
    for v in values:
        acc = q5_add(acc, v)
    return acc


def _q5_half(x: Q5) -> Q5:
    return (x[0] / 2, x[1] / 2)


def recover_descent(
    comp: Sequence[int],
    comp_seams: Sequence[tuple[int, int]],
    comp_table: Sequence[tuple[int, int, Fraction, Fraction, Fraction, Fraction]],
    initial: Sequence[Fraction],
    log_final: Sequence[Fraction],
) -> dict[str, Any]:
    """Event-wise descent of the centered squared norm on one component.

    A strict-descent violation is an event that changes the multiset of the
    two endpoint readings without strictly decreasing the potential.  Events
    that leave both readings in place are stutters, events that exchange them
    are swaps; neither changes the multiset.  Increases of the seam Laplacian
    form are reported separately.
    """

    local = {p: k for k, p in enumerate(comp)}
    m = len(comp)
    state = [initial[p] for p in comp]
    adj = _adjacency_sets(m, [(local[i], local[j]) for i, j in comp_seams])

    def local_laplacian_form(x: Sequence[Fraction], i: int, j: int) -> Fraction:
        acc = Fraction(0)
        for k in adj[i]:
            acc += (x[i] - x[k]) ** 2
        for k in adj[j]:
            if k != i:
                acc += (x[j] - x[k]) ** 2
        return acc

    violations = 0
    stutters = 0
    swaps = 0
    laplacian_increase = 0
    consistent = True
    total = sum(state)
    sum_sq = sum(v * v for v in state)
    for a, b, bi, ai, bj, aj in comp_table:
        i, j = local[a], local[b]
        if state[i] != bi or state[j] != bj:
            consistent = False
        lap_before = local_laplacian_form(state, i, j)
        v_before = sum_sq - total * total / m
        multiset_changed = sorted((bi, bj)) != sorted((ai, aj))
        state[i], state[j] = ai, aj
        total += (ai - bi) + (aj - bj)
        sum_sq += (ai * ai - bi * bi) + (aj * aj - bj * bj)
        v_after = sum_sq - total * total / m
        lap_after = local_laplacian_form(state, i, j)
        if (bi, bj) == (ai, aj):
            stutters += 1
        elif not multiset_changed:
            swaps += 1
        elif not v_after < v_before:
            violations += 1
        if lap_after > lap_before:
            laplacian_increase += 1
    return {
        "potential": "centered_squared_norm",
        "potential_reference": POTENTIAL_REFERENCE,
        "strict_descent_violations": violations,
        "stutter_events": stutters,
        "swap_events": swaps,
        "laplacian_form_increase_events": laplacian_increase,
        "replay_consistent_with_log": consistent and state == [log_final[p] for p in comp],
    }


def replay_rule(rule: str, xi: Fraction, xj: Fraction, direction: int) -> tuple[Fraction, Fraction]:
    """The recovered rule class as the recovery replays it (uniform tie and side placement)."""

    if rule == RULE_SEAM_MEAN:
        m = (xi + xj) / 2
        return m, m
    if rule == RULE_INTEGER:
        s = int(xi) + int(xj)
        low, high = s // 2, -((-s) // 2)
        return (Fraction(high), Fraction(low)) if direction == 0 else (Fraction(low), Fraction(high))
    if rule == RULE_OVERWRITE:
        return (xi, xi) if direction == 0 else (xj, xj)
    raise ValueError(rule)


def recover_terminal(
    comp: Sequence[int], comp_seams: Sequence[tuple[int, int]], rule: str, initial: Sequence[Fraction], log_final: Sequence[Fraction]
) -> dict[str, Any]:
    """Terminal law of the recovered rule on the recovered seams from the log's initial loads."""

    loads = [initial[p] for p in comp]
    finals = [log_final[p] for p in comp]
    m = len(comp)
    mean = sum(loads) / m
    spread = max(loads) - min(loads)
    tol = TERMINAL_TOLERANCE * spread if spread > 0 else Fraction(0)
    local = {p: k for k, p in enumerate(comp)}
    seams = [(local[i], local[j]) for i, j in comp_seams]
    terminals: list[list[Fraction]] = []
    if rule in (RULE_SEAM_MEAN, RULE_INTEGER, RULE_OVERWRITE) and seams:
        events = REPLAY_EVENTS_PER_SEAM * len(seams)
        for k in range(REPLAY_SCHEDULES):
            rng = random.Random(REPLAY_SEED_BASE + k)
            state = list(loads)
            labels = 2 * len(seams)
            for _ in range(events):
                label = rng.randrange(labels)
                i, j = seams[label // 2]
                state[i], state[j] = replay_rule(rule, state[i], state[j], label % 2)
            terminals.append(state)
    equals_mean = bool(terminals) and all(max(abs(v - mean) for v in t) <= tol for t in terminals)
    independent = bool(terminals) and all(
        max(abs(x - y) for x, y in zip(terminals[0], t)) <= tol for t in terminals[1:]
    )
    quotient = bool(terminals) and all(
        max(abs(x - y) for x, y in zip(sorted(terminals[0]), sorted(t))) <= tol for t in terminals[1:]
    )
    if equals_mean and independent:
        kind = "state"
    elif quotient:
        kind = "quotient_multiset"
    else:
        kind = "none"
    log_dev = max(abs(v - mean) for v in finals)
    return {
        "replay_schedules": REPLAY_SCHEDULES,
        "replay_events": REPLAY_EVENTS_PER_SEAM * len(seams),
        "terminal_equals_initial_mean": equals_mean,
        "terminal_schedule_independent": independent,
        "quotient_multiset_schedule_independent": quotient,
        "terminal_invariant_type": kind,
        "log_final_max_deviation_from_mean_float": float(log_dev),
        "log_final_deviation_ratio_float": float(log_dev / spread) if spread > 0 else 0.0,
        "log_final_total_equals_initial_total": sum(finals) == sum(loads),
    }


def _component_invariants(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ports": spec["ports"],
        "seams": spec["seam_count"],
        "degree_sequence": spec["degree_sequence"],
        "pairing_type": spec["pairing"]["type"],
        "farthest_pairing_distance": None
        if spec["pairing"]["farthest_port_pairing"] is None
        else spec["pairing"]["farthest_port_pairing"]["distance"],
        "automorphism_order": spec["automorphism_order"],
        "rotation_order": spec["orientation"]["rotation_order"],
        "rule_class": spec["rule"]["class"],
        "nonconservative_events_present": spec["rule"]["nonconservative_events"] > 0,
        "tie_law": spec["rule"]["tie_law"],
        "schedule_law": spec["schedule"]["law"],
        "gram_eigenvalues": None
        if spec["gram"]["eigenvalues"] is None
        else [round(v, 9) + 0.0 for v in spec["gram"]["eigenvalues"]],
        "gram_rank": spec["gram"]["rank"],
        "gram_step_index": spec["gram"]["gram_step_index"],
        "probe_prediction_within_tolerance": spec["gram"]["prediction_within_readback_tolerance"],
        "lie_split": spec["lie_type"]["split"],
        "lie_assignment_count": spec["lie_type"]["assignment_count"],
        "trivial_isotypic_dimension": spec["lie_type"]["trivial_isotypic_dimension"],
        "strict_descent_violation_count": spec["descent"]["strict_descent_violations"],
        "terminal_equals_initial_mean": spec["terminal"]["terminal_equals_initial_mean"],
        "terminal_schedule_independent": spec["terminal"]["terminal_schedule_independent"],
        "quotient_multiset_schedule_independent": spec["terminal"]["quotient_multiset_schedule_independent"],
        "terminal_invariant_type": spec["terminal"]["terminal_invariant_type"],
    }


def recover_component(
    comp: Sequence[int],
    table: Sequence[tuple[int, int, Fraction, Fraction, Fraction, Fraction]],
    all_seams: Sequence[tuple[int, int]],
    probe: ProbeData,
    initial: Sequence[Fraction],
    final: Sequence[Fraction],
) -> dict[str, Any]:
    comp_set = set(comp)
    comp_seams = [s for s in all_seams if s[0] in comp_set]
    comp_table = [row for row in table if row[0] in comp_set]
    local = {p: k for k, p in enumerate(comp)}
    m = len(comp)
    local_seams = [(local[i], local[j]) for i, j in comp_seams]
    degree = [0] * m
    for i, j in local_seams:
        degree[i] += 1
        degree[j] += 1
    autos = graph_automorphisms(m, local_seams)
    orientation = recover_orientation(m, local_seams, autos)
    rotations = orientation.pop("_rotations", None)
    gram = recover_gram(comp, comp_seams, len(all_seams), probe)
    rule = recover_rule(comp_table)
    spec = {
        "ports": m,
        "port_ids": list(comp),
        "seams": [list(s) for s in comp_seams],
        "seam_count": len(comp_seams),
        "degree_sequence": sorted(degree),
        "event_count": len(comp_table),
        "pairing": recover_pairing(m, local_seams),
        "automorphism_order": len(autos),
        "orientation": orientation,
        "rule": rule,
        "schedule": recover_schedule(comp_table, comp_seams),
        "gram": gram,
        "lie_type": recover_lie_type(m, local_seams, rotations),
        "descent": recover_descent(comp, comp_seams, comp_table, initial, final),
        "terminal": recover_terminal(comp, comp_seams, rule["class"], initial, final),
    }
    spec["invariants"] = _component_invariants(spec)
    return spec


def _isomorphic(m: int, seams_a: Sequence[tuple[int, int]], seams_b: Sequence[tuple[int, int]]) -> bool:
    adj_a = _adjacency_sets(m, seams_a)
    adj_b = _adjacency_sets(m, seams_b)
    if sorted(len(x) for x in adj_a) != sorted(len(x) for x in adj_b):
        return False

    def extend(assignment: list[int], used: set[int]) -> bool:
        p = len(assignment)
        if p == m:
            return True
        for image in range(m):
            if image in used or len(adj_a[p]) != len(adj_b[image]):
                continue
            if all((q in adj_a[p]) == (assignment[q] in adj_b[image]) for q in range(p)):
                assignment.append(image)
                used.add(image)
                if extend(assignment, used):
                    return True
                assignment.pop()
                used.discard(image)
        return False

    return extend([], set())


def recover_specification(log: Mapping[str, Any]) -> dict[str, Any]:
    """Stage B.  The fixed recovery algorithm; its only input is the log."""

    _validate_log(log)
    structure = recover_seams(log)
    table = _events_table(log)
    seams = [tuple(s) for s in structure["seams"]]
    ports = structure["ports"]
    components = _components(ports, seams)
    probe = _parse_probe(log)
    initial = [parse_value(v) for v in log["initial"]]
    final = [parse_value(v) for v in log["final"]]
    comp_specs = [recover_component(comp, table, seams, probe, initial, final) for comp in components]
    reference = comp_specs[0]
    identical = all(c["invariants"] == reference["invariants"] for c in comp_specs[1:])
    isomorphic = identical and all(
        _isomorphic(
            reference["ports"],
            [(reference["port_ids"].index(i), reference["port_ids"].index(j)) for i, j in reference["seams"]],
            [(c["port_ids"].index(i), c["port_ids"].index(j)) for i, j in c["seams"]],
        )
        for c in comp_specs[1:]
    )
    global_schedule = recover_schedule(table, seams)
    invariants = dict(reference["invariants"])
    invariants["components"] = len(components)
    invariants["components_isomorphic"] = isomorphic
    invariants["global_schedule_law"] = global_schedule["law"]
    recovered = {
        "ports": ports,
        "seam_count": len(seams),
        "structure": structure,
        "components": len(components),
        "components_identical_invariants": identical,
        "components_isomorphic": isomorphic,
        "component": reference,
        "global_schedule": global_schedule,
        "invariants": invariants,
    }
    if len(components) > 1:
        recovered["component_rule_classes"] = sorted({c["rule"]["class"] for c in comp_specs})
        recovered["description"] = f"{len(components)} disjoint copies" if isomorphic else "components differ"
    else:
        recovered["description"] = "one connected carrier"
    return recovered


# ==========================================================================
# Stage C: instantiation from the recovered specification only
# ==========================================================================


def instantiate(recovered: Mapping[str, Any], name: str) -> CarrierSpec:
    """Build a carrier from the recovered seams, rule and schedule alone."""

    rule = recovered["component"]["rule"]["class"]
    if rule not in (RULE_SEAM_MEAN, RULE_INTEGER, RULE_OVERWRITE):
        raise ValueError(f"recovered rule {rule!r} has no instantiation")
    if recovered["global_schedule"]["law"] != "uniform":
        raise ValueError("recovered schedule is not uniform; no instantiation declared")
    seams = tuple(sorted((int(i), int(j)) for i, j in recovered["structure"]["seams"]))
    carriers = int(recovered["components"]) if recovered["components_isomorphic"] else 1
    return CarrierSpec(name=name, ports=int(recovered["ports"]), seams=seams, rule=rule, carriers=carriers)


def log_text(log: Mapping[str, Any]) -> bytes:
    return canonical_bytes(log)


def run_loop(spec: CarrierSpec, seed: int, iterations: int = 2) -> dict[str, Any]:
    """Source run, recovery, instantiation, repeated ``iterations`` times."""

    stages = []
    logs: list[tuple[str, bytes]] = []
    text = log_text(run_source(spec, seed))
    recovered = recover_specification(json.loads(text.decode("ascii")))
    stages.append({"role": "inhabited_structure", "seed": seed, "log_sha256": sha256_bytes(text), "events": len(json.loads(text)["events"]), "recovered": recovered})
    logs.append(("inhabited_structure", text))
    current = recovered
    for k in range(1, iterations + 1):
        realized = instantiate(current, f"{spec.name}__realization_{k}")
        text = log_text(run_source(realized, seed + k))
        current = recover_specification(json.loads(text.decode("ascii")))
        role = "constructed_realization" if k == 1 else f"constructed_realization_iteration_{k}"
        stages.append({"role": role, "seed": seed + k, "log_sha256": sha256_bytes(text), "events": len(json.loads(text)["events"]), "recovered": current})
        logs.append((role, text))
    vectors = [s["recovered"]["invariants"] for s in stages]
    agree = all(v == vectors[0] for v in vectors[1:])
    terminal_kind = vectors[0]["terminal_invariant_type"]
    fixed_point = agree and terminal_kind != "none"
    return {
        "source_spec_name": spec.name,
        "iterations": iterations,
        "stages": stages,
        "invariant_vectors_agree": agree,
        "fixed_point": fixed_point,
        "fixed_point_level": terminal_kind if fixed_point else "none",
        "receipt": FIXED_POINT_RECEIPT if fixed_point else "NO_FIXED_POINT",
        "_logs": logs,
    }


# ==========================================================================
# Receipt
# ==========================================================================


def _compact(recovered: Mapping[str, Any]) -> dict[str, Any]:
    """The recovered specification without the per-port lists that inflate the receipt."""

    out = copy.deepcopy(dict(recovered))
    comp = out["component"]
    comp.pop("port_ids", None)
    out["structure"].pop("seams", None)
    return out


def negative_control_summary(loops: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def inv(name: str) -> Mapping[str, Any]:
        return loops[name]["stages"][0]["recovered"]["invariants"]

    ico, tet, octa = inv("icosahedron_seam_mean"), inv("tetrahedron_seam_mean"), inv("octahedron_seam_mean")
    integer, overwrite = inv("icosahedron_integer_nearest_agreement"), inv("icosahedron_overwrite")
    overwrite_rule = loops["icosahedron_overwrite"]["stages"][0]["recovered"]["component"]["rule"]
    integer_rule = loops["icosahedron_integer_nearest_agreement"]["stages"][0]["recovered"]["component"]["rule"]
    return {
        "tetrahedron": {
            "ports": tet["ports"],
            "seams": tet["seams"],
            "automorphism_order": tet["automorphism_order"],
            "rotation_order": tet["rotation_order"],
            "pairing_type": tet["pairing_type"],
            "farthest_pairing_distance": tet["farthest_pairing_distance"],
            "gram_rank": tet["gram_rank"],
            "lie_split": tet["lie_split"],
            "fixed_point": loops["tetrahedron_seam_mean"]["fixed_point"],
        },
        "octahedron": {
            "ports": octa["ports"],
            "seams": octa["seams"],
            "automorphism_order": octa["automorphism_order"],
            "rotation_order": octa["rotation_order"],
            "pairing_type": octa["pairing_type"],
            "farthest_pairing_distance": octa["farthest_pairing_distance"],
            "gram_rank": octa["gram_rank"],
            "lie_split": octa["lie_split"],
            "fixed_point": loops["octahedron_seam_mean"]["fixed_point"],
        },
        "overwrite": {
            "rule_class": overwrite["rule_class"],
            "nonconservative_events": overwrite_rule["nonconservative_events"],
            "terminal_schedule_independent": overwrite["terminal_schedule_independent"],
            "terminal_invariant_type": overwrite["terminal_invariant_type"],
            "invariant_vectors_agree": loops["icosahedron_overwrite"]["invariant_vectors_agree"],
            "fixed_point": loops["icosahedron_overwrite"]["fixed_point"],
        },
        "integer_nearest_agreement": {
            "rule_class": integer["rule_class"],
            "tie_ceiling_to_lower_index_frequency": integer_rule["tie_ceiling_to_lower_index_frequency"],
            "tie_law": integer_rule["tie_law"],
            "terminal_invariant_type": integer["terminal_invariant_type"],
            "fixed_point": loops["icosahedron_integer_nearest_agreement"]["fixed_point"],
            "fixed_point_level": loops["icosahedron_integer_nearest_agreement"]["fixed_point_level"],
        },
        "gram_rank_alone_does_not_select_icosahedron": ico["gram_rank"] == tet["gram_rank"] == octa["gram_rank"] == 3,
        "pairing_plus_rotation_order_selects_icosahedron": (
            (ico["pairing_type"], ico["rotation_order"]) == ("distance_three_involution", 60)
            and (tet["pairing_type"], tet["rotation_order"]) != ("distance_three_involution", 60)
            and (octa["pairing_type"], octa["rotation_order"]) != ("distance_three_involution", 60)
        ),
        "lie_split_by_carrier": {"icosahedron": ico["lie_split"], "tetrahedron": tet["lie_split"], "octahedron": octa["lie_split"]},
    }


def build_receipt(include_federation: bool = True) -> tuple[dict[str, Any], dict[str, bytes]]:
    loops: dict[str, Any] = {}
    stored: dict[str, bytes] = {}
    names = [n for n in LOOP_SEEDS if include_federation or n != "federation_isolated_20"]
    for name in names:
        result = run_loop(SOURCE_SPECS[name](), LOOP_SEEDS[name], LOOP_ITERATIONS[name])
        for role, text in result.pop("_logs"):
            if name != "federation_isolated_20":
                stored[f"{name}__{role}.json"] = text
        loops[name] = result
    firewall_logs = {}
    for name, result in loops.items():
        for stage in result["stages"]:
            key = f"{name}__{stage['role']}.json"
            firewall_logs[key] = {
                "sha256": stage["log_sha256"],
                "events": stage["events"],
                "stored": key in stored,
                "path": f"data/exact/closure_loop_logs/{key}" if key in stored else None,
            }
    canonical_loop = loops["icosahedron_seam_mean"]
    federation = loops.get("federation_isolated_20")
    federation_block: dict[str, Any] = {"status": "skipped"}
    if federation is not None:
        fed_inv = federation["stages"][0]["recovered"]["invariants"]
        ico_inv = dict(canonical_loop["stages"][0]["recovered"]["invariants"])
        comparable = {k: v for k, v in fed_inv.items() if k not in ("components", "components_isomorphic", "global_schedule_law")}
        ico_comparable = {k: v for k, v in ico_inv.items() if k not in ("components", "components_isomorphic", "global_schedule_law")}
        federation_block = {
            "status": "isolated_federation_recovered",
            "description": federation["stages"][0]["recovered"]["description"],
            "components": fed_inv["components"],
            "components_isomorphic": fed_inv["components_isomorphic"],
            "component_invariants_equal_single_carrier": comparable == ico_comparable,
            "global_schedule_law": fed_inv["global_schedule_law"],
            "fixed_point": federation["fixed_point"],
            "glued_federation": "work in progress (lane L1, oph_exact/federation.py)",
        }
    compact_loops = {}
    for name, result in loops.items():
        compact_loops[name] = {
            "source_spec_name": result["source_spec_name"],
            "iterations": result["iterations"],
            "stages": [
                {
                    "role": s["role"],
                    "seed": s["seed"],
                    "log_sha256": s["log_sha256"],
                    "events": s["events"],
                    "invariants": s["recovered"]["invariants"],
                }
                for s in result["stages"]
            ],
            "invariant_vectors_agree": result["invariant_vectors_agree"],
            "fixed_point": result["fixed_point"],
            "fixed_point_level": result["fixed_point_level"],
            "receipt": result["receipt"],
        }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "parameters": {
            "probe_steps": list(PROBE_STEPS),
            "probe_steps_scaled_by_carrier_count": True,
            "readback_digits": READBACK_DIGITS,
            "events_per_seam": EVENTS_PER_SEAM,
            "federation_events_per_seam": FEDERATION_EVENTS_PER_SEAM,
            "load_range": LOAD_RANGE,
            "federation_carriers": FEDERATION_CARRIERS,
            "replay_schedules": REPLAY_SCHEDULES,
            "replay_events_per_seam": REPLAY_EVENTS_PER_SEAM,
            "replay_seed_base": REPLAY_SEED_BASE,
            "chi_square_significance": CHI_SQUARE_SIGNIFICANCE,
            "tie_z_threshold": TIE_Z_THRESHOLD,
            "gram_rank_tolerance": GRAM_RANK_TOLERANCE,
            "gram_resolution_digits": GRAM_RESOLUTION_DIGITS,
            "terminal_tolerance_relative": str(TERMINAL_TOLERANCE),
            "probe_prediction_tolerance_margin_digits": PROBE_PREDICTION_TOLERANCE_MARGIN,
            "lie_simple_dimension_table": list(LIE_SIMPLE_DIMENSIONS),
            "descent_potential": "centered_squared_norm",
            "potential_reference": POTENTIAL_REFERENCE,
            "loop_seeds": dict(LOOP_SEEDS),
            "loop_iterations": dict(LOOP_ITERATIONS),
        },
        "firewall": {
            "statement": (
                "Stage B reads only the event log: port count, initial loads, per-event endpoint "
                "readings before and after, final loads, and the probe readback at declared "
                "resolution. The log carries no seam list, group, orientation, law description or "
                "schedule description. The recovery algorithm and its constants are fixed in the "
                "producer before any log is read; the same algorithm runs on every log."
            ),
            "logs": firewall_logs,
            "federation_log_stored": False,
            "federation_log_note": "producer-side only; digest pinned above; size exceeds the fixture budget",
        },
        "canonical_recovered_specification": _compact(canonical_loop["stages"][0]["recovered"]),
        "canonical_loop": compact_loops["icosahedron_seam_mean"],
        "loops": compact_loops,
        "negative_controls": negative_control_summary(loops),
        "federation": federation_block,
        "scope": {
            "declared": [
                "orientation: A1 declares one of the two orientation classes; the log fixes the class only up to global reversal",
                "schedule: uniform A3 over the seams, one directed label per event",
                "initial loads: seeded integers uniform in [0, load_range)",
                "Lie classification table: compact simple dimensions {3, 8, 10} with centre dimension at most one",
                "probe: readback after 2n synchronous expectation steps at the declared decimal resolution",
                "descent potential: centered squared norm of the readings, the flagship quadratic descent functional V(N) = sum_i N_i^2 up to the conserved total (see parameters.potential_reference)",
            ],
            "not_claimed": [
                "universe-level closure",
                "existence or uniqueness of the cosmic fixed point",
                "physical identification of any recovered invariant",
                "glued federation (work in progress under lane L1)",
                "records as the only observer access; the probe is a direct readback",
                "refinement across levels",
            ],
        },
        "claim_boundary": (
            "One exact carrier and the isolated federation of twenty carriers. From the event log "
            "alone the fixed recovery returns the seam set, the inverse-port pairing, the incidence "
            "automorphism group with its orientation-preserving subgroup, the repair rule class, "
            "the schedule law, the normalized response Gram with its rank, the Lie type dimension "
            "count and the terminal law. The instantiated carrier built from that specification "
            "returns the same invariant vector on two further iterations; that equality is the "
            "carrier-scale fixed point of the closure operator and nothing more. The tetrahedron "
            "and octahedron sources recover their own specifications, the overwrite source recovers "
            "a nonconservative rule without a unique terminal invariant and yields no fixed point, "
            "the integer source attains the fixed point on quotient invariants. The glued "
            "federation, records as the only observer access, refinement, and every physical "
            "identification are outside this receipt."
        ),
        "pins": {},
    }
    receipt = round_floats(receipt)
    return receipt, stored


def pin_files() -> dict[str, dict[str, str]]:
    pins = {}
    for label, path in (
        ("producer", PRODUCER_PATH),
        ("verifier", VERIFIER_PATH),
        ("test", TEST_PATH),
        ("carrier", CARRIER_PATH),
    ):
        pins[label] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path) if path.exists() else "missing"}
    return pins


def finalize_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    receipt = copy.deepcopy(receipt)
    receipt["pins"] = pin_files()
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = sha256_bytes(canonical_bytes(receipt))
    return receipt


def write_receipt(path: Path = RECEIPT_PATH, log_dir: Path = LOG_DIR, include_federation: bool = True) -> dict[str, Any]:
    receipt, stored = build_receipt(include_federation)
    log_dir.mkdir(parents=True, exist_ok=True)
    for name, text in stored.items():
        (log_dir / name).write_bytes(text)
    receipt = finalize_receipt(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(receipt))
    return receipt


def check_receipt(path: Path = RECEIPT_PATH, log_dir: Path = LOG_DIR, include_federation: bool = True) -> list[str]:
    """Rebuild and compare; the list of differences is empty on success."""

    problems: list[str] = []
    stored_receipt = json.loads(path.read_text(encoding="ascii"))
    rebuilt, stored_logs = build_receipt(include_federation)
    rebuilt = finalize_receipt(rebuilt)
    if canonical_bytes(rebuilt) != canonical_bytes(stored_receipt):
        for key in sorted(set(rebuilt) | set(stored_receipt)):
            if rebuilt.get(key) != stored_receipt.get(key):
                problems.append(f"receipt field differs: {key}")
    for name, text in stored_logs.items():
        on_disk = log_dir / name
        if not on_disk.exists():
            problems.append(f"missing log {name}")
        elif on_disk.read_bytes() != text:
            problems.append(f"log differs: {name}")
    return problems


def load_receipt(path: Path = RECEIPT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Closure loop at carrier scale (lane L4).")
    parser.add_argument("--write", action="store_true", help="run the loops and write the receipt and logs")
    parser.add_argument("--check", action="store_true", help="rebuild and compare with the stored receipt and logs")
    parser.add_argument("--output", type=Path, default=RECEIPT_PATH)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--no-federation", action="store_true", help="development flag: skip the federation hook")
    args = parser.parse_args(argv)
    if args.write == args.check:
        parser.error("choose exactly one of --write / --check")
    if args.write:
        receipt = write_receipt(args.output, args.log_dir, not args.no_federation)
        print(json.dumps({"output": str(args.output), "status": receipt["status"], "canonical_fixed_point": receipt["canonical_loop"]["fixed_point"]}, sort_keys=True))
        return 0
    problems = check_receipt(args.output, args.log_dir, not args.no_federation)
    if problems:
        for p in problems:
            print(p)
        return 1
    print("CLOSURE_LOOP_RECEIPT_REPRODUCED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
