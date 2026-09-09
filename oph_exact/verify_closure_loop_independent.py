"""Independent verifier for the closure-loop receipt (lane L4).

The verifier does not import the producer.  It carries its own copy of the
fixed recovery algorithm, re-runs it on every stored event log, recomputes
the invariant vectors, and checks the receipt's equality claims, the
negative-control statements, the firewall digests, the self digest and the
file pins.  The federation log is pinned by digest only; its block is checked
for internal consistency against the receipt's own stage vectors.
"""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
from itertools import combinations, combinations_with_replacement
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import chi2 as _chi2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = ROOT / "data" / "exact" / "closure_loop_receipt.json"
DEFAULT_LOG_DIR = ROOT / "data" / "exact" / "closure_loop_logs"
SCHEMA = "oph.exact.closure-loop.v1"
LOG_SCHEMA = "oph.exact.closure-loop.event-log.v1"
FIXED_POINT_RECEIPT = "CLOSURE_FIXED_POINT_AT_CARRIER_SCALE"
LOG_KEYS = {"schema", "ports", "carriers", "ports_per_carrier", "initial", "events", "final", "probe"}

# The verifier's own copy of the recovery constants; the receipt must declare the same values.
EXPECTED_PARAMETERS = {
    "replay_schedules": 8,
    "replay_events_per_seam": 32,
    "replay_seed_base": 7919,
    "chi_square_significance": 1e-3,
    "tie_z_threshold": 3.29,
    "gram_rank_tolerance": 1e-9,
    "gram_resolution_digits": 12,
    "terminal_tolerance_relative": "1/1000000000",
    "probe_prediction_tolerance_margin_digits": 3,
    "lie_simple_dimension_table": [3, 8, 10],
}
REPLAY_SCHEDULES = 8
REPLAY_EVENTS_PER_SEAM = 32
REPLAY_SEED_BASE = 7919
CHI_SQUARE_SIGNIFICANCE = 1e-3
TIE_Z_THRESHOLD = 3.29
GRAM_RANK_TOLERANCE = 1e-9
GRAM_RESOLUTION_DIGITS = 12
TERMINAL_TOLERANCE = Fraction(1, 10**9)
PREDICTION_MARGIN = 3
LIE_SIMPLE_DIMENSIONS = (3, 8, 10)
CLUSTER_TOL = 1e-6

RULE_SEAM_MEAN = "seam_mean"
RULE_INTEGER = "integer_nearest_agreement"
RULE_OVERWRITE = "overwrite"


class IndependentVerificationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise IndependentVerificationError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ==========================================================================
# Q(sqrt 5)
# ==========================================================================

Q5 = tuple[Fraction, Fraction]
ZERO: Q5 = (Fraction(0), Fraction(0))
ONE: Q5 = (Fraction(1), Fraction(0))


def q5(a: int | Fraction, b: int | Fraction = 0) -> Q5:
    return (Fraction(a), Fraction(b))


def add(x: Q5, y: Q5) -> Q5:
    return (x[0] + y[0], x[1] + y[1])


def sub(x: Q5, y: Q5) -> Q5:
    return (x[0] - y[0], x[1] - y[1])


def mul(x: Q5, y: Q5) -> Q5:
    return (x[0] * y[0] + 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def inv(x: Q5) -> Q5:
    norm = x[0] * x[0] - 5 * x[1] * x[1]
    if norm == 0:
        raise ZeroDivisionError
    return (x[0] / norm, -x[1] / norm)


def q5sum(values: Any) -> Q5:
    acc = ZERO
    for v in values:
        acc = add(acc, v)
    return acc


def matmul(x: Sequence[Sequence[Q5]], y: Sequence[Sequence[Q5]]) -> list[list[Q5]]:
    n, k, m = len(x), len(y), len(y[0])
    out = [[ZERO] * m for _ in range(n)]
    for i in range(n):
        for t in range(k):
            if x[i][t] == ZERO:
                continue
            for j in range(m):
                out[i][j] = add(out[i][j], mul(x[i][t], y[t][j]))
    return out


def nullity(matrix: Sequence[Sequence[Q5]]) -> int:
    rows = [list(r) for r in matrix]
    n_rows, n_cols = len(rows), len(rows[0])
    rank = 0
    col = 0
    while rank < n_rows and col < n_cols:
        pivot = next((r for r in range(rank, n_rows) if rows[r][col] != ZERO), None)
        if pivot is None:
            col += 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = inv(rows[rank][col])
        rows[rank] = [mul(v, scale) for v in rows[rank]]
        for r in range(n_rows):
            if r != rank and rows[r][col] != ZERO:
                f = rows[r][col]
                rows[r] = [sub(rows[r][c], mul(f, rows[rank][c])) for c in range(n_cols)]
        rank += 1
        col += 1
    return n_cols - rank


# ==========================================================================
# Integer matrix power (repeated squaring; the stored logs are single carriers)
# ==========================================================================


def int_matmul(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> list[list[int]]:
    n = len(a)
    bt = [[b[k][j] for k in range(n)] for j in range(n)]
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def int_matpow(m: Sequence[Sequence[int]], e: int) -> list[list[int]]:
    n = len(m)
    result = [[int(i == j) for j in range(n)] for i in range(n)]
    base = [list(r) for r in m]
    while e:
        if e & 1:
            result = int_matmul(result, base)
        e >>= 1
        if e:
            base = int_matmul(base, base)
    return result


def expectation_power(ports: int, seams: Sequence[tuple[int, int]], total_seams: int, steps: int) -> tuple[list[list[int]], int]:
    two_s = 2 * total_seams
    lap = [[0] * ports for _ in range(ports)]
    for i, j in seams:
        lap[i][i] += 1
        lap[j][j] += 1
        lap[i][j] -= 1
        lap[j][i] -= 1
    m = [[(two_s if r == c else 0) - lap[r][c] for c in range(ports)] for r in range(ports)]
    return int_matpow(m, steps), two_s**steps


# ==========================================================================
# Recovery (the verifier's own copy)
# ==========================================================================


def events_table(log: Mapping[str, Any]) -> list[tuple[int, int, Fraction, Fraction, Fraction, Fraction]]:
    ports = int(log["ports"])
    table = []
    for event in log["events"]:
        changed = event["changed"]
        if len(changed) != 2:
            _fail("event touches a number of ports different from two")
        (a, va), (b, vb) = sorted(((int(k), v) for k, v in changed.items()), key=lambda kv: kv[0])
        if not (0 <= a < b < ports):
            _fail("event port outside the port range")
        table.append((a, b, Fraction(va[0]), Fraction(va[1]), Fraction(vb[0]), Fraction(vb[1])))
    return table


def components(ports: int, seams: Sequence[tuple[int, int]]) -> list[list[int]]:
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
    return [sorted(g) for g in sorted(groups.values(), key=min)]


def adjacency(ports: int, seams: Sequence[tuple[int, int]]) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(ports)]
    for i, j in seams:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def distances(ports: int, adj: Sequence[set[int]]) -> list[list[int]]:
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


def pairing(ports: int, seams: Sequence[tuple[int, int]]) -> dict[str, Any]:
    dist = distances(ports, adjacency(ports, seams))

    def involution(cands: list[list[int]]) -> list[int] | None:
        if any(len(c) != 1 for c in cands):
            return None
        p = [c[0] for c in cands]
        if any(p[i] == i or p[p[i]] != i for i in range(ports)):
            return None
        return p

    three = involution([[q for q in range(ports) if dist[p][q] == 3] for p in range(ports)])
    ecc = [max(dist[p]) for p in range(ports)]
    farthest = None
    if len(set(ecc)) == 1 and ecc[0] > 0:
        far = involution([[q for q in range(ports) if dist[p][q] == ecc[0]] for p in range(ports)])
        if far is not None:
            farthest = ecc[0]
    return {"type": "distance_three_involution" if three is not None else "none", "farthest_distance": farthest}


def automorphisms(ports: int, seams: Sequence[tuple[int, int]]) -> list[tuple[int, ...]]:
    adj = adjacency(ports, seams)
    degree = [len(a) for a in adj]
    found: list[tuple[int, ...]] = []

    def extend(assign: list[int], used: set[int]) -> None:
        p = len(assign)
        if p == ports:
            found.append(tuple(assign))
            return
        for image in range(ports):
            if image in used or degree[image] != degree[p]:
                continue
            if all((q in adj[p]) == (assign[q] in adj[image]) for q in range(p)):
                assign.append(image)
                used.add(image)
                extend(assign, used)
                assign.pop()
                used.discard(image)

    extend([], set())
    return found


def isomorphic(m: int, seams_a: Sequence[tuple[int, int]], seams_b: Sequence[tuple[int, int]]) -> bool:
    adj_a, adj_b = adjacency(m, seams_a), adjacency(m, seams_b)

    def extend(assign: list[int], used: set[int]) -> bool:
        p = len(assign)
        if p == m:
            return True
        for image in range(m):
            if image in used or len(adj_a[p]) != len(adj_b[image]):
                continue
            if all((q in adj_a[p]) == (assign[q] in adj_b[image]) for q in range(p)):
                assign.append(image)
                used.add(image)
                if extend(assign, used):
                    return True
                assign.pop()
                used.discard(image)
        return False

    return extend([], set())


def cyc(face: Sequence[int]) -> tuple[int, ...]:
    face = tuple(face)
    k = face.index(min(face))
    return face[k:] + face[:k]


def rotations(ports: int, seams: Sequence[tuple[int, int]], autos: Sequence[tuple[int, ...]]) -> list[tuple[int, ...]] | None:
    adj = adjacency(ports, seams)
    tris = sorted({tuple(sorted((a, b, c))) for a in range(ports) for b in adj[a] if b > a for c in adj[a] & adj[b] if c > b})
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for t, (a, b, c) in enumerate(tris):
        for e in ((a, b), (a, c), (b, c)):
            edge_faces.setdefault(e, []).append(t)
    if not tris or any(len(v) != 2 for v in edge_faces.values()) or set(edge_faces) != set(seams):
        return None
    oriented: dict[int, tuple[int, int, int]] = {0: tris[0]}
    stack = [0]
    while stack:
        t = stack.pop()
        a, b, c = oriented[t]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            other = next(s for s in edge_faces[key] if s != t)
            w = next(x for x in tris[other] if x not in (u, v))
            want = (v, u, w)
            if other in oriented:
                if cyc(oriented[other]) != cyc(want):
                    return None
            else:
                oriented[other] = want
                stack.append(other)
    if len(oriented) != len(tris):
        return None
    cls = {cyc(f) for f in oriented.values()}
    rev = {cyc((f[0], f[2], f[1])) for f in oriented.values()}
    proper = [g for g in autos if all(cyc((g[a], g[b], g[c])) in cls for a, b, c in cls)]
    proper_rev = [g for g in autos if all(cyc((g[a], g[b], g[c])) in rev for a, b, c in rev)]
    if set(proper) != set(proper_rev):
        _fail("orientation classes give different proper subgroups")
    return proper


def rule_class(table: Sequence[tuple[int, int, Fraction, Fraction, Fraction, Fraction]]) -> dict[str, Any]:
    conservative = discriminating = mean_form = shell_form = overwrite_form = odd_total = ceiling_lower = 0
    integer_values = True
    for a, b, bi, ai, bj, aj in table:
        if ai + aj == bi + bj:
            conservative += 1
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
                        ceiling_lower += 1
        if (ai == bi and aj == bi) or (aj == bj and ai == bj):
            overwrite_form += 1
    n = len(table)
    nonconservative = n - conservative
    if discriminating == 0:
        rule = "unclassified"
    elif nonconservative == 0 and mean_form == discriminating:
        rule = RULE_SEAM_MEAN
    elif nonconservative == 0 and integer_values and shell_form == discriminating:
        rule = RULE_INTEGER
    elif overwrite_form == discriminating:
        rule = RULE_OVERWRITE
    else:
        rule = "unclassified"
    tie_law = None
    if rule == RULE_INTEGER and odd_total > 0:
        f = Fraction(ceiling_lower, odd_total)
        z = abs(float(f) - 0.5) * 2.0 * odd_total**0.5
        tie_law = "uniform" if z <= TIE_Z_THRESHOLD else "biased"
    return {"class": rule, "nonconservative": nonconservative, "tie_law": tie_law}


def schedule_law(table: Sequence[tuple[int, int, Fraction, Fraction, Fraction, Fraction]], seams: Sequence[tuple[int, int]]) -> str:
    counts = {tuple(s): 0 for s in seams}
    for a, b, *_ in table:
        counts[(a, b)] += 1
    n, s = len(table), len(seams)
    statistic = Fraction(s, n) * sum(Fraction(c * c) for c in counts.values()) - n if n else Fraction(0)
    df = s - 1
    threshold = float(_chi2.ppf(1.0 - CHI_SQUARE_SIGNIFICANCE, df)) if df > 0 else 0.0
    return "uniform" if float(statistic) <= threshold else "nonuniform"


def gram(comp: Sequence[int], comp_seams: Sequence[tuple[int, int]], total_seams: int, probe: Mapping[str, Any]) -> dict[str, Any]:
    m = len(comp)
    local = {p: k for k, p in enumerate(comp)}
    steps = [int(n) for n in probe["steps"]]
    digits = int(probe["readback_digits"])
    tol_scale = 10 ** (digits - PREDICTION_MARGIN)
    ranks: dict[int, int | None] = {}
    eig_by: dict[int, list[float]] = {}
    prediction_ok = True
    for n in steps:
        vectors = [{int(q): Fraction(v) for q, v in vec.items()} for vec in probe["readback"][str(n)]]
        r = [[Fraction(0)] * m for _ in range(m)]
        for p in comp:
            for q, v in vectors[p].items():
                if q not in local:
                    _fail("probe support leaves the component")
                r[local[q]][local[p]] = v
        row_mean = [sum(row) / m for row in r]
        col_mean = [sum(r[a][b] for a in range(m)) / m for b in range(m)]
        total = sum(row_mean) / m
        c = [[r[a][b] - row_mean[a] - col_mean[b] + total for b in range(m)] for a in range(m)]
        raw_scale = max(abs(v) for row in r for v in row)
        centered_scale = max(abs(v) for row in c for v in row)
        trace = sum(c[a][a] for a in range(m))
        ranks[n] = None
        if centered_scale != 0 and raw_scale != 0 and trace != 0:
            resolved = digits + float(np.log10(float(centered_scale / raw_scale)))
            if resolved >= GRAM_RESOLUTION_DIGITS:
                k = np.array([[float(c[a][b] * m / trace) for b in range(m)] for a in range(m)])
                eig = sorted(np.linalg.eigvalsh(0.5 * (k + k.T)).tolist(), reverse=True)
                top = max(abs(e) for e in eig)
                ranks[n] = int(sum(1 for e in eig if e > GRAM_RANK_TOLERANCE * top))
                eig_by[n] = eig
        power, den = expectation_power(m, [(local[i], local[j]) for i, j in comp_seams], total_seams, 2 * n)
        for p in comp:
            for q in comp:
                pred = power[local[q]][local[p]]
                obs = vectors[p].get(q, Fraction(0))
                if pred == 0:
                    ok = obs == 0
                else:
                    ok = abs(obs.numerator * den - pred * obs.denominator) * tol_scale <= abs(pred * obs.denominator)
                prediction_ok = prediction_ok and ok
    resolved_steps = [n for n in steps if ranks[n] is not None]
    chosen = max(resolved_steps) if resolved_steps else None
    return {
        "eigenvalues": None if chosen is None else [round(v, 9) + 0.0 for v in eig_by[chosen]],
        "rank": None if chosen is None else ranks[chosen],
        "step_index": None if chosen is None else steps.index(chosen),
        "prediction_ok": prediction_ok,
    }


def compose(g: Sequence[int], h: Sequence[int]) -> tuple[int, ...]:
    return tuple(g[h[p]] for p in range(len(g)))


def lie_type(ports: int, seams: Sequence[tuple[int, int]], group: Sequence[tuple[int, ...]] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"split": "unidentified", "assignment_count": 0, "trivial_isotypic_dimension": None}
    if not group:
        out["split"] = "no_rotation_group"
        return out
    lap_int = [[0] * ports for _ in range(ports)]
    for i, j in seams:
        lap_int[i][i] += 1
        lap_int[j][j] += 1
        lap_int[i][j] -= 1
        lap_int[j][i] -= 1
    lap = [[q5(lap_int[r][c]) for c in range(ports)] for r in range(ports)]
    clusters: list[list[float]] = []
    for e in sorted(np.linalg.eigvalsh(np.array(lap_int, dtype=float)).tolist()):
        if clusters and abs(e - clusters[-1][0]) < CLUSTER_TOL:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    exact: list[tuple[Q5, int]] = []
    bound = 4 * ports
    for cl in clusters:
        value = sum(cl) / len(cl)
        found = None
        for kb in range(0, bound + 1):
            for sign_b in ((1,) if kb == 0 else (1, -1)):
                b = Fraction(sign_b * kb, 2)
                for ka in range(-bound, bound + 1):
                    a = Fraction(ka, 2)
                    if abs(float(a) + float(b) * 5.0**0.5 - value) < CLUSTER_TOL:
                        mu = (a, b)
                        shifted = [[sub(lap[r][c], mu) if r == c else lap[r][c] for c in range(ports)] for r in range(ports)]
                        if nullity(shifted) == len(cl):
                            found = mu
                            break
                if found:
                    break
            if found:
                break
        if found is None:
            out["split"] = "band_not_in_q_sqrt5"
            return out
        exact.append((found, len(cl)))
    identity = [[q5(int(r == c)) for c in range(ports)] for r in range(ports)]
    projectors = []
    for k, (mu_k, _) in enumerate(exact):
        acc = [row[:] for row in identity]
        for m_, (mu_m, _) in enumerate(exact):
            if m_ == k:
                continue
            factor = [[sub(lap[r][c], mu_m) if r == c else lap[r][c] for c in range(ports)] for r in range(ports)]
            acc = matmul(acc, factor)
            scale = inv(sub(mu_k, mu_m))
            acc = [[mul(v, scale) for v in row] for row in acc]
        projectors.append(acc)
    order = len(group)
    identity_perm = tuple(range(ports))
    perm_char = {g: sum(int(g[p] == p) for p in range(ports)) for g in group}
    trivial_dim = Fraction(sum(perm_char.values()), order)
    out["trivial_isotypic_dimension"] = int(trivial_dim) if trivial_dim.denominator == 1 else str(trivial_dim)
    chars = []
    irreducible = True
    trivial_bands = []
    for b, ((mu, mult), proj) in enumerate(zip(exact, projectors)):
        ch = {g: q5sum(proj[q][g[q]] for q in range(ports)) for g in group}
        if q5sum(mul(v, v) for v in ch.values()) != (Fraction(order), Fraction(0)):
            irreducible = False
        if mult == 1 and all(v == ONE for v in ch.values()):
            trivial_bands.append(b)
        chars.append(ch)
    if not irreducible:
        out["split"] = "bands_not_irreducible"
        return out

    def uchar(bands: frozenset[int], g: tuple[int, ...]) -> Q5:
        return q5sum(chars[b][g] for b in bands)

    def udim(bands: frozenset[int]) -> int:
        return sum(exact[b][1] for b in bands)

    pow_cache: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {}

    def ppow(g: tuple[int, ...], k: int) -> tuple[int, ...]:
        if (g, k) not in pow_cache:
            h = identity_perm
            for _ in range(k):
                h = compose(g, h)
            pow_cache[(g, k)] = h
        return pow_cache[(g, k)]

    def det(bands: frozenset[int], g: tuple[int, ...]) -> Q5:
        d = udim(bands)
        p = [None] + [uchar(bands, ppow(g, k)) for k in range(1, d + 1)]
        e: list[Q5] = [ONE]
        for k in range(1, d + 1):
            acc = ZERO
            for i in range(1, k + 1):
                term = mul(e[k - i], p[i])
                acc = add(acc, term) if i % 2 == 1 else sub(acc, term)
            e.append((acc[0] / k, acc[1] / k))
        return e[d]

    so_cache: dict[frozenset[int], bool] = {}

    def so(bands: frozenset[int]) -> bool:
        if bands not in so_cache:
            so_cache[bands] = all(det(bands, g) == ONE for g in group)
        return so_cache[bands]

    nb = len(exact)
    unions = [frozenset(c) for k in range(1, nb + 1) for c in combinations(range(nb), k)]
    threes = [u for u in unions if udim(u) == 3 and so(u)]
    fives = [u for u in unions if udim(u) == 5 and so(u)]

    def passes(bands: frozenset[int], dim: int) -> bool:
        if dim == 3:
            return so(bands)
        if dim == 8:
            if all(uchar(bands, g) == q5(8) for g in group):
                return True
            return any(all(uchar(bands, g) == sub(mul(uchar(rho, g), uchar(rho, g)), ONE) for g in group) for rho in threes)
        if dim == 10:
            if all(uchar(bands, g) == q5(10) for g in group):
                return True
            for sigma in fives:
                if all(
                    uchar(bands, g)
                    == tuple(v / 2 for v in sub(mul(uchar(sigma, g), uchar(sigma, g)), uchar(sigma, ppow(g, 2))))
                    for g in group
                ):
                    return True
        return False

    candidates: list[tuple[int, tuple[int, ...]]] = []
    for centre in (0, 1):
        rem = ports - centre
        for k in range(1, rem // min(LIE_SIMPLE_DIMENSIONS) + 1):
            for parts in combinations_with_replacement(LIE_SIMPLE_DIMENSIONS, k):
                if sum(parts) == rem:
                    candidates.append((centre, parts))
    survivors = []
    for centre, parts in candidates:
        if centre > int(trivial_dim):
            continue
        seen: set[tuple[Any, ...]] = set()
        count = 0

        def assign(idx: int, free: frozenset[int], chosen: list[frozenset[int]]) -> None:
            nonlocal count
            if idx == len(parts):
                if free:
                    return
                key = tuple(sorted((parts[i], tuple(sorted(chosen[i]))) for i in range(len(parts))))
                if key in seen:
                    return
                seen.add(key)
                if all(passes(chosen[i], parts[i]) for i in range(len(parts))):
                    count += 1
                return
            for k in range(1, len(free) + 1):
                for c in combinations(sorted(free), k):
                    if udim(frozenset(c)) == parts[idx]:
                        assign(idx + 1, free - frozenset(c), chosen + [frozenset(c)])

        for cb in ([frozenset()] if centre == 0 else [frozenset([b]) for b in trivial_bands]):
            assign(0, frozenset(range(nb)) - cb, [])
        if count:
            survivors.append(("+".join([str(centre)] * (centre > 0) + [str(x) for x in parts]), count))
    if len(survivors) == 1:
        out["split"], out["assignment_count"] = survivors[0]
    elif not survivors:
        out["split"] = "none"
    else:
        out["split"] = "ambiguous:" + ",".join(s[0] for s in survivors)
        out["assignment_count"] = sum(s[1] for s in survivors)
    return out


def descent_violations(comp: Sequence[int], comp_table: Sequence[tuple[int, int, Fraction, Fraction, Fraction, Fraction]], initial: Sequence[Fraction]) -> int:
    local = {p: k for k, p in enumerate(comp)}
    m = len(comp)
    state = [initial[p] for p in comp]
    total = sum(state)
    sum_sq = sum(v * v for v in state)
    violations = 0
    for a, b, bi, ai, bj, aj in comp_table:
        i, j = local[a], local[b]
        v_before = sum_sq - total * total / m
        changed = sorted((bi, bj)) != sorted((ai, aj))
        state[i], state[j] = ai, aj
        total += (ai - bi) + (aj - bj)
        sum_sq += (ai * ai - bi * bi) + (aj * aj - bj * bj)
        v_after = sum_sq - total * total / m
        if changed and not v_after < v_before:
            violations += 1
    return violations


def replay(rule: str, xi: Fraction, xj: Fraction, direction: int) -> tuple[Fraction, Fraction]:
    if rule == RULE_SEAM_MEAN:
        mid = (xi + xj) / 2
        return mid, mid
    if rule == RULE_INTEGER:
        s = int(xi) + int(xj)
        low, high = s // 2, -((-s) // 2)
        return (Fraction(high), Fraction(low)) if direction == 0 else (Fraction(low), Fraction(high))
    if rule == RULE_OVERWRITE:
        return (xi, xi) if direction == 0 else (xj, xj)
    raise ValueError(rule)


def terminal(comp: Sequence[int], comp_seams: Sequence[tuple[int, int]], rule: str, initial: Sequence[Fraction]) -> dict[str, Any]:
    loads = [initial[p] for p in comp]
    m = len(comp)
    mean = sum(loads) / m
    spread = max(loads) - min(loads)
    tol = TERMINAL_TOLERANCE * spread if spread > 0 else Fraction(0)
    local = {p: k for k, p in enumerate(comp)}
    seams = [(local[i], local[j]) for i, j in comp_seams]
    terminals = []
    if rule in (RULE_SEAM_MEAN, RULE_INTEGER, RULE_OVERWRITE) and seams:
        events = REPLAY_EVENTS_PER_SEAM * len(seams)
        for k in range(REPLAY_SCHEDULES):
            rng = random.Random(REPLAY_SEED_BASE + k)
            state = list(loads)
            labels = 2 * len(seams)
            for _ in range(events):
                label = rng.randrange(labels)
                i, j = seams[label // 2]
                state[i], state[j] = replay(rule, state[i], state[j], label % 2)
            terminals.append(state)
    equals_mean = bool(terminals) and all(max(abs(v - mean) for v in t) <= tol for t in terminals)
    independent = bool(terminals) and all(max(abs(x - y) for x, y in zip(terminals[0], t)) <= tol for t in terminals[1:])
    quotient = bool(terminals) and all(
        max(abs(x - y) for x, y in zip(sorted(terminals[0]), sorted(t))) <= tol for t in terminals[1:]
    )
    kind = "state" if equals_mean and independent else ("quotient_multiset" if quotient else "none")
    return {
        "terminal_equals_initial_mean": equals_mean,
        "terminal_schedule_independent": independent,
        "quotient_multiset_schedule_independent": quotient,
        "terminal_invariant_type": kind,
    }


def recover_invariants(log: Mapping[str, Any]) -> dict[str, Any]:
    """The verifier's recovery: the invariant vector of one log."""

    if set(log) != LOG_KEYS or log["schema"] != LOG_SCHEMA:
        _fail("log keys or schema differ from the declared firewall format")
    ports = int(log["ports"])
    table = events_table(log)
    seams = sorted({(a, b) for a, b, *_ in table})
    initial = [Fraction(v) for v in log["initial"]]
    comps = components(ports, seams)
    vectors = []
    comp_seam_lists = []
    for comp in comps:
        cs = set(comp)
        comp_seams = [s for s in seams if s[0] in cs]
        comp_table = [row for row in table if row[0] in cs]
        local = {p: k for k, p in enumerate(comp)}
        m = len(comp)
        local_seams = [(local[i], local[j]) for i, j in comp_seams]
        comp_seam_lists.append(local_seams)
        degree = [0] * m
        for i, j in local_seams:
            degree[i] += 1
            degree[j] += 1
        autos = automorphisms(m, local_seams)
        rots = rotations(m, local_seams, autos)
        rule = rule_class(comp_table)
        g = gram(comp, comp_seams, len(seams), log["probe"])
        lie = lie_type(m, local_seams, rots)
        term = terminal(comp, comp_seams, rule["class"], initial)
        pair = pairing(m, local_seams)
        vectors.append(
            {
                "ports": m,
                "seams": len(comp_seams),
                "degree_sequence": sorted(degree),
                "pairing_type": pair["type"],
                "farthest_pairing_distance": pair["farthest_distance"],
                "automorphism_order": len(autos),
                "rotation_order": None if rots is None else len(rots),
                "rule_class": rule["class"],
                "nonconservative_events_present": rule["nonconservative"] > 0,
                "tie_law": rule["tie_law"],
                "schedule_law": schedule_law(comp_table, comp_seams),
                "gram_eigenvalues": g["eigenvalues"],
                "gram_rank": g["rank"],
                "gram_step_index": g["step_index"],
                "probe_prediction_within_tolerance": g["prediction_ok"],
                "lie_split": lie["split"],
                "lie_assignment_count": lie["assignment_count"],
                "trivial_isotypic_dimension": lie["trivial_isotypic_dimension"],
                "strict_descent_violation_count": descent_violations(comp, comp_table, initial),
                **term,
            }
        )
    identical = all(v == vectors[0] for v in vectors[1:])
    iso = identical and all(isomorphic(len(comps[0]), comp_seam_lists[0], other) for other in comp_seam_lists[1:])
    inv = dict(vectors[0])
    inv["components"] = len(comps)
    inv["components_isomorphic"] = iso
    inv["global_schedule_law"] = schedule_law(table, seams)
    return inv


# ==========================================================================
# Receipt checks
# ==========================================================================


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="ascii"))


def verify_receipt(receipt: Mapping[str, Any], root: Path = ROOT, log_dir: Path = DEFAULT_LOG_DIR, check_pins: bool = True) -> dict[str, Any]:
    if receipt.get("schema") != SCHEMA:
        _fail("schema differs")
    payload = copy.deepcopy(dict(receipt))
    digest = payload.pop("receipt_sha256", None)
    if digest != sha256_bytes(canonical_bytes(payload)):
        _fail("receipt self digest differs")
    if canonical_bytes(json.loads(canonical_bytes(receipt).decode("ascii"))) != canonical_bytes(receipt):
        _fail("receipt is not canonical JSON")
    params = receipt["parameters"]
    for key, value in EXPECTED_PARAMETERS.items():
        if params.get(key) != value:
            _fail(f"parameter {key} differs from the verifier's declared value")
    for key in ("firewall", "loops", "canonical_loop", "negative_controls", "federation", "scope", "claim_boundary", "pins", "status", "canonical_recovered_specification"):
        if key not in receipt:
            _fail(f"missing block {key}")
    reference = params.get("potential_reference")
    if params.get("descent_potential") != "centered_squared_norm" or not isinstance(reference, str) or "N_i^2" not in reference or "2(d-1)" not in reference:
        _fail("descent potential declaration or its flagship reference is missing")
    if receipt["canonical_recovered_specification"]["component"]["descent"].get("potential_reference") != reference:
        _fail("descent block reference differs from the parameters block")
    if not receipt["scope"]["declared"] or not receipt["scope"]["not_claimed"]:
        _fail("scope lists must be nonempty")
    if check_pins:
        for label, pin in receipt["pins"].items():
            path = root / pin["path"]
            if not path.exists():
                _fail(f"pinned file missing: {label}")
            if label == "producer" and "closure_loop.py" not in pin["path"]:
                _fail("producer pin points elsewhere")
            if sha256_bytes(path.read_bytes()) != pin["sha256"]:
                _fail(f"pin differs: {label}")
    loops = receipt["loops"]
    if receipt["canonical_loop"] != loops["icosahedron_seam_mean"]:
        _fail("canonical loop block differs from the loops entry")
    recomputed = 0
    for key, info in receipt["firewall"]["logs"].items():
        name, role = key[: -len(".json")].split("__", 1)
        stage = next(s for s in loops[name]["stages"] if s["role"] == role)
        if stage["log_sha256"] != info["sha256"] or stage["events"] != info["events"]:
            _fail(f"firewall digest or event count differs for {key}")
        if not info["stored"]:
            continue
        path = log_dir / key
        if not path.exists():
            _fail(f"stored log missing: {key}")
        data = path.read_bytes()
        if sha256_bytes(data) != info["sha256"]:
            _fail(f"stored log digest differs: {key}")
        log = json.loads(data.decode("ascii"))
        if len(log["events"]) != info["events"]:
            _fail(f"event count differs for {key}")
        invariants = recover_invariants(log)
        if invariants != stage["invariants"]:
            diff = sorted(k for k in set(invariants) | set(stage["invariants"]) if invariants.get(k) != stage["invariants"].get(k))
            _fail(f"recomputed invariants differ for {key}: {diff}")
        recomputed += 1
    if recomputed == 0:
        _fail("no stored log was recomputed")
    for name, loop in loops.items():
        vectors = [s["invariants"] for s in loop["stages"]]
        agree = all(v == vectors[0] for v in vectors[1:])
        if loop["invariant_vectors_agree"] != agree:
            _fail(f"agreement claim differs for {name}")
        kind = vectors[0]["terminal_invariant_type"]
        fixed = agree and kind != "none"
        if loop["fixed_point"] != fixed or loop["fixed_point_level"] != (kind if fixed else "none"):
            _fail(f"fixed point claim differs for {name}")
        if loop["receipt"] != (FIXED_POINT_RECEIPT if fixed else "NO_FIXED_POINT"):
            _fail(f"receipt string differs for {name}")
        if len(loop["stages"]) != loop["iterations"] + 1:
            _fail(f"stage count differs for {name}")
    canonical = loops["icosahedron_seam_mean"]
    if not canonical["fixed_point"] or canonical["fixed_point_level"] != "state":
        _fail("canonical loop is not a state-level fixed point")
    first = canonical["stages"][0]["invariants"]
    expected_canonical = {
        "ports": 12,
        "seams": 30,
        "pairing_type": "distance_three_involution",
        "automorphism_order": 120,
        "rotation_order": 60,
        "rule_class": RULE_SEAM_MEAN,
        "gram_rank": 3,
        "lie_split": "1+3+8",
        "lie_assignment_count": 2,
        "strict_descent_violation_count": 0,
        "components": 1,
    }
    for k, v in expected_canonical.items():
        if first.get(k) != v:
            _fail(f"canonical invariant {k} differs from {v!r}")
    if receipt["canonical_recovered_specification"]["invariants"] != first:
        _fail("canonical recovered specification invariants differ from stage 0")
    if loops["icosahedron_overwrite"]["fixed_point"]:
        _fail("overwrite loop must not be a fixed point")
    if loops["icosahedron_overwrite"]["stages"][0]["invariants"]["rule_class"] != RULE_OVERWRITE:
        _fail("overwrite source not recovered as overwrite")
    integer = loops["icosahedron_integer_nearest_agreement"]
    if not (integer["fixed_point"] and integer["fixed_point_level"] == "quotient_multiset"):
        _fail("integer loop must be a quotient-level fixed point")
    if integer["stages"][0]["invariants"]["rule_class"] != RULE_INTEGER:
        _fail("integer source not recovered as integer nearest agreement")
    tet = loops["tetrahedron_seam_mean"]["stages"][0]["invariants"]
    octa = loops["octahedron_seam_mean"]["stages"][0]["invariants"]
    if (tet["ports"], tet["seams"], tet["automorphism_order"], tet["rotation_order"], tet["pairing_type"], tet["gram_rank"], tet["lie_split"]) != (4, 6, 24, 12, "none", 3, "1+3"):
        _fail("tetrahedron control differs")
    if (octa["ports"], octa["seams"], octa["automorphism_order"], octa["rotation_order"], octa["pairing_type"], octa["farthest_pairing_distance"], octa["gram_rank"], octa["lie_split"]) != (6, 12, 48, 24, "none", 2, 3, "none"):
        _fail("octahedron control differs")
    controls = receipt["negative_controls"]
    if controls["gram_rank_alone_does_not_select_icosahedron"] is not True or controls["pairing_plus_rotation_order_selects_icosahedron"] is not True:
        _fail("negative control selection statements differ")
    if controls["overwrite"]["fixed_point"] is not False or controls["overwrite"]["nonconservative_events"] <= 0:
        _fail("overwrite control block differs")
    if controls["lie_split_by_carrier"] != {"icosahedron": "1+3+8", "tetrahedron": "1+3", "octahedron": "none"}:
        _fail("Lie split table differs")
    fed = receipt["federation"]
    if fed["status"] == "isolated_federation_recovered":
        fed_loop = loops["federation_isolated_20"]
        fed_inv = fed_loop["stages"][0]["invariants"]
        strip = ("components", "components_isomorphic", "global_schedule_law")
        same = {k: v for k, v in fed_inv.items() if k not in strip} == {k: v for k, v in first.items() if k not in strip}
        if fed["component_invariants_equal_single_carrier"] != same or fed["components"] != fed_inv["components"] or fed["components"] != 20:
            _fail("federation block differs from its stage vectors")
        if not fed_inv["components_isomorphic"] or fed["fixed_point"] != fed_loop["fixed_point"]:
            _fail("federation isomorphism or fixed point claim differs")
        if receipt["firewall"]["federation_log_stored"] is not False:
            _fail("federation log storage flag differs")
    return {"recomputed_logs": recomputed, "loops": sorted(loops), "status": receipt["status"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--no-pins", action="store_true", help="skip the file pins (development)")
    args = parser.parse_args(argv)
    try:
        summary = verify_receipt(_load_json(args.receipt), ROOT, args.log_dir, check_pins=not args.no_pins)
    except IndependentVerificationError as exc:
        print(f"CLOSURE_LOOP_VERIFICATION_FAILED: {exc}")
        return 1
    print("CLOSURE_LOOP_VERIFIED " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
