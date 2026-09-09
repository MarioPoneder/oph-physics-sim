"""Independent verifier for the carrier-class selection receipt.

The verifier imports neither the producer (``oph_exact.carrier_class``) nor
``oph_exact.carrier``.  It builds the tetrahedron, the octahedron and the
icosahedron from its own face lists in its own vertex labelling, enumerates
the rotation groups by flag propagation (a rotation is fixed by the image of
one oriented face and spreads across the surface face by face), recomputes
conjugacy classes, power maps and the permutation character, verifies the
declared standard real character tables of ``A4``, ``S4`` and ``A5`` against
the recomputed groups (orthogonality, completeness, Frobenius-Schur
indicators, integrality of the ring operations), recomputes the reductive
candidates, the adjoint homomorphisms, the assignment tests and the
equivariant pairings, and compares every invariant with the receipt exactly.
The receipt's own face lists are checked as well: their rotation groups are
rebuilt and the labelled objects of the receipt (class representatives,
pairings, centralizers, reflections) are verified against them.

Usage::

    python -m oph_exact.verify_carrier_class_independent [--receipt PATH]

Exit code 0 when every check passes; the failures are printed otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from itertools import combinations_with_replacement, product
from pathlib import Path
from typing import Any, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = ROOT / "data" / "exact" / "carrier_class_selection_receipt.json"
SCHEMA = "oph.exact.carrier-class-selection.v1"
PIN_FILES = {
    "producer": "oph_exact/carrier_class.py",
    "verifier": "oph_exact/verify_carrier_class_independent.py",
    "test": "tests/test_exact_carrier_class.py",
    "carrier_module": "oph_exact/carrier.py",
}
EXPECTED_PAPER_REFERENCE = {
    "repository": "oph-meta",
    "path": "trt-scspl/instantiating_the_self_configuring_self_processing_language_that_is_our_universe.tex",
    "title": (
        "Instantiating the Self-Configuring Self-Processing Language That Is Our Universe: "
        "Observer Patch Holography as a Consistent Realization of the CTMU"
    ),
    "euler_proposition": "Uniform triangulations of the sphere",
    "carrier_proposition": "Carrier selection among uniform spherical triangulations",
    "forcing_theorem_label": "thm:forcing",
}

Perm = tuple[int, ...]
Face = tuple[int, int, int]
Q5 = tuple[Fraction, Fraction]
CF = tuple[Q5, ...]


# ==========================================================================
# Q(sqrt 5)
# ==========================================================================


def q(a: int | Fraction, b: int | Fraction = 0) -> Q5:
    return (Fraction(a), Fraction(b))


def q_add(x: Q5, y: Q5) -> Q5:
    return (x[0] + y[0], x[1] + y[1])


def q_sub(x: Q5, y: Q5) -> Q5:
    return (x[0] - y[0], x[1] - y[1])


def q_mul(x: Q5, y: Q5) -> Q5:
    return (x[0] * y[0] + 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def q_json(x: Q5) -> list[str]:
    return [str(x[0]), str(x[1])]


def parse_q5(value: Any) -> Q5:
    if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, str) for v in value)):
        raise VerifierError(f"Q(sqrt5) value encoding: {value!r}")
    return (Fraction(value[0]), Fraction(value[1]))


ZERO = q(0)
ONE = q(1)
PHI = q(Fraction(1, 2), Fraction(1, 2))
PSI = q(Fraction(1, 2), Fraction(-1, 2))  # 1 - phi


class VerifierError(RuntimeError):
    """Raised for a malformed receipt."""


# ==========================================================================
# Own carriers, own labellings
# ==========================================================================

TETRAHEDRON = ((2, 3, 0), (2, 0, 1), (2, 1, 3), (3, 1, 0))
# 0 = +x, 1 = +y, 2 = +z, 3 = -x, 4 = -y, 5 = -z
OCTAHEDRON = ((0, 1, 2), (1, 3, 2), (3, 4, 2), (4, 0, 2), (1, 0, 5), (3, 1, 5), (4, 3, 5), (0, 4, 5))


def icosahedron_faces() -> tuple[Face, ...]:
    """Top vertex 0, upper ring 1..5, lower ring 6..10, bottom vertex 11."""

    faces: list[Face] = []
    for i in range(5):
        u, u_next = 1 + i, 1 + (i + 1) % 5
        low, low_prev = 6 + i, 6 + (i - 1) % 5
        faces.append((0, u, u_next))
        faces.append((u_next, u, low))
        faces.append((low, u, low_prev))
        faces.append((11, low, low_prev))
    return tuple(faces)


OWN_FACES: dict[str, tuple[Face, ...]] = {
    "tetrahedron": TETRAHEDRON,
    "octahedron": OCTAHEDRON,
    "icosahedron": icosahedron_faces(),
}
GROUP_NAMES = {"tetrahedron": "A4", "octahedron": "S4", "icosahedron": "A5"}


def complex_checks(faces: Sequence[Face]) -> tuple[int, list[tuple[int, int]], list[list[int]], dict[str, bool]]:
    vertices = sorted({v for f in faces for v in f})
    n = len(vertices)
    directed = [(f[i], f[(i + 1) % 3]) for f in faces for i in range(3)]
    directed_set = set(directed)
    edges = sorted({(min(a, b), max(a, b)) for a, b in directed})
    adjacency = [[0] * n for _ in range(n)]
    for a, b in edges:
        adjacency[a][b] = adjacency[b][a] = 1
    degrees = [sum(row) for row in adjacency]
    reach = {0}
    stack = [0]
    while stack:
        v = stack.pop()
        for w in range(n):
            if adjacency[v][w] and w not in reach:
                reach.add(w)
                stack.append(w)
    checks = {
        "vertices_are_range": vertices == list(range(n)),
        "consistent_orientation": len(directed_set) == len(directed) and all((b, a) in directed_set for a, b in directed_set),
        "simplicial": all(len(set(f)) == 3 for f in faces) and len({frozenset(f) for f in faces}) == len(faces),
        "uniform_degree": len(set(degrees)) == 1,
        "connected": len(reach) == n,
        "euler_characteristic_two": n - len(edges) + len(faces) == 2,
        "links_are_cycles": all(_link_cycle(faces, v) for v in range(n)),
    }
    return n, edges, adjacency, checks


def _link_cycle(faces: Sequence[Face], v: int) -> bool:
    succ: dict[int, int] = {}
    for f in faces:
        for i in range(3):
            if f[i] == v:
                a, b = f[(i + 1) % 3], f[(i + 2) % 3]
                if a in succ:
                    return False
                succ[a] = b
    if not succ or set(succ.values()) != set(succ):
        return False
    start = min(succ)
    cur, count = succ[start], 1
    while cur != start:
        cur = succ[cur]
        count += 1
        if count > len(succ):
            return False
    return count == len(succ)


# ==========================================================================
# Flag propagation
# ==========================================================================


def compose(a: Perm, b: Perm) -> Perm:
    return tuple(a[x] for x in b)


def inverse(a: Perm) -> Perm:
    inv = [0] * len(a)
    for i, x in enumerate(a):
        inv[x] = i
    return tuple(inv)


def perm_order(a: Perm) -> int:
    e = tuple(range(len(a)))
    p, k = a, 1
    while p != e:
        p = compose(a, p)
        k += 1
    return k


def perm_power(a: Perm, k: int) -> Perm:
    r = tuple(range(len(a)))
    for _ in range(k):
        r = compose(a, r)
    return r


def _cyc(face: Sequence[int]) -> Face:
    f = tuple(face)
    i = f.index(min(f))
    return f[i:] + f[:i]  # type: ignore[return-value]


def flag_automorphisms(faces: Sequence[Face], n: int, reverse: bool) -> list[Perm]:
    """Automorphisms fixed by the image of the base oriented face.

    ``reverse=False`` gives the orientation-preserving automorphisms (the
    oriented base face maps onto an oriented face), ``reverse=True`` the
    orientation-reversing ones (it maps onto a reversed oriented face)."""

    third: dict[tuple[int, int], int] = {}
    for f in faces:
        for i in range(3):
            third[(f[i], f[(i + 1) % 3])] = f[(i + 2) % 3]
    oriented = {_cyc(f) for f in faces}
    base = faces[0]
    targets = []
    for f in faces:
        for i in range(3):
            t = (f[i], f[(i + 1) % 3], f[(i + 2) % 3])
            targets.append((t[0], t[2], t[1]) if reverse else t)
    found: list[Perm] = []
    for target in targets:
        perm = [-1] * n
        for v, w in zip(base, target, strict=True):
            perm[v] = w
        ok = True
        seen = {frozenset(base)}
        queue = [base]
        while queue and ok:
            f = queue.pop()
            for i in range(3):
                a, b = f[i], f[(i + 1) % 3]
                if (b, a) not in third:
                    ok = False
                    break
                d = third[(b, a)]
                g = (b, a, d)
                if frozenset(g) in seen:
                    continue
                key = (perm[a], perm[b]) if reverse else (perm[b], perm[a])
                if key not in third:
                    ok = False
                    break
                image_d = third[key]
                if perm[d] == -1:
                    perm[d] = image_d
                elif perm[d] != image_d:
                    ok = False
                    break
                seen.add(frozenset(g))
                queue.append(g)
        if not ok or -1 in perm or len(set(perm)) != n:
            continue
        p = tuple(perm)
        if reverse:
            preserved = all(_cyc((p[c], p[b], p[a])) in oriented for a, b, c in faces)
        else:
            preserved = all(_cyc((p[a], p[b], p[c])) in oriented for a, b, c in faces)
        if preserved:
            found.append(p)
    return found


# ==========================================================================
# Classes, keys, class functions
# ==========================================================================


def _graph_distance(adjacency: Sequence[Sequence[int]]) -> list[list[int]]:
    n = len(adjacency)
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for v in frontier:
                for w in range(n):
                    if adjacency[v][w] and dist[s][w] == -1:
                        dist[s][w] = dist[s][v] + 1
                        nxt.append(w)
            frontier = nxt
    return dist


def link_step(perm: Perm, adjacency: Sequence[Sequence[int]]) -> int | None:
    if perm_order(perm) != 5:
        return None
    fixed = [i for i, x in enumerate(perm) if i == x]
    if not fixed:
        return None
    v = fixed[0]
    u = next(w for w in range(len(perm)) if adjacency[v][w])
    return 1 if adjacency[u][perm[u]] else 2


def class_key(order: int, fixed: int, step: int | None, n: int) -> str:
    if order == 1:
        return f"o1f{n}"
    if order == 5:
        return f"o5f{fixed}s{step}"
    return f"o{order}f{fixed}"


class Group:
    def __init__(self, name: str, faces: Sequence[Face]) -> None:
        self.name = name
        self.faces = tuple(faces)
        self.n, self.edges, self.adjacency, self.checks = complex_checks(self.faces)
        self.rotations = flag_automorphisms(self.faces, self.n, reverse=False)
        self.reflections = flag_automorphisms(self.faces, self.n, reverse=True)
        self.order = len(self.rotations)
        self.identity = tuple(range(self.n))
        members = set(self.rotations)
        self.axioms = {
            "closed": all(compose(a, b) in members for a in self.rotations for b in self.rotations),
            "identity": self.identity in members,
            "inverses": all(inverse(a) in members for a in self.rotations),
            "transitive": len({a[0] for a in self.rotations}) == self.n,
            "no_duplicates": len(members) == self.order,
        }
        self.classes = self._classes()
        self.keys = [c["key"] for c in self.classes]
        self.index_of = {g: i for i, c in enumerate(self.classes) for g in c["elements"]}
        self.square = tuple(self.index_of[perm_power(c["rep"], 2)] for c in self.classes)
        self.cube = tuple(self.index_of[perm_power(c["rep"], 3)] for c in self.classes)
        self.chi_p: CF = tuple(q(c["fixed"]) for c in self.classes)

    def _classes(self) -> list[dict[str, Any]]:
        remaining = set(self.rotations)
        raw = []
        while remaining:
            g = min(remaining)
            cls = {compose(compose(h, g), inverse(h)) for h in self.rotations}
            remaining -= cls
            raw.append(sorted(cls))
        decorated = []
        for elements in raw:
            rep = elements[0]
            order = perm_order(rep)
            fixed = sum(1 for i, x in enumerate(rep) if i == x)
            step = link_step(rep, self.adjacency)
            decorated.append((order, -fixed, step or 0, rep, elements, fixed, step))
        decorated.sort(key=lambda t: t[:4])
        counts: dict[str, int] = {}
        prelim = []
        for order, _, _, rep, elements, fixed, step in decorated:
            key = class_key(order, fixed, step, self.n)
            counts[key] = counts.get(key, 0) + 1
            prelim.append((key, order, fixed, step, rep, elements))
        suffix: dict[str, int] = {}
        classes = []
        for key, order, fixed, step, rep, elements in prelim:
            if counts[key] > 1:
                idx = suffix.get(key, 0)
                suffix[key] = idx + 1
                key = key + "abcdefgh"[idx]
            classes.append(
                {"key": key, "order": order, "fixed": fixed, "step": step, "rep": rep, "elements": elements, "size": len(elements)}
            )
        return classes

    # class functions
    def const(self, value: int) -> CF:
        return tuple(q(value) for _ in self.classes)

    def inner(self, f: CF, g: CF) -> Q5:
        total = ZERO
        for c, x, y in zip(self.classes, f, g, strict=True):
            total = q_add(total, q_mul(q(c["size"]), q_mul(x, y)))
        return q_mul(total, q(Fraction(1, self.order)))

    def pull(self, f: CF, k: int) -> CF:
        idx = self.square if k == 2 else self.cube
        return tuple(f[i] for i in idx)

    def fs(self, f: CF) -> Q5:
        return self.inner(self.pull(f, 2), self.const(1))

    def alt2(self, f: CF) -> CF:
        return tuple(q_mul(q_sub(q_mul(x, x), y), q(Fraction(1, 2))) for x, y in zip(f, self.pull(f, 2), strict=True))

    def sym2(self, f: CF) -> CF:
        return tuple(q_mul(q_add(q_mul(x, x), y), q(Fraction(1, 2))) for x, y in zip(f, self.pull(f, 2), strict=True))

    def alt3(self, f: CF) -> CF:
        out = []
        for x, y, z in zip(f, self.pull(f, 2), self.pull(f, 3), strict=True):
            value = q_sub(q_mul(q_mul(x, x), x), q_mul(q(3), q_mul(x, y)))
            value = q_add(value, q_mul(q(2), z))
            out.append(q_mul(value, q(Fraction(1, 6))))
        return tuple(out)

    def by_key(self, f: CF) -> dict[str, list[str]]:
        return {c["key"]: q_json(x) for c, x in zip(self.classes, f, strict=True)}


def cf_add(f: CF, g: CF) -> CF:
    return tuple(q_add(x, y) for x, y in zip(f, g, strict=True))


def cf_sub(f: CF, g: CF) -> CF:
    return tuple(q_sub(x, y) for x, y in zip(f, g, strict=True))


def cf_mul(f: CF, g: CF) -> CF:
    return tuple(q_mul(x, y) for x, y in zip(f, g, strict=True))


def cf_scale(f: CF, c: int) -> CF:
    return tuple(q_mul(x, q(c)) for x in f)


# ==========================================================================
# Declared standard real character tables, keyed by class key
# ==========================================================================

NORMS = {"real": 1, "complex": 2, "quaternionic": 4}

DECLARED_TABLES: dict[str, dict[str, Any]] = {
    "tetrahedron": {
        "keys": ["o1f4", "o2f0", "o3f1a", "o3f1b"],
        "irreps": [
            ("1", "real", [q(1), q(1), q(1), q(1)]),
            ("2c", "complex", [q(2), q(2), q(-1), q(-1)]),
            ("3", "real", [q(3), q(-1), q(0), q(0)]),
        ],
    },
    "octahedron": {
        "keys": ["o1f6", "o2f2", "o2f0", "o3f0", "o4f2"],
        "irreps": [
            ("1", "real", [q(1), q(1), q(1), q(1), q(1)]),
            ("sgn", "real", [q(1), q(1), q(-1), q(1), q(-1)]),
            ("2", "real", [q(2), q(2), q(0), q(-1), q(0)]),
            ("3", "real", [q(3), q(-1), q(-1), q(0), q(1)]),
            ("3'", "real", [q(3), q(-1), q(1), q(0), q(-1)]),
        ],
    },
    "icosahedron": {
        "keys": ["o1f12", "o2f0", "o3f0", "o5f2s1", "o5f2s2"],
        "irreps": [
            ("1", "real", [q(1), q(1), q(1), q(1), q(1)]),
            ("3", "real", [q(3), q(-1), q(0), PHI, PSI]),
            ("3'", "real", [q(3), q(-1), q(0), PSI, PHI]),
            ("4", "real", [q(4), q(0), q(1), q(-1), q(-1)]),
            ("5", "real", [q(5), q(1), q(-1), q(0), q(0)]),
        ],
    },
}

SIMPLE_TABLE = (("su(2)", 3), ("su(3)", 8), ("so(5)", 10))


class Irrep:
    def __init__(self, name: str, kind: str, values: CF) -> None:
        self.name = name
        self.kind = kind
        self.norm = NORMS[kind]
        self.values = values
        self.dim = int(values[0][0])


def declared_table(group: Group) -> list[Irrep]:
    spec = DECLARED_TABLES[group.name]
    if sorted(spec["keys"]) != sorted(group.keys):
        raise VerifierError(f"{group.name}: declared class keys {spec['keys']} versus recomputed {group.keys}")
    position = {key: i for i, key in enumerate(spec["keys"])}
    irreps = []
    for name, kind, values in spec["irreps"]:
        ordered = tuple(values[position[key]] for key in group.keys)
        irreps.append(Irrep(name, kind, ordered))
    return irreps


def multiplicities(group: Group, chi: CF, table: Sequence[Irrep]) -> dict[str, int]:
    out = {}
    for r in table:
        m = q_mul(group.inner(chi, r.values), q(Fraction(1, r.norm)))
        if m[1] != 0 or m[0].denominator != 1 or m[0] < 0:
            raise VerifierError(f"{group.name}: multiplicity of {r.name} is {m}")
        out[r.name] = int(m[0])
    rebuilt = group.const(0)
    for r in table:
        rebuilt = cf_add(rebuilt, cf_scale(r.values, out[r.name]))
    if rebuilt != chi:
        raise VerifierError(f"{group.name}: character is outside the span of the declared table")
    return {k: v for k, v in out.items() if v}


def verify_table(group: Group, table: Sequence[Irrep]) -> list[str]:
    failures = []
    for i, r in enumerate(table):
        for j, s in enumerate(table):
            expected = q(r.norm) if i == j else ZERO
            if group.inner(r.values, s.values) != expected:
                failures.append(f"{group.name}: <{r.name},{s.name}> is {group.inner(r.values, s.values)}")
    completeness = sum((Fraction(r.dim * r.dim, r.norm) for r in table), Fraction(0))
    if completeness != group.order:
        failures.append(f"{group.name}: completeness sum {completeness} versus order {group.order}")
    for r in table:
        expected_fs = {"real": q(1), "complex": q(0), "quaternionic": q(-2)}[r.kind]
        if group.fs(r.values) != expected_fs:
            failures.append(f"{group.name}: Frobenius-Schur indicator of {r.name}")
        if r.values[group.keys.index(f"o1f{group.n}")] != q(r.dim):
            failures.append(f"{group.name}: {r.name} dimension at the identity")
    try:
        for r in table:
            multiplicities(group, group.sym2(r.values), table)
            multiplicities(group, group.alt2(r.values), table)
            for s in table:
                multiplicities(group, cf_mul(r.values, s.values), table)
    except VerifierError as error:
        failures.append(f"{group.name}: ring operations leave the declared table: {error}")
    return failures


# ==========================================================================
# Candidates, homomorphisms, assignments
# ==========================================================================


def partitions(target: int, parts: Sequence[tuple[str, int]], start: int = 0) -> Iterator[tuple[str, ...]]:
    if target == 0:
        yield ()
        return
    for i in range(start, len(parts)):
        name, dim = parts[i]
        if dim <= target:
            for rest in partitions(target - dim, parts, i):
                yield (name,) + rest


def candidates_for(total: int) -> tuple[list[dict[str, Any]], list[int]]:
    found = []
    empty = []
    for centre in (0, 1):
        semisimple = total - centre
        parts = list(partitions(semisimple, sorted(SIMPLE_TABLE, key=lambda t: t[1])))
        if not parts:
            empty.append(semisimple)
        for ideals in parts:
            found.append({"label": "+".join(["u(1)"] * centre + list(ideals)), "centre_dim": centre, "ideals": list(ideals)})
    return found, empty


class Hom:
    def __init__(self, label: str, adjoint: CF, fixed: int) -> None:
        self.label = label
        self.adjoint = adjoint
        self.fixed = fixed


def so3_homs(group: Group, table: Sequence[Irrep]) -> tuple[list[Hom], list[tuple[str, str]]]:
    small = [r for r in table if r.dim <= 3]
    homs, rejected = [], []
    for k in (1, 2, 3):
        for combo in combinations_with_replacement(range(len(small)), k):
            if sum(small[i].dim for i in combo) != 3:
                continue
            chi = group.const(0)
            for i in combo:
                chi = cf_add(chi, small[i].values)
            label = "+".join(small[i].name for i in combo)
            det = group.alt3(chi)
            if det != group.const(1):
                det_name = next((r.name for r in table if r.values == det), "?")
                rejected.append((label, det_name))
                continue
            adjoint = group.alt2(chi)
            fixed = group.inner(adjoint, group.const(1))
            homs.append(Hom(label, adjoint, int(fixed[0])))
    return homs, rejected


def psu3_homs(group: Group, table: Sequence[Irrep]) -> list[Hom]:
    if any(r.kind != "real" for r in table):
        raise VerifierError(f"{group.name}: PSU(3) enumeration needs a real-type table")
    homs = []
    for k in (1, 2, 3):
        for combo in combinations_with_replacement(range(len(table)), k):
            if sum(table[i].dim for i in combo) != 3:
                continue
            chi = group.const(0)
            for i in combo:
                chi = cf_add(chi, table[i].values)
            adjoint = cf_sub(cf_mul(chi, chi), group.const(1))
            fixed = group.inner(adjoint, group.const(1))
            homs.append(Hom("+".join(table[i].name for i in combo), adjoint, int(fixed[0])))
    return homs


def evaluate(group: Group, candidate: dict[str, Any], homs: dict[str, list[Hom]]) -> dict[str, Any]:
    ideals = list(candidate["ideals"])
    centre = int(candidate["centre_dim"])
    types = sorted(set(ideals), key=ideals.index)
    per_type = [list(combinations_with_replacement(homs[t], ideals.count(t))) for t in types]
    assignments = []
    survivors = []
    fixed_values = set()
    for choice in product(*per_type):
        flat = [h for grp in choice for h in grp]
        rhs = group.const(centre)
        for h in flat:
            rhs = cf_add(rhs, h.adjoint)
        fixed_values.add(centre + sum(h.fixed for h in flat))
        failing = next((i for i in range(len(group.classes)) if rhs[i] != group.chi_p[i]), None)
        labels = tuple(h.label for h in flat)
        assignments.append((labels, failing is None, None if failing is None else group.keys[failing], rhs))
        if failing is None:
            survivors.append(list(labels))
    witness = None
    for i, key in enumerate(group.keys):
        values = {a[3][i] for a in assignments}
        if group.chi_p[i] not in values:
            witness = {"class": key, "permutation_character": q_json(group.chi_p[i]), "candidate_values": sorted(q_json(v) for v in values)}
            break
    return {
        "candidate": candidate["label"],
        "assignment_count": len(assignments),
        "assignments": {(a[0], a[1], a[2]) for a in assignments},
        "survivors": survivors,
        "witness": witness,
        "fixed_values": sorted(fixed_values),
    }


# ==========================================================================
# Pairings and automorphisms
# ==========================================================================


def perfect_matchings(n: int) -> Iterator[Perm]:
    perm = [-1] * n

    def rec() -> Iterator[Perm]:
        try:
            i = perm.index(-1)
        except ValueError:
            yield tuple(perm)
            return
        for j in range(i + 1, n):
            if perm[j] == -1:
                perm[i], perm[j] = j, i
                yield from rec()
                perm[i] = perm[j] = -1

    yield from rec()


def commutes(c: Perm, group: Sequence[Perm]) -> bool:
    return all(g[c[i]] == c[g[i]] for g in group for i in range(len(c)))


def centralizer(group: Group) -> list[Perm]:
    found = []
    for x in range(group.n):
        c = [-1] * group.n
        ok = True
        for g in group.rotations:
            v, w = g[0], g[x]
            if c[v] == -1:
                c[v] = w
            elif c[v] != w:
                ok = False
                break
        if ok and -1 not in c and len(set(c)) == group.n and commutes(tuple(c), group.rotations):
            found.append(tuple(c))
    return found


def pairing_data(group: Group) -> dict[str, Any]:
    total = 0
    commuting = []
    for m in perfect_matchings(group.n):
        total += 1
        if commutes(m, group.rotations):
            commuting.append(m)
    central = centralizer(group)
    dist = _graph_distance(group.adjacency)
    return {
        "perfect_matching_count": total,
        "commuting": commuting,
        "commuting_count": len(commuting),
        "centralizer_order": len(central),
        "centralizer": central,
        "distances": sorted({dist[i][c[i]] for c in commuting for i in range(group.n)}),
    }


def induced_maps(group: Group, table: Sequence[Irrep], phi: dict[Perm, Perm]) -> tuple[dict[str, str], dict[str, str]]:
    class_map = {c["key"]: group.keys[group.index_of[phi[c["rep"]]]] for c in group.classes}
    irrep_map = {}
    for r in table:
        pulled = tuple(r.values[group.keys.index(class_map[c["key"]])] for c in group.classes)
        match = [s.name for s in table if s.values == pulled]
        if len(match) != 1:
            raise VerifierError(f"{group.name}: automorphism does not permute the declared table")
        irrep_map[r.name] = match[0]
    return class_map, irrep_map


def is_automorphism(group: Group, phi: dict[Perm, Perm]) -> bool:
    members = set(group.rotations)
    return (
        set(phi) == members
        and set(phi.values()) == members
        and all(phi[compose(g, h)] == compose(phi[g], phi[h]) for g in group.rotations for h in group.rotations)
    )


def reflection_maps(group: Group, table: Sequence[Irrep]) -> tuple[dict[str, str], dict[str, str]]:
    r = group.reflections[0]
    r_inv = inverse(r)
    phi = {g: compose(compose(r, g), r_inv) for g in group.rotations}
    if not is_automorphism(group, phi):
        raise VerifierError(f"{group.name}: reflection conjugation")
    return induced_maps(group, table, phi)


def a5_outer_maps(group: Group, table: Sequence[Irrep]) -> tuple[dict[str, str], dict[str, str]]:
    e = group.identity
    involutions = [g for g in group.rotations if g != e and compose(g, g) == e]
    triples: list[frozenset[Perm]] = []
    for g in involutions:
        t = frozenset([g] + [h for h in involutions if h != g and compose(g, h) == compose(h, g)])
        if t not in triples:
            triples.append(t)
    if len(involutions) != 15 or len(triples) != 5 or any(len(t) != 3 for t in triples):
        raise VerifierError("A5: Klein four-subgroups")
    pi = {}
    for g in group.rotations:
        g_inv = inverse(g)
        pi[g] = tuple(triples.index(frozenset(compose(compose(g, x), g_inv) for x in t)) for t in triples)
    pi_inv = {v: g for g, v in pi.items()}
    if len(pi_inv) != group.order:
        raise VerifierError("A5: action on the five subgroups is unfaithful")
    tau = (1, 0, 2, 3, 4)
    phi = {g: pi_inv[tuple(tau[pi[g][tau[i]]] for i in range(5))] for g in group.rotations}
    if not is_automorphism(group, phi):
        raise VerifierError("A5: outer automorphism")
    return induced_maps(group, table, phi)


# ==========================================================================
# Receipt loading and comparison
# ==========================================================================


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise VerifierError(f"duplicate key {k}")
        out[k] = v
    return out


def load_receipt(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    receipt = json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=lambda c: (_ for _ in ()).throw(VerifierError(c)))
    if not isinstance(receipt, dict):
        raise VerifierError("receipt is not an object")
    return receipt


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def euler_rows() -> list[dict[str, Any]]:
    rows = []
    for d in range(1, 8):
        if d >= 6:
            rows.append({"degree": d, "status": "excluded_positivity"})
            continue
        f = Fraction(4 * d, 6 - d)
        e, v = 3 * f / 2, 3 * f / d
        integral = all(x.denominator == 1 for x in (f, e, v))
        status = "excluded_non_integral" if not integral else ("excluded_non_simplicial" if d == 2 else "solution")
        rows.append({"degree": d, "F": str(f), "E": str(e), "V": str(v), "status": status})
    return rows


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)


def verify_carrier(name: str, entry: dict[str, Any], report: Report) -> None:
    group = Group(name, OWN_FACES[name])
    report.check(all(group.checks.values()), f"{name}: own complex checks {group.checks}")
    report.check(all(group.axioms.values()), f"{name}: own group axioms {group.axioms}")
    report.check(entry.get("rotation_group_order") == group.order, f"{name}: rotation group order")
    report.check(
        entry.get("incidence_automorphism_order") == group.order + len(group.reflections),
        f"{name}: incidence automorphism order",
    )
    report.check(len(group.reflections) == group.order, f"{name}: reflections match rotations in number")
    report.check(entry.get("rotation_group_name") == GROUP_NAMES[name], f"{name}: group name")
    counts = entry.get("counts", {})
    d = group.n and sum(group.adjacency[0])
    report.check(
        counts.get("degree") == d
        and counts.get("V") == group.n
        and counts.get("E") == len(group.edges)
        and counts.get("F") == len(group.faces)
        and Fraction(4 * d, 6 - d) == len(group.faces),
        f"{name}: Euler counts",
    )

    # classes
    receipt_classes = {c["key"]: c for c in entry.get("classes", [])}
    report.check(sorted(receipt_classes) == sorted(group.keys), f"{name}: class keys {sorted(receipt_classes)} versus {sorted(group.keys)}")
    for i, c in enumerate(group.classes):
        rc = receipt_classes.get(c["key"], {})
        report.check(
            rc.get("size") == c["size"]
            and rc.get("element_order") == c["order"]
            and rc.get("fixed_vertices") == c["fixed"]
            and rc.get("link_step") == c["step"]
            and rc.get("square_class") == group.keys[group.square[i]]
            and rc.get("cube_class") == group.keys[group.cube[i]],
            f"{name}: class data for {c['key']}",
        )
    report.check(entry.get("permutation_character") == group.by_key(group.chi_p), f"{name}: permutation character")
    report.check(entry.get("trivial_isotypic_dimension") == 1 and group.inner(group.chi_p, group.const(1)) == ONE, f"{name}: trivial isotypic dimension")

    # character table
    table = declared_table(group)
    report.failures.extend(verify_table(group, table))
    receipt_irreps = {r["name"]: r for r in entry.get("real_irreducible_characters", [])}
    report.check(sorted(receipt_irreps) == sorted(r.name for r in table), f"{name}: irreducible names")
    for r in table:
        rr = receipt_irreps.get(r.name, {})
        values = rr.get("values")
        keys = entry.get("character_table_checks", {}).get("class_keys", [])
        ok = isinstance(values, list) and len(values) == len(keys) == len(group.keys)
        if ok:
            by_key = {k: parse_q5(v) for k, v in zip(keys, values, strict=True)}
            ok = all(by_key.get(c["key"]) == x for c, x in zip(group.classes, r.values, strict=True))
        report.check(ok and rr.get("dim") == r.dim and rr.get("kind") == r.kind and rr.get("norm") == r.norm, f"{name}: irreducible {r.name}")
    report.check(entry.get("permutation_character_decomposition") == multiplicities(group, group.chi_p, table), f"{name}: chi_P decomposition")

    # candidates and homomorphisms
    cands, empty = candidates_for(group.n)
    report.check(entry.get("lie_candidates") == cands, f"{name}: Lie candidates {entry.get('lie_candidates')} versus {cands}")
    report.check(
        sorted(e["semisimple_dim"] for e in entry.get("semisimple_dimensions_without_candidate", [])) == sorted(empty),
        f"{name}: semisimple dimensions without candidate",
    )
    homs: dict[str, list[Hom]] = {}
    so3, rejected = so3_homs(group, table)
    homs["su(2)"] = so3
    adjoint_entry = entry.get("adjoint_homomorphisms", {})
    su2_entry = adjoint_entry.get("su(2)", {})
    report.check(
        [(h["label"], h["adjoint_character"], h["fixed_dimension"]) for h in su2_entry.get("homomorphisms", [])]
        == [(h.label, [q_json(x) for x in h.adjoint], h.fixed) for h in so3],
        f"{name}: SO(3) homomorphisms",
    )
    report.check(
        [(r["label"], r["determinant"]) for r in su2_entry.get("rejected_orthogonal_representations", [])] == rejected,
        f"{name}: rejected orthogonal representations",
    )
    needs_su3 = any("su(3)" in c["ideals"] for c in cands)
    su3_entry = adjoint_entry.get("su(3)", {})
    report.check(su3_entry.get("needed") == needs_su3, f"{name}: su(3) need flag")
    if needs_su3:
        psu3 = psu3_homs(group, table)
        homs["su(3)"] = psu3
        report.check(
            [(h["label"], h["adjoint_character"], h["fixed_dimension"]) for h in su3_entry.get("homomorphisms", [])]
            == [(h.label, [q_json(x) for x in h.adjoint], h.fixed) for h in psu3],
            f"{name}: PSU(3) homomorphisms",
        )
    report.check(not any("so(5)" in c["ideals"] for c in cands) and adjoint_entry.get("so(5)", {}).get("needed") is False, f"{name}: so(5) unused")

    # assignments
    evaluations = {e["candidate"]: e for e in entry.get("candidate_evaluations", [])}
    all_survivors = []
    for cand in cands:
        own = evaluate(group, cand, homs)
        rec = evaluations.get(cand["label"], {})
        rec_assignments = {(tuple(a["homomorphisms"]), a["survives"], a["failing_class"]) for a in rec.get("assignments", [])}
        report.check(rec.get("assignment_count") == own["assignment_count"], f"{name}: assignment count for {cand['label']}")
        report.check(rec_assignments == own["assignments"], f"{name}: assignments for {cand['label']}")
        report.check(rec.get("survivors") == own["survivors"], f"{name}: survivors for {cand['label']}")
        report.check(rec.get("uniform_witness_class") == own["witness"], f"{name}: uniform witness for {cand['label']}")
        report.check(rec.get("trivial_multiplicity_values") == own["fixed_values"], f"{name}: trivial multiplicities for {cand['label']}")
        all_survivors.extend({"candidate": cand["label"], "homomorphisms": s} for s in own["survivors"])
    report.check(entry.get("survivors") == all_survivors, f"{name}: survivor list")

    # pairings
    pairing = pairing_data(group)
    rp = entry.get("equivariant_inverse_pairing", {})
    report.check(rp.get("perfect_matching_count") == pairing["perfect_matching_count"], f"{name}: perfect matching count")
    report.check(rp.get("commuting_count") == pairing["commuting_count"], f"{name}: commuting involution count")
    report.check(rp.get("centralizer_order") == pairing["centralizer_order"], f"{name}: centralizer order")
    if pairing["commuting"]:
        report.check(rp.get("graph_distance_of_pairs") == pairing["distances"], f"{name}: pairing distances")

    # automorphisms
    class_map, irrep_map = reflection_maps(group, table)
    outer = entry.get("outer_automorphism", {})
    refl = outer.get("reflection_conjugation", {})
    report.check(refl.get("class_map") == class_map and refl.get("irreducible_map") == irrep_map, f"{name}: reflection conjugation maps")
    if name == "icosahedron":
        class_map, irrep_map = a5_outer_maps(group, table)
        abstract = outer.get("abstract_outer_automorphism", {})
        report.check(abstract.get("class_map") == class_map and abstract.get("irreducible_map") == irrep_map, "icosahedron: outer automorphism maps")
        report.check(irrep_map.get("3") == "3'" and irrep_map.get("3'") == "3", "icosahedron: outer automorphism swaps 3 and 3'")

    verify_receipt_labelling(name, entry, group, table, report)


def verify_receipt_labelling(name: str, entry: dict[str, Any], own: Group, table: Sequence[Irrep], report: Report) -> None:
    """Rebuild the receipt's own face lists and verify its labelled objects."""

    faces = entry.get("faces")
    if not (isinstance(faces, list) and all(isinstance(f, list) and len(f) == 3 for f in faces)):
        report.failures.append(f"{name}: receipt faces malformed")
        return
    labelled = Group(name, tuple(tuple(int(v) for v in f) for f in faces))
    report.check(all(labelled.checks.values()), f"{name}: receipt complex checks")
    if not all(labelled.checks.values()):
        return
    report.check(labelled.order == own.order, f"{name}: receipt-labelled rotation order")
    report.check(sorted(labelled.keys) == sorted(own.keys), f"{name}: receipt-labelled class keys")
    edges = {tuple(e) for e in entry.get("edges", [])}
    report.check(edges == set(labelled.edges), f"{name}: receipt edges")
    for rc in entry.get("classes", []):
        rep = tuple(rc.get("representative", []))
        idx = labelled.index_of.get(rep)
        ok = idx is not None and labelled.classes[idx]["key"] == rc.get("key") and labelled.classes[idx]["size"] == rc.get("size")
        report.check(ok, f"{name}: receipt representative of {rc.get('key')}")
    pairing = pairing_data(labelled)
    rp = entry.get("equivariant_inverse_pairing", {})
    report.check(
        sorted(tuple(c) for c in rp.get("commuting_fixed_point_free_involutions", [])) == sorted(pairing["commuting"]),
        f"{name}: receipt-labelled commuting involutions",
    )
    report.check(
        sorted(tuple(c) for c in rp.get("centralizer_elements", [])) == sorted(pairing["centralizer"]),
        f"{name}: receipt-labelled centralizer",
    )
    refl = tuple(entry.get("outer_automorphism", {}).get("reflection_conjugation", {}).get("reflection", []))
    report.check(refl in set(labelled.reflections), f"{name}: receipt reflection is an orientation-reversing automorphism")


def verify(path: Path = DEFAULT_RECEIPT) -> list[str]:
    """Return the list of failures (empty when the receipt verifies)."""

    report = Report()
    try:
        receipt = load_receipt(path)
    except (OSError, ValueError, VerifierError) as error:
        return [f"cannot load receipt: {error}"]
    report.check(receipt.get("schema") == SCHEMA, "schema")
    report.check(isinstance(receipt.get("status"), str) and bool(receipt.get("status")), "status")
    scope = receipt.get("scope", {})
    declared, not_claimed = scope.get("declared"), scope.get("not_claimed")
    report.check(
        isinstance(declared, list) and bool(declared) and isinstance(not_claimed, list) and bool(not_claimed),
        "scope block",
    )
    report.check(isinstance(receipt.get("claim_boundary"), str) and bool(receipt.get("claim_boundary")), "claim boundary")
    table = receipt.get("lie_classification_table", {})
    report.check(
        table.get("simple_compact_dimension_at_most_twelve") == {n: d for n, d in SIMPLE_TABLE} and table.get("declared") is True,
        "Lie classification table",
    )
    euler = receipt.get("euler_count", {}).get("table", [])
    own_rows = euler_rows()
    report.check(
        len(euler) == len(own_rows)
        and all(
            row.get("degree") == own["degree"] and row.get("status") == own["status"] and all(row.get(k) == own[k] for k in ("F", "E", "V") if k in own)
            for row, own in zip(euler, own_rows, strict=True)
        ),
        "Euler count table",
    )
    carriers = receipt.get("carriers", {})
    report.check(sorted(carriers) == sorted(OWN_FACES), "carrier set")
    for name in OWN_FACES:
        entry = carriers.get(name)
        if not isinstance(entry, dict):
            report.failures.append(f"{name}: missing")
            continue
        try:
            verify_carrier(name, entry, report)
        except (VerifierError, KeyError, TypeError, ValueError, IndexError) as error:
            report.failures.append(f"{name}: verification error {error!r}")
    summary = receipt.get("summary", {})
    report.check(summary.get("selected_carrier") == "icosahedron", "summary selects the icosahedron")
    report.check(summary.get("icosahedron_survivors") == carriers.get("icosahedron", {}).get("survivors"), "summary survivors")
    report.check(
        summary.get("octahedron_survivors") == []
        and summary.get("tetrahedron_survivors") == carriers.get("tetrahedron", {}).get("survivors"),
        "summary exclusions",
    )

    pins = receipt.get("implementation_pins", {})
    for key, rel in PIN_FILES.items():
        pin = pins.get(key, {})
        file_path = ROOT / rel
        report.check(pin.get("path") == rel, f"pin path for {key}")
        report.check(file_path.exists() and pin.get("sha256") == sha_file(file_path), f"pin sha256 for {key}")
    report.check(sorted(pins) == sorted(PIN_FILES), "pins are exactly the four repository files")
    reference = receipt.get("paper_reference", {})
    report.check(
        isinstance(reference, dict)
        and all(reference.get(k) == v for k, v in EXPECTED_PAPER_REFERENCE.items())
        and "sha256" not in reference,
        "paper reference block",
    )

    digest = receipt.get("receipt_sha256")
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    report.check(digest == "sha256:" + hashlib.sha256(canonical(body)).hexdigest(), "receipt self-digest")
    report.check(path.read_bytes() == canonical(receipt), "receipt is canonical JSON")
    return report.failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    failures = verify(args.receipt)
    if failures:
        for line in failures:
            print(f"FAIL {line}")
        return 1
    print(f"verified: {args.receipt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
