"""Carrier-class selection among the uniform spherical triangulations.

Machine check of the carrier-selection proposition of the instantiation paper
(``prop:carrier`` together with ``prop:euler`` and the dimension argument of
``thm:forcing``).  For each of the three uniform simplicial triangulations of
the sphere, the tetrahedron, the octahedron and the icosahedron, the module

* builds the oriented face complex and checks the Euler count
  ``F = 4d/(6-d)``, ``E = 3F/2``, ``V = 3F/d``;
* derives the incidence automorphisms and the orientation-preserving rotation
  group as vertex permutations, with conjugacy classes named geometrically;
* derives the real irreducible characters from the permutation module through
  exact spectral projectors of the adjacency matrix over ``Q(sqrt 5)`` and
  closes the character table under tensor, symmetric and exterior squares;
* enumerates the compact reductive Lie algebras ``z + s_1 + ... + s_k`` with
  ``dim g = |P|`` and ``dim z <= 1`` from the declared classification table,
  and the homomorphisms of the rotation group into the adjoint groups
  ``SO(3)`` and ``PSU(3)`` as classes of representations;
* tests the character identity ``chi_P = dim z + sum_i chi_{Ad o rho_i}``
  for every candidate and every assignment of homomorphisms;
* enumerates the fixed-point-free involutions of the port set that commute
  with the rotation group (equivariant inverse-port pairings).

Every decision is taken in exact arithmetic: integers, ``Fraction`` and
``Q(sqrt 5)`` pairs ``(a, b)`` meaning ``a + b sqrt 5``.  The icosahedron is
the committed carrier of :mod:`oph_exact.carrier`; the tetrahedron and the
octahedron are built here.

Receipt: ``data/exact/carrier_class_selection_receipt.json`` (canonical JSON,
``--write`` and ``--check``).  Independent verifier:
``oph_exact/verify_carrier_class_independent.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement, product
from pathlib import Path
from typing import Iterator, Sequence

from oph_exact import carrier
from oph_exact.carrier import GOLDEN, QS5, q5, q5_add, q5_inv, q5_mul, q5_sub

ROOT = Path(__file__).resolve().parents[1]
PRODUCER_PATH = Path(__file__).resolve()
VERIFIER_PATH = ROOT / "oph_exact" / "verify_carrier_class_independent.py"
TEST_PATH = ROOT / "tests" / "test_exact_carrier_class.py"
CARRIER_MODULE_PATH = ROOT / "oph_exact" / "carrier.py"
# The paper source lives in the oph-meta repository (uncommitted there) and is
# referenced without a hash.
PAPER_REFERENCE: dict[str, str] = {
    "repository": "oph-meta",
    "path": "trt-scspl/instantiating_the_self_configuring_self_processing_language_that_is_our_universe.tex",
    "title": (
        "Instantiating the Self-Configuring Self-Processing Language That Is Our Universe: "
        "Observer Patch Holography as a Consistent Realization of the CTMU"
    ),
    "euler_proposition": "Uniform triangulations of the sphere",
    "euler_label": "prop:euler",
    "carrier_proposition": "Carrier selection among uniform spherical triangulations",
    "carrier_label": "prop:carrier",
    "forcing_theorem_label": "thm:forcing",
    "hashing": "none; the source is outside this repository and changes independently of the receipt",
}
DEFAULT_RECEIPT = ROOT / "data" / "exact" / "carrier_class_selection_receipt.json"

SCHEMA = "oph.exact.carrier-class-selection.v1"
STATUS = (
    "EXACT_CARRIER_CLASS_SELECTION_CHECKED__"
    "OCTAHEDRON_EXCLUDED_BY_EQUIVARIANCE_TETRAHEDRON_EXCLUDED_BY_PAIRING_"
    "ICOSAHEDRON_CARRIES_U1_SU2_SU3__CARRIER_CLASS_TABLE_AND_IMPLEMENTERS_DECLARED"
)

Perm = tuple[int, ...]
Face = tuple[int, int, int]
ClassFunction = tuple[QS5, ...]

ZERO: QS5 = q5(0)
ONE: QS5 = q5(1)
HALF: QS5 = q5(Fraction(1, 2))
SIXTH: QS5 = q5(Fraction(1, 6))

# Declared classification table: compact simple Lie algebras of dimension at
# most twelve, by dimension.  The next simple algebra is g2 of dimension 14.
SIMPLE_COMPACT_LIE_ALGEBRAS: tuple[tuple[str, int], ...] = (
    ("su(2)", 3),
    ("su(3)", 8),
    ("so(5)", 10),
)
FIRST_EXCLUDED_SIMPLE: tuple[str, int] = ("g2", 14)
ADJOINT_GROUP = {"su(2)": "SO(3)", "su(3)": "PSU(3)", "so(5)": "SO(5)"}

CARRIER_NAMES: tuple[str, ...] = ("tetrahedron", "octahedron", "icosahedron")

TETRAHEDRON_FACES: tuple[Face, ...] = ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))
# Octahedron vertices: 0 = +x, 1 = -x, 2 = +y, 3 = -y, 4 = +z, 5 = -z; faces are
# the eight octants with outward orientation.
OCTAHEDRON_FACES: tuple[Face, ...] = (
    (0, 2, 4),
    (2, 1, 4),
    (1, 3, 4),
    (3, 0, 4),
    (2, 0, 5),
    (1, 2, 5),
    (3, 1, 5),
    (0, 3, 5),
)


class CarrierClassError(RuntimeError):
    """Raised when an exact check of the carrier-class selection fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierClassError(message)


# ==========================================================================
# Q(sqrt 5) helpers on top of oph_exact.carrier
# ==========================================================================


def q5_is_rational(x: QS5) -> bool:
    return x[1] == 0


def q5_is_integer(x: QS5) -> bool:
    return x[1] == 0 and x[0].denominator == 1


def q5_json(x: QS5) -> list[str]:
    return [str(x[0]), str(x[1])]


def cf_json(f: ClassFunction) -> list[list[str]]:
    return [q5_json(x) for x in f]


def cf_const(value: int | Fraction, count: int) -> ClassFunction:
    return tuple(q5(value) for _ in range(count))


def cf_add(f: ClassFunction, g: ClassFunction) -> ClassFunction:
    return tuple(q5_add(x, y) for x, y in zip(f, g, strict=True))


def cf_sub(f: ClassFunction, g: ClassFunction) -> ClassFunction:
    return tuple(q5_sub(x, y) for x, y in zip(f, g, strict=True))


def cf_mul(f: ClassFunction, g: ClassFunction) -> ClassFunction:
    return tuple(q5_mul(x, y) for x, y in zip(f, g, strict=True))


def cf_scale(f: ClassFunction, c: QS5) -> ClassFunction:
    return tuple(q5_mul(x, c) for x in f)


def cf_is_zero(f: ClassFunction) -> bool:
    return all(x[0] == 0 and x[1] == 0 for x in f)


# ==========================================================================
# Oriented face complexes and the Euler count
# ==========================================================================


def carrier_faces(name: str) -> tuple[Face, ...]:
    if name == "tetrahedron":
        return TETRAHEDRON_FACES
    if name == "octahedron":
        return OCTAHEDRON_FACES
    if name == "icosahedron":
        return carrier.oriented_faces()
    raise ValueError(f"unknown carrier {name!r}")


@dataclass(frozen=True)
class Complex:
    name: str
    faces: tuple[Face, ...]
    vertex_count: int
    edges: tuple[tuple[int, int], ...]
    degree: int
    adjacency: tuple[tuple[int, ...], ...]
    checks: dict[str, bool]

    @property
    def euler_characteristic(self) -> int:
        return self.vertex_count - len(self.edges) + len(self.faces)


def _link_is_single_cycle(faces: Sequence[Face], vertex: int) -> bool:
    """The oriented link of ``vertex`` is one directed cycle through all neighbours."""

    successor: dict[int, int] = {}
    indegree: dict[int, int] = {}
    for face in faces:
        for i in range(3):
            if face[i] == vertex:
                a, b = face[(i + 1) % 3], face[(i + 2) % 3]
                if a in successor:
                    return False
                successor[a] = b
                indegree[b] = indegree.get(b, 0) + 1
    if not successor or set(successor) != set(indegree) or any(v != 1 for v in indegree.values()):
        return False
    start = min(successor)
    seen = {start}
    current = successor[start]
    while current != start:
        if current in seen:
            return False
        seen.add(current)
        current = successor[current]
    return len(seen) == len(successor)


def build_complex(name: str) -> Complex:
    faces = tuple(tuple(int(v) for v in face) for face in carrier_faces(name))
    vertices = sorted({v for face in faces for v in face})
    n = len(vertices)
    _require(vertices == list(range(n)), f"{name}: vertices are 0..{n - 1}")
    directed = [(face[i], face[(i + 1) % 3]) for face in faces for i in range(3)]
    directed_set = set(directed)
    consistent_orientation = len(directed_set) == len(directed) and all(
        (b, a) in directed_set for (a, b) in directed_set
    )
    edges = tuple(sorted({(min(a, b), max(a, b)) for a, b in directed}))
    simplicial = all(len(set(face)) == 3 for face in faces) and len({frozenset(face) for face in faces}) == len(
        faces
    )
    each_edge_in_two_faces = all(
        sum(1 for face in faces if a in face and b in face) == 2 for a, b in edges
    )
    adjacency = [[0] * n for _ in range(n)]
    for a, b in edges:
        adjacency[a][b] = 1
        adjacency[b][a] = 1
    degrees = [sum(row) for row in adjacency]
    uniform = len(set(degrees)) == 1
    links = all(_link_is_single_cycle(faces, v) for v in range(n))
    reach = {0}
    frontier = [0]
    while frontier:
        v = frontier.pop()
        for w in range(n):
            if adjacency[v][w] and w not in reach:
                reach.add(w)
                frontier.append(w)
    connected = len(reach) == n
    euler = n - len(edges) + len(faces)
    checks = {
        "consistent_orientation": bool(consistent_orientation),
        "simplicial": bool(simplicial),
        "each_edge_in_two_faces": bool(each_edge_in_two_faces),
        "uniform_degree": bool(uniform),
        "vertex_links_single_cycles": bool(links),
        "connected": bool(connected),
        "euler_characteristic_two": euler == 2,
    }
    _require(all(checks.values()), f"{name}: complex checks {checks}")
    return Complex(
        name=name,
        faces=faces,
        vertex_count=n,
        edges=edges,
        degree=degrees[0],
        adjacency=tuple(tuple(row) for row in adjacency),
        checks=checks,
    )


def euler_count_table() -> list[dict[str, object]]:
    """``F = 4d/(6-d)``, ``E = 3F/2``, ``V = 3F/d`` for ``d = 1..7``."""

    rows: list[dict[str, object]] = []
    for d in range(1, 8):
        row: dict[str, object] = {"degree": d}
        if d >= 6:
            row["status"] = "excluded_positivity"
            row["reason"] = "F (6 - d) = 4 d forces 6 - d > 0"
            rows.append(row)
            continue
        f = Fraction(4 * d, 6 - d)
        e = 3 * f / 2
        v = 3 * f / d
        row.update({"F": str(f), "E": str(e), "V": str(v)})
        integral = f.denominator == 1 and e.denominator == 1 and v.denominator == 1
        if not integral:
            row["status"] = "excluded_non_integral"
            row["reason"] = "face count is not an integer"
        elif d == 2:
            row["status"] = "excluded_non_simplicial"
            row["reason"] = "two faces on the same three vertices (dihedron)"
        else:
            row["status"] = "solution"
            row["carrier"] = {3: "tetrahedron", 4: "octahedron", 5: "icosahedron"}[d]
        rows.append(row)
    return rows


def euler_count_matches(cx: Complex) -> dict[str, object]:
    d = cx.degree
    f_formula = Fraction(4 * d, 6 - d)
    e_formula = 3 * f_formula / 2
    v_formula = 3 * f_formula / d
    matches = (
        Fraction(len(cx.faces)) == f_formula
        and Fraction(len(cx.edges)) == e_formula
        and Fraction(cx.vertex_count) == v_formula
    )
    _require(matches, f"{cx.name}: Euler count formula")
    return {
        "degree": d,
        "V": cx.vertex_count,
        "E": len(cx.edges),
        "F": len(cx.faces),
        "euler_characteristic": cx.euler_characteristic,
        "formula_matches": True,
    }


# ==========================================================================
# Permutations and groups
# ==========================================================================


def compose(a: Perm, b: Perm) -> Perm:
    """``(a o b)(i) = a[b[i]]``."""

    return tuple(a[x] for x in b)


def inverse(a: Perm) -> Perm:
    inv = [0] * len(a)
    for i, x in enumerate(a):
        inv[x] = i
    return tuple(inv)


def identity(n: int) -> Perm:
    return tuple(range(n))


def perm_order(a: Perm) -> int:
    e = identity(len(a))
    power = a
    k = 1
    while power != e:
        power = compose(a, power)
        k += 1
    return k


def perm_power(a: Perm, k: int) -> Perm:
    result = identity(len(a))
    for _ in range(k):
        result = compose(a, result)
    return result


def fixed_points(a: Perm) -> int:
    return sum(1 for i, x in enumerate(a) if i == x)


def incidence_automorphisms(n: int, adjacency: Sequence[Sequence[int]]) -> tuple[Perm, ...]:
    """All vertex permutations preserving the edge set (backtracking)."""

    found: list[Perm] = []
    assignment: list[int] = []
    used = [False] * n

    def extend() -> None:
        p = len(assignment)
        if p == n:
            found.append(tuple(assignment))
            return
        for image in range(n):
            if used[image]:
                continue
            if all(adjacency[p][q] == adjacency[image][assignment[q]] for q in range(p)):
                assignment.append(image)
                used[image] = True
                extend()
                assignment.pop()
                used[image] = False

    extend()
    return tuple(found)


def _cyclic(face: Sequence[int]) -> Face:
    face_t = tuple(face)
    i = face_t.index(min(face_t))
    return face_t[i:] + face_t[:i]  # type: ignore[return-value]


def orientation_preserving(faces: Sequence[Face], automorphisms: Sequence[Perm]) -> tuple[Perm, ...]:
    oriented = {_cyclic(face) for face in faces}
    return tuple(
        perm
        for perm in automorphisms
        if all(_cyclic((perm[a], perm[b], perm[c])) in oriented for a, b, c in faces)
    )


def group_axiom_checks(group: Sequence[Perm]) -> dict[str, bool]:
    members = set(group)
    n = len(group[0])
    closed = all(compose(a, b) in members for a in group for b in group)
    has_identity = identity(n) in members
    inverses = all(inverse(a) in members for a in group)
    transitive = len({a[0] for a in group}) == n
    return {
        "closed_under_composition": bool(closed),
        "contains_identity": bool(has_identity),
        "closed_under_inverse": bool(inverses),
        "associative_as_permutations": True,
        "transitive_on_vertices": bool(transitive),
    }


# ==========================================================================
# Conjugacy classes with geometric names
# ==========================================================================


@dataclass(frozen=True)
class ConjugacyClass:
    key: str
    name: str
    size: int
    element_order: int
    fixed_vertices: int
    link_step: int | None
    representative: Perm
    elements: tuple[Perm, ...]
    axis: dict[str, object]


def _link_step(perm: Perm, adjacency: Sequence[Sequence[int]]) -> int | None:
    """For a rotation fixing a vertex ``v`` of degree five: whether a neighbour
    ``u`` of ``v`` is carried to a neighbour adjacent to ``u`` (step 1, angle
    ``2 pi / 5``) or to a non-adjacent neighbour (step 2, angle ``4 pi / 5``)."""

    fixed = [i for i, x in enumerate(perm) if i == x]
    if not fixed or perm_order(perm) != 5:
        return None
    v = fixed[0]
    neighbours = [w for w in range(len(perm)) if adjacency[v][w]]
    u = neighbours[0]
    w = perm[u]
    _require(w in neighbours and w != u, "order-five rotation moves the link")
    return 1 if adjacency[u][w] else 2


def _axis_witness(perm: Perm, cx: Complex, order: int, fixed: int) -> dict[str, object]:
    n = cx.vertex_count
    fixed_list = [i for i in range(n) if perm[i] == i]
    if order == 1:
        return {"type": "identity"}
    if fixed == 0 and order == 2:
        swapped = [list(e) for e in cx.edges if perm[e[0]] == e[1] and perm[e[1]] == e[0]]
        _require(len(swapped) >= 1, "edge half-turn swaps an edge")
        return {"type": "edge_midpoint_axis", "swapped_edges": swapped}
    if fixed == 0 and order == 3:
        invariant = [list(face) for face in cx.faces if {perm[v] for v in face} == set(face)]
        _require(len(invariant) == 2, "face third-turn has two invariant faces")
        return {"type": "face_axis", "invariant_faces": invariant}
    if fixed == 1 and order == 3:
        v = fixed_list[0]
        invariant = [list(face) for face in cx.faces if v not in face and {perm[w] for w in face} == set(face)]
        _require(len(invariant) == 1, "vertex-face third-turn has one invariant opposite face")
        return {"type": "vertex_face_axis", "vertex": v, "opposite_face": invariant[0]}
    if fixed == 2:
        a, b = fixed_list
        dist = _graph_distance(cx.adjacency)
        _require(dist[a][b] == max(max(row) for row in dist), "vertex-axis endpoints are at maximal distance")
        return {"type": "vertex_axis", "vertices": [a, b], "graph_distance": dist[a][b]}
    raise CarrierClassError(f"unnamed class: order {order}, fixed {fixed}")


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


def _class_key_and_name(order: int, fixed: int, link_step: int | None, n: int) -> tuple[str, str]:
    if order == 1:
        return f"o1f{n}", "identity"
    if order == 2 and fixed == 0:
        return "o2f0", "half turn about an edge-midpoint axis"
    if order == 2 and fixed == 2:
        return "o2f2", "half turn about a vertex axis"
    if order == 3 and fixed == 0:
        return "o3f0", "third turn about a face axis"
    if order == 3 and fixed == 1:
        return "o3f1", "third turn about a vertex-face axis"
    if order == 4 and fixed == 2:
        return "o4f2", "quarter turn about a vertex axis"
    if order == 5 and fixed == 2:
        _require(link_step in (1, 2), "order-five link step")
        return (
            f"o5f2s{link_step}",
            f"fifth turn about a vertex axis, neighbour step {link_step} (angle {2 * link_step}pi/5)",
        )
    raise CarrierClassError(f"unnamed class: order {order}, fixed {fixed}")


def conjugacy_classes(group: Sequence[Perm], cx: Complex) -> tuple[ConjugacyClass, ...]:
    n = cx.vertex_count
    remaining = set(group)
    raw: list[tuple[Perm, ...]] = []
    while remaining:
        g = min(remaining)
        cls = {compose(compose(h, g), inverse(h)) for h in group}
        _require(cls <= remaining, "conjugacy class closure")
        remaining -= cls
        raw.append(tuple(sorted(cls)))
    decorated = []
    for elements in raw:
        rep = elements[0]
        order = perm_order(rep)
        fixed = fixed_points(rep)
        _require(all(perm_order(x) == order and fixed_points(x) == fixed for x in elements), "class invariants")
        step = _link_step(rep, cx.adjacency)
        if step is not None:
            _require(all(_link_step(x, cx.adjacency) == step for x in elements), "class link step")
        decorated.append((order, -fixed, step or 0, rep, elements, fixed, step))
    decorated.sort(key=lambda t: t[:4])
    keys: dict[str, int] = {}
    prelim = []
    for order, _, _, rep, elements, fixed, step in decorated:
        key, name = _class_key_and_name(order, fixed, step, n)
        keys[key] = keys.get(key, 0) + 1
        prelim.append((key, name, order, fixed, step, rep, elements))
    suffix_counter: dict[str, int] = {}
    classes = []
    for key, name, order, fixed, step, rep, elements in prelim:
        if keys[key] > 1:
            index = suffix_counter.get(key, 0)
            suffix_counter[key] = index + 1
            letter = "abcdefgh"[index]
            key = key + letter
            name = name + f" (class {letter})"
        classes.append(
            ConjugacyClass(
                key=key,
                name=name,
                size=len(elements),
                element_order=order,
                fixed_vertices=fixed,
                link_step=step,
                representative=rep,
                elements=elements,
                axis=_axis_witness(rep, cx, order, fixed),
            )
        )
    _require(sum(c.size for c in classes) == len(group), "class sizes sum to the group order")
    return tuple(classes)


def class_index_of(classes: Sequence[ConjugacyClass], perm: Perm) -> int:
    for i, cls in enumerate(classes):
        if perm in cls.elements:
            return i
    raise CarrierClassError("permutation outside the group")


def power_maps(classes: Sequence[ConjugacyClass]) -> dict[int, tuple[int, ...]]:
    """Class index of ``g^k`` for ``k = 2, 3``, checked on every element."""

    out: dict[int, tuple[int, ...]] = {}
    for k in (2, 3):
        indices = []
        for cls in classes:
            target = class_index_of(classes, perm_power(cls.representative, k))
            _require(
                all(class_index_of(classes, perm_power(x, k)) == target for x in cls.elements),
                "power map is a class function",
            )
            indices.append(target)
        out[k] = tuple(indices)
    return out


# ==========================================================================
# Class functions and characters
# ==========================================================================


@dataclass(frozen=True)
class GroupData:
    cx: Complex
    automorphisms: tuple[Perm, ...]
    rotations: tuple[Perm, ...]
    classes: tuple[ConjugacyClass, ...]
    powers: dict[int, tuple[int, ...]]

    @property
    def order(self) -> int:
        return len(self.rotations)

    @property
    def class_count(self) -> int:
        return len(self.classes)

    def inner(self, f: ClassFunction, g: ClassFunction) -> QS5:
        total = ZERO
        for cls, x, y in zip(self.classes, f, g, strict=True):
            total = q5_add(total, q5_mul(q5(cls.size), q5_mul(x, y)))
        return q5_mul(total, q5(Fraction(1, self.order)))

    def pullback(self, f: ClassFunction, k: int) -> ClassFunction:
        """``g -> f(g^k)``."""

        return tuple(f[i] for i in self.powers[k])

    def frobenius_schur(self, f: ClassFunction) -> QS5:
        return self.inner(self.pullback(f, 2), cf_const(1, self.class_count))

    def exterior_square(self, f: ClassFunction) -> ClassFunction:
        return cf_scale(cf_sub(cf_mul(f, f), self.pullback(f, 2)), HALF)

    def symmetric_square(self, f: ClassFunction) -> ClassFunction:
        return cf_scale(cf_add(cf_mul(f, f), self.pullback(f, 2)), HALF)

    def exterior_cube(self, f: ClassFunction) -> ClassFunction:
        cube = cf_mul(cf_mul(f, f), f)
        mixed = cf_scale(cf_mul(f, self.pullback(f, 2)), q5(3))
        third = cf_scale(self.pullback(f, 3), q5(2))
        return cf_scale(cf_add(cf_sub(cube, mixed), third), SIXTH)

    def permutation_character(self) -> ClassFunction:
        return tuple(q5(cls.fixed_vertices) for cls in self.classes)

    def trivial(self) -> ClassFunction:
        return cf_const(1, self.class_count)

    def value_by_key(self, f: ClassFunction, key: str) -> QS5:
        for cls, x in zip(self.classes, f, strict=True):
            if cls.key == key:
                return x
        raise KeyError(key)


def build_group(cx: Complex) -> GroupData:
    autos = incidence_automorphisms(cx.vertex_count, cx.adjacency)
    rot = orientation_preserving(cx.faces, autos)
    _require(2 * len(rot) == len(autos), "rotations have index two in the incidence automorphisms")
    if cx.name == "icosahedron":
        _require(set(autos) == set(carrier.incidence_automorphisms()), "icosahedral automorphisms agree with carrier.py")
        _require(set(rot) == set(carrier.rotations()), "icosahedral rotations agree with carrier.py")
    checks = group_axiom_checks(rot)
    _require(all(checks.values()), f"{cx.name}: group axioms {checks}")
    classes = conjugacy_classes(rot, cx)
    return GroupData(cx=cx, automorphisms=autos, rotations=rot, classes=classes, powers=power_maps(classes))


# ==========================================================================
# Exact spectral decomposition of the adjacency matrix over Q(sqrt 5)
# ==========================================================================


def _int_matmul(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> list[list[int]]:
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def _solve_dependency(vectors: Sequence[Sequence[Fraction]], target: Sequence[Fraction]) -> list[Fraction] | None:
    """Coefficients ``c`` with ``sum c_i vectors_i = target``, or ``None``."""

    k = len(vectors)
    rows = [[vectors[j][i] for j in range(k)] + [Fraction(target[i])] for i in range(len(target))]
    pivot_row = 0
    pivots: list[int] = []
    for col in range(k):
        pivot = next((r for r in range(pivot_row, len(rows)) if rows[r][col] != 0), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][col]
        rows[pivot_row] = [x / scale for x in rows[pivot_row]]
        for r in range(len(rows)):
            if r != pivot_row and rows[r][col] != 0:
                factor = rows[r][col]
                rows[r] = [x - factor * y for x, y in zip(rows[r], rows[pivot_row], strict=True)]
        pivots.append(col)
        pivot_row += 1
    if any(all(x == 0 for x in row[:k]) and row[k] != 0 for row in rows):
        return None
    _require(len(pivots) == k, "Krylov vectors are independent")
    solution = [Fraction(0)] * k
    for r, col in enumerate(pivots):
        solution[col] = rows[r][k]
    return solution


def minimal_polynomial(matrix: Sequence[Sequence[int]]) -> tuple[Fraction, ...]:
    """Monic minimal polynomial over ``Q``, coefficients from degree zero up."""

    n = len(matrix)
    power = [[int(i == j) for j in range(n)] for i in range(n)]
    vectors = [[Fraction(x) for row in power for x in row]]
    while True:
        power = _int_matmul(power, matrix)
        target = [Fraction(x) for row in power for x in row]
        coefficients = _solve_dependency(vectors, target)
        if coefficients is not None:
            return tuple(-c for c in coefficients) + (Fraction(1),)
        vectors.append(target)


def _poly_eval(poly: Sequence[Fraction], x: Fraction) -> Fraction:
    total = Fraction(0)
    for c in reversed(poly):
        total = total * x + c
    return total


def _poly_deflate(poly: Sequence[Fraction], root: Fraction) -> list[Fraction]:
    """Divide by ``x - root`` (exact synthetic division)."""

    degree = len(poly) - 1
    quotient = [Fraction(0)] * degree
    quotient[degree - 1] = poly[degree]
    for i in range(degree - 1, 0, -1):
        quotient[i - 1] = poly[i] + root * quotient[i]
    remainder = poly[0] + root * quotient[0]
    _require(remainder == 0, "exact deflation")
    return quotient


def roots_in_q_sqrt5(poly: Sequence[Fraction]) -> tuple[QS5, ...]:
    """Distinct roots of a monic integer polynomial whose roots lie in ``Q(sqrt 5)``."""

    _require(all(c.denominator == 1 for c in poly) and poly[-1] == 1, "monic integer polynomial")
    current = list(poly)
    roots: list[QS5] = []
    progress = True
    while progress and len(current) > 1:
        progress = False
        constant = current[0]
        candidates = [Fraction(0)] if constant == 0 else []
        if constant != 0:
            bound = abs(int(constant))
            candidates = [Fraction(s * d) for d in range(1, bound + 1) if bound % d == 0 for s in (1, -1)]
        for cand in candidates:
            if _poly_eval(current, cand) == 0:
                roots.append(q5(cand))
                current = _poly_deflate(current, cand)
                progress = True
                break
    if len(current) == 3:
        b, c = current[1], current[0]
        disc = b * b - 4 * c
        _require(disc > 0 and disc.denominator == 1 and int(disc) % 5 == 0, "quadratic factor splits over Q(sqrt5)")
        square = int(disc) // 5
        s = math.isqrt(square)
        _require(s * s == square, "discriminant is five times a square")
        roots.append((-b / 2, Fraction(s, 2)))
        roots.append((-b / 2, Fraction(-s, 2)))
        current = [Fraction(1)]
    _require(len(current) == 1, "all roots lie in Q(sqrt5)")
    _require(len(set(roots)) == len(roots), "distinct roots")
    return tuple(roots)


def q5_matrix_identity(n: int) -> list[list[QS5]]:
    return [[q5(int(i == j)) for j in range(n)] for i in range(n)]


def q5_matrix_mul(a: Sequence[Sequence[QS5]], b: Sequence[Sequence[QS5]]) -> list[list[QS5]]:
    n = len(a)
    out = [[ZERO for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for k in range(n):
            x = a[i][k]
            if x[0] == 0 and x[1] == 0:
                continue
            for j in range(n):
                out[i][j] = q5_add(out[i][j], q5_mul(x, b[k][j]))
    return out


def spectral_projectors(adjacency: Sequence[Sequence[int]], eigenvalues: Sequence[QS5]) -> list[list[list[QS5]]]:
    n = len(adjacency)
    a = [[q5(adjacency[i][j]) for j in range(n)] for i in range(n)]
    projectors = []
    for k, lam_k in enumerate(eigenvalues):
        result = q5_matrix_identity(n)
        for m, lam_m in enumerate(eigenvalues):
            if m == k:
                continue
            factor = [[q5_sub(a[i][j], lam_m) if i == j else a[i][j] for j in range(n)] for i in range(n)]
            result = q5_matrix_mul(result, factor)
            scale = q5_inv(q5_sub(lam_k, lam_m))
            result = [[q5_mul(v, scale) for v in row] for row in result]
        projectors.append(result)
    total = q5_matrix_identity(n)
    for p in projectors:
        square = q5_matrix_mul(p, p)
        _require(square == p, "projector is idempotent")
    summed = [[ZERO for _ in range(n)] for _ in range(n)]
    for p in projectors:
        summed = [[q5_add(summed[i][j], p[i][j]) for j in range(n)] for i in range(n)]
    _require(summed == total, "projectors sum to the identity")
    for p, lam in zip(projectors, eigenvalues, strict=True):
        ap = q5_matrix_mul(a, p)
        _require(ap == [[q5_mul(lam, v) for v in row] for row in p], "projector is an eigenprojector")
    return projectors


def trace_with_permutation(p: Sequence[Sequence[QS5]], perm: Perm) -> QS5:
    """``tr(P M_g) = sum_i P[i][g(i)]``."""

    total = ZERO
    for i, x in enumerate(perm):
        total = q5_add(total, p[i][x])
    return total


# ==========================================================================
# Real irreducible characters: derivation and closure
# ==========================================================================


@dataclass(frozen=True)
class RealIrrep:
    name: str
    dim: int
    kind: str
    norm: int
    values: ClassFunction
    origin: str
    frobenius_schur: QS5
    in_permutation_module: bool


def _classify_remainder(gd: GroupData, r: ClassFunction) -> tuple[str, int] | None:
    n = gd.inner(r, r)
    nu = gd.frobenius_schur(r)
    if n == q5(1) and nu == q5(1):
        return "real", 1
    if n == q5(2) and nu == q5(0):
        return "complex", 2
    if n == q5(4) and nu == q5(-2):
        return "quaternionic", 4
    return None


def _multiplicities(gd: GroupData, chi: ClassFunction, table: Sequence[RealIrrep]) -> list[int]:
    out = []
    for r in table:
        m = q5_mul(gd.inner(chi, r.values), q5(Fraction(1, r.norm)))
        _require(q5_is_integer(m) and m[0] >= 0, f"non-negative integer multiplicity for {r.name}")
        out.append(int(m[0]))
    return out


def _subtract_known(gd: GroupData, chi: ClassFunction, table: Sequence[RealIrrep]) -> ClassFunction:
    remainder = chi
    for r, m in zip(table, _multiplicities(gd, chi, table), strict=True):
        remainder = cf_sub(remainder, cf_scale(r.values, q5(m)))
    return remainder


def _completeness_sum(table: Sequence[RealIrrep]) -> Fraction:
    return sum((Fraction(r.dim * r.dim, r.norm) for r in table), Fraction(0))


def derive_permutation_module_constituents(gd: GroupData) -> tuple[list[RealIrrep], dict[str, object]]:
    """Constituents of ``R^P`` as characters of the adjacency eigenspaces."""

    cx = gd.cx
    poly = minimal_polynomial(cx.adjacency)
    eigenvalues = roots_in_q_sqrt5(poly)
    projectors = spectral_projectors(cx.adjacency, eigenvalues)
    chi_p = gd.permutation_character()
    suborbits = gd.inner(chi_p, chi_p)
    _require(q5_is_integer(suborbits), "orbital count is an integer")
    _require(int(suborbits[0]) == len(eigenvalues), "distinct adjacency eigenvalues count the suborbits")
    constituents: list[RealIrrep] = []
    bands: list[dict[str, object]] = []
    for lam, p in zip(eigenvalues, projectors, strict=True):
        values = tuple(trace_with_permutation(p, cls.representative) for cls in gd.classes)
        dim = values[0]
        _require(q5_is_integer(dim) and dim[0] > 0, "band dimension is a positive integer")
        norm = gd.inner(values, values)
        _require(norm == q5(1), "adjacency band is irreducible")
        nu = gd.frobenius_schur(values)
        _require(nu == q5(1), "adjacency band is of real type")
        constituents.append(
            RealIrrep(
                name=f"band[{lam[0]},{lam[1]}]",
                dim=int(dim[0]),
                kind="real",
                norm=1,
                values=values,
                origin=f"adjacency eigenspace of eigenvalue {lam[0]} + {lam[1]} sqrt5",
                frobenius_schur=nu,
                in_permutation_module=True,
            )
        )
        bands.append({"eigenvalue": q5_json(lam), "dimension": int(dim[0])})
    total = constituents[0].values
    for r in constituents[1:]:
        total = cf_add(total, r.values)
    _require(total == chi_p, "band characters sum to the permutation character")
    report = {
        "minimal_polynomial_low_to_high": [str(c) for c in poly],
        "eigenvalues": bands,
        "suborbit_count": int(suborbits[0]),
    }
    return constituents, report


def close_character_table(gd: GroupData, seed: Sequence[RealIrrep]) -> list[RealIrrep]:
    table = list(seed)
    order = gd.order
    while _completeness_sum(table) != order:
        found = False
        candidates: list[tuple[str, ClassFunction]] = []
        for i, r in enumerate(table):
            candidates.append((f"Sym2({r.name})", gd.symmetric_square(r.values)))
            candidates.append((f"Alt2({r.name})", gd.exterior_square(r.values)))
            for j in range(i, len(table)):
                candidates.append((f"{r.name} x {table[j].name}", cf_mul(r.values, table[j].values)))
        for desc, chi in candidates:
            remainder = _subtract_known(gd, chi, table)
            if cf_is_zero(remainder):
                continue
            classified = _classify_remainder(gd, remainder)
            if classified is None:
                continue
            kind, norm = classified
            dim = remainder[0]
            _require(q5_is_integer(dim) and dim[0] > 0, "remainder dimension")
            table.append(
                RealIrrep(
                    name=f"rem{len(table)}",
                    dim=int(dim[0]),
                    kind=kind,
                    norm=norm,
                    values=remainder,
                    origin=f"{desc} with the known constituents removed",
                    frobenius_schur=gd.frobenius_schur(remainder),
                    in_permutation_module=False,
                )
            )
            found = True
            break
        _require(found, "character table closure makes progress")
    _require(_completeness_sum(table) == order, "sum of dim^2/norm equals the group order")
    for i, r in enumerate(table):
        for j, s in enumerate(table):
            expected = q5(r.norm) if i == j else ZERO
            _require(gd.inner(r.values, s.values) == expected, "orthogonality of real irreducible characters")
    return table


def name_irreps(gd: GroupData, table: Sequence[RealIrrep]) -> list[RealIrrep]:
    """Names by dimension; ``3`` is the vector representation (value ``phi`` on
    the ``2 pi / 5`` class for the icosahedron, the permutation-module copy for
    the octahedron); the nontrivial one-dimensional character is ``sgn``; the
    complex-type two-dimensional character is ``2c``."""

    named: list[RealIrrep] = []
    by_dim: dict[int, list[RealIrrep]] = {}
    for r in table:
        by_dim.setdefault(r.dim, []).append(r)
    step_one = next((c.key for c in gd.classes if c.key == "o5f2s1"), None)
    for dim, members in by_dim.items():
        if dim == 1:
            for r in members:
                is_trivial = r.values == gd.trivial()
                named.append(_rename(r, "1" if is_trivial else "sgn"))
            continue
        if len(members) == 1:
            r = members[0]
            named.append(_rename(r, "2c" if (dim == 2 and r.kind == "complex") else str(dim)))
            continue
        _require(dim == 3 and len(members) == 2, "only dimension three carries two irreducibles here")
        if step_one is not None:
            vector = [r for r in members if gd.value_by_key(r.values, step_one) == GOLDEN]
        else:
            vector = [r for r in members if r.in_permutation_module]
        _require(len(vector) == 1, "the vector representation is singled out")
        other = [r for r in members if r is not vector[0]]
        named.append(_rename(vector[0], "3"))
        named.append(_rename(other[0], "3'"))
    order = {"1": 0, "sgn": 1, "2": 2, "2c": 2, "3": 3, "3'": 4, "4": 5, "5": 6}
    named.sort(key=lambda r: (order.get(r.name, 99), r.name))
    _require(len({r.name for r in named}) == len(named), "irreducible names are distinct")
    mapping = {old.name: new.name for old, new in zip(table, _in_table_order(table, named), strict=True)}
    rewritten = []
    for r in named:
        origin = r.origin
        for old_name in sorted(mapping, key=len, reverse=True):
            origin = origin.replace(old_name, mapping[old_name])
        rewritten.append(RealIrrep(r.name, r.dim, r.kind, r.norm, r.values, origin, r.frobenius_schur, r.in_permutation_module))
    return rewritten


def _in_table_order(table: Sequence[RealIrrep], named: Sequence[RealIrrep]) -> list[RealIrrep]:
    by_values = {r.values: r for r in named}
    return [by_values[r.values] for r in table]


def _rename(r: RealIrrep, name: str) -> RealIrrep:
    return RealIrrep(
        name=name,
        dim=r.dim,
        kind=r.kind,
        norm=r.norm,
        values=r.values,
        origin=r.origin,
        frobenius_schur=r.frobenius_schur,
        in_permutation_module=r.in_permutation_module,
    )


def decompose(gd: GroupData, chi: ClassFunction, table: Sequence[RealIrrep]) -> dict[str, int]:
    mult = _multiplicities(gd, chi, table)
    rebuilt = cf_const(0, gd.class_count)
    for r, m in zip(table, mult, strict=True):
        rebuilt = cf_add(rebuilt, cf_scale(r.values, q5(m)))
    _require(rebuilt == chi, "decomposition rebuilds the character")
    return {r.name: m for r, m in zip(table, mult, strict=True) if m}


def _induced_maps(gd: GroupData, table: Sequence[RealIrrep], phi: dict[Perm, Perm]) -> tuple[dict[str, str], dict[str, str]]:
    """Maps induced on class keys and irreducible names by a group automorphism ``phi``."""

    class_map = {}
    for cls in gd.classes:
        class_map[cls.key] = gd.classes[class_index_of(gd.classes, phi[cls.representative])].key
    irrep_map = {}
    for r in table:
        pulled = tuple(gd.value_by_key(r.values, class_map[cls.key]) for cls in gd.classes)
        match = [s.name for s in table if s.values == pulled]
        _require(len(match) == 1, "an automorphism permutes the irreducibles")
        irrep_map[r.name] = match[0]
    return class_map, irrep_map


def _check_automorphism(gd: GroupData, phi: dict[Perm, Perm]) -> None:
    _require(set(phi) == set(gd.rotations) and set(phi.values()) == set(gd.rotations), "automorphism is a bijection")
    _require(all(phi[compose(g, h)] == compose(phi[g], phi[h]) for g in gd.rotations for h in gd.rotations), "automorphism is a homomorphism")


def reflection_conjugation(gd: GroupData, table: Sequence[RealIrrep]) -> dict[str, object]:
    """Conjugation by one incidence automorphism outside the rotation group."""

    reflection = next(a for a in gd.automorphisms if a not in set(gd.rotations))
    r_inv = inverse(reflection)
    phi = {g: compose(compose(reflection, g), r_inv) for g in gd.rotations}
    _check_automorphism(gd, phi)
    class_map, irrep_map = _induced_maps(gd, table, phi)
    inner = all(k == v for k, v in class_map.items())
    return {
        "reflection": list(reflection),
        "class_map": class_map,
        "irreducible_map": irrep_map,
        "acts_trivially_on_classes": inner,
    }


def a5_outer_automorphism(gd: GroupData, table: Sequence[RealIrrep]) -> dict[str, object]:
    """The outer automorphism of ``A5``: conjugation by a transposition of the
    five Klein four-subgroups (each the three mutually commuting involutions of
    one triple of perpendicular edge-midpoint axes)."""

    e = identity(gd.cx.vertex_count)
    involutions = [g for g in gd.rotations if g != e and compose(g, g) == e]
    _require(len(involutions) == 15, "fifteen involutions")
    triples: list[frozenset[Perm]] = []
    for g in involutions:
        triple = frozenset([g] + [h for h in involutions if h != g and compose(g, h) == compose(h, g)])
        _require(len(triple) == 3, "each involution commutes with two others")
        if triple not in triples:
            triples.append(triple)
    _require(len(triples) == 5, "five Klein four-subgroups")

    def image(g: Perm) -> tuple[int, ...]:
        g_inv = inverse(g)
        return tuple(triples.index(frozenset(compose(compose(g, x), g_inv) for x in t)) for t in triples)

    pi = {g: image(g) for g in gd.rotations}
    _require(len(set(pi.values())) == gd.order, "the action on the five subgroups is faithful")
    pi_inverse = {v: g for g, v in pi.items()}
    tau = (1, 0, 2, 3, 4)
    phi = {}
    for g in gd.rotations:
        p = pi[g]
        conjugated = tuple(tau[p[tau[i]]] for i in range(5))
        phi[g] = pi_inverse[conjugated]
    _check_automorphism(gd, phi)
    class_map, irrep_map = _induced_maps(gd, table, phi)
    return {
        "construction": "conjugation by a transposition of the five Klein four-subgroups (S5 outer action)",
        "class_map": class_map,
        "irreducible_map": irrep_map,
        "swaps_three_and_three_prime": irrep_map.get("3") == "3'" and irrep_map.get("3'") == "3",
    }


def outer_automorphism_action(gd: GroupData, table: Sequence[RealIrrep]) -> dict[str, object]:
    report: dict[str, object] = {"reflection_conjugation": reflection_conjugation(gd, table)}
    if gd.cx.name == "icosahedron":
        report["abstract_outer_automorphism"] = a5_outer_automorphism(gd, table)
    return report


# ==========================================================================
# Lie candidates and adjoint homomorphisms
# ==========================================================================


def _partitions(target: int, parts: Sequence[tuple[str, int]], start: int = 0) -> Iterator[tuple[str, ...]]:
    if target == 0:
        yield ()
        return
    for i in range(start, len(parts)):
        name, dim = parts[i]
        if dim <= target:
            for rest in _partitions(target - dim, parts, i):
                yield (name,) + rest


def reductive_candidates(total_dim: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compact reductive ``z + s_1 + ... + s_k`` with ``dim = total_dim`` and ``dim z <= 1``."""

    parts = tuple(sorted(SIMPLE_COMPACT_LIE_ALGEBRAS, key=lambda t: t[1]))
    candidates = []
    empty = []
    for centre in (0, 1):
        semisimple = total_dim - centre
        found = list(_partitions(semisimple, parts))
        if not found:
            empty.append({"centre_dim": centre, "semisimple_dim": semisimple, "reason": "no compact semisimple Lie algebra of this dimension"})
        for ideals in found:
            label = "+".join(["u(1)"] * centre + list(ideals))
            candidates.append({"label": label, "centre_dim": centre, "ideals": list(ideals)})
    return candidates, empty


@dataclass(frozen=True)
class AdjointHom:
    target: str
    label: str
    constituents: tuple[str, ...]
    representation_character: ClassFunction
    determinant: ClassFunction | None
    adjoint_character: ClassFunction
    adjoint_decomposition: dict[str, int]
    fixed_dimension: int


def so3_homomorphisms(gd: GroupData, table: Sequence[RealIrrep]) -> tuple[list[AdjointHom], list[dict[str, object]]]:
    """Three-dimensional real orthogonal representations with trivial determinant."""

    small = [r for r in table if r.dim <= 3]
    homs = []
    rejected = []
    for k in (1, 2, 3):
        for combo in combinations_with_replacement(range(len(small)), k):
            if sum(small[i].dim for i in combo) != 3:
                continue
            chi = cf_const(0, gd.class_count)
            for i in combo:
                chi = cf_add(chi, small[i].values)
            label = "+".join(small[i].name for i in combo)
            det = gd.exterior_cube(chi)
            det_name = next((r.name for r in table if r.values == det), None)
            _require(det_name is not None, "determinant character is a one-dimensional character")
            if det != gd.trivial():
                rejected.append({"label": label, "determinant": det_name})
                continue
            ad = gd.exterior_square(chi)
            _require(ad == chi, "adjoint character of SO(3) equals the vector character")
            homs.append(
                AdjointHom(
                    target="SO(3)",
                    label=label,
                    constituents=tuple(small[i].name for i in combo),
                    representation_character=chi,
                    determinant=det,
                    adjoint_character=ad,
                    adjoint_decomposition=decompose(gd, ad, table),
                    fixed_dimension=_trivial_multiplicity(gd, ad),
                )
            )
    return homs, rejected


def _trivial_multiplicity(gd: GroupData, chi: ClassFunction) -> int:
    m = gd.inner(chi, gd.trivial())
    _require(q5_is_integer(m), "trivial multiplicity")
    return int(m[0])


def psu3_homomorphisms(gd: GroupData, table: Sequence[RealIrrep]) -> list[AdjointHom]:
    """Three-dimensional complex representations; ``chi_Ad = |chi|^2 - 1``.

    Restricted to groups whose real irreducibles are all of real type, where
    the complex irreducibles coincide with the real table (the icosahedral
    ``A5``).  The lift of every homomorphism into ``PSU(3)`` to a linear
    representation is declared (``H^2(A5, Z/3) = 0``)."""

    _require(all(r.kind == "real" for r in table), "PSU(3) enumeration needs a real-type table")
    homs = []
    for k in (1, 2, 3):
        for combo in combinations_with_replacement(range(len(table)), k):
            if sum(table[i].dim for i in combo) != 3:
                continue
            chi = cf_const(0, gd.class_count)
            for i in combo:
                chi = cf_add(chi, table[i].values)
            ad = cf_sub(cf_mul(chi, chi), gd.trivial())
            homs.append(
                AdjointHom(
                    target="PSU(3)",
                    label="+".join(table[i].name for i in combo),
                    constituents=tuple(table[i].name for i in combo),
                    representation_character=chi,
                    determinant=None,
                    adjoint_character=ad,
                    adjoint_decomposition=decompose(gd, ad, table),
                    fixed_dimension=_trivial_multiplicity(gd, ad),
                )
            )
    return homs


def evaluate_candidate(
    gd: GroupData,
    candidate: dict[str, object],
    homs: dict[str, list[AdjointHom]],
) -> dict[str, object]:
    chi_p = gd.permutation_character()
    ideals = list(candidate["ideals"])  # type: ignore[arg-type]
    centre = int(candidate["centre_dim"])  # type: ignore[arg-type]
    types = sorted(set(ideals), key=ideals.index)
    per_type = []
    for t in types:
        count = ideals.count(t)
        per_type.append(list(combinations_with_replacement(homs[t], count)))
    assignments = []
    survivors = []
    fixed_dimensions = set()
    for choice in product(*per_type):
        flat = [h for group in choice for h in group]
        rhs = cf_const(centre, gd.class_count)
        for h in flat:
            rhs = cf_add(rhs, h.adjoint_character)
        fixed = centre + sum(h.fixed_dimension for h in flat)
        fixed_dimensions.add(fixed)
        failing = next((i for i in range(gd.class_count) if rhs[i] != chi_p[i]), None)
        record = {
            "homomorphisms": [h.label for h in flat],
            "by_ideal": [f"{ideal}:{h.label}" for ideal, h in zip(ideals, flat, strict=True)],
            "targets": [h.target for h in flat],
            "character": cf_json(rhs),
            "trivial_multiplicity": fixed,
            "survives": failing is None,
            "failing_class": None if failing is None else gd.classes[failing].key,
            "failing_values": None
            if failing is None
            else {"permutation_character": q5_json(chi_p[failing]), "candidate": q5_json(rhs[failing])},
        }
        assignments.append(record)
        if failing is None:
            survivors.append([h.label for h in flat])
    uniform_witness = None
    for i, cls in enumerate(gd.classes):
        values = {tuple(a["character"][i]) for a in assignments}
        if all(tuple(q5_json(chi_p[i])) != v for v in values):
            uniform_witness = {
                "class": cls.key,
                "permutation_character": q5_json(chi_p[i]),
                "candidate_values": sorted([list(v) for v in values], key=lambda v: (Fraction(v[0]), Fraction(v[1]))),
            }
            break
    return {
        "candidate": candidate["label"],
        "centre_dim": centre,
        "ideals": ideals,
        "assignment_count": len(assignments),
        "assignments": assignments,
        "survivors": survivors,
        "uniform_witness_class": uniform_witness,
        "trivial_multiplicity_values": sorted(fixed_dimensions),
        "trivial_multiplicity_required": 1,
    }


# ==========================================================================
# Equivariant inverse pairings
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
                perm[i] = j
                perm[j] = i
                yield from rec()
                perm[i] = -1
                perm[j] = -1

    yield from rec()


def commutes_with_all(c: Perm, group: Sequence[Perm]) -> bool:
    n = len(c)
    return all(g[c[i]] == c[g[i]] for g in group for i in range(n))


def centralizer_in_symmetric_group(group: Sequence[Perm], n: int) -> list[Perm]:
    """Centralizer of a transitive group: an element is fixed by its value at vertex 0."""

    found = []
    for x in range(n):
        c = [-1] * n
        consistent = True
        for g in group:
            v, w = g[0], g[x]
            if c[v] == -1:
                c[v] = w
            elif c[v] != w:
                consistent = False
                break
        if consistent and -1 not in c and len(set(c)) == n and commutes_with_all(tuple(c), group):
            found.append(tuple(c))
    return found


def inverse_pairing_report(gd: GroupData) -> dict[str, object]:
    n = gd.cx.vertex_count
    total = 0
    commuting = []
    for m in perfect_matchings(n):
        total += 1
        if commutes_with_all(m, gd.rotations):
            commuting.append(m)
    centralizer = centralizer_in_symmetric_group(gd.rotations, n)
    central_involutions = [c for c in centralizer if c != identity(n) and compose(c, c) == identity(n) and fixed_points(c) == 0]
    _require(sorted(central_involutions) == sorted(commuting), "matching enumeration agrees with the centralizer")
    report: dict[str, object] = {
        "perfect_matching_count": total,
        "commuting_fixed_point_free_involutions": [list(c) for c in commuting],
        "commuting_count": len(commuting),
        "centralizer_order": len(centralizer),
        "centralizer_elements": [list(c) for c in centralizer],
    }
    if gd.cx.name == "icosahedron":
        _require(len(commuting) == 1 and commuting[0] == tuple(carrier.antipode()), "icosahedral pairing is the carrier antipode")
    if commuting:
        dist = _graph_distance(gd.cx.adjacency)
        report["graph_distance_of_pairs"] = sorted({dist[i][c[i]] for c in commuting for i in range(n)})
    return report


# ==========================================================================
# Per-carrier analysis
# ==========================================================================


GROUP_NAMES = {"tetrahedron": "A4", "octahedron": "S4", "icosahedron": "A5"}
EXPECTED_ORDERS = {"tetrahedron": 12, "octahedron": 24, "icosahedron": 60}


@lru_cache(maxsize=None)
def analyze(name: str) -> dict[str, object]:
    cx = build_complex(name)
    gd = build_group(cx)
    _require(gd.order == EXPECTED_ORDERS[name], f"{name}: rotation group order {gd.order}")
    chi_p = gd.permutation_character()
    trivial_dim = _trivial_multiplicity(gd, chi_p)
    _require(trivial_dim == 1, "trivial isotypic component of R^P is one-dimensional")
    seed, spectral = derive_permutation_module_constituents(gd)
    table = name_irreps(gd, close_character_table(gd, seed))
    chi_p_decomposition = decompose(gd, chi_p, table)
    outer = outer_automorphism_action(gd, table)

    candidates, empty = reductive_candidates(cx.vertex_count)
    needed_types = {t for c in candidates for t in c["ideals"]}  # type: ignore[union-attr]
    homs: dict[str, list[AdjointHom]] = {}
    hom_reports: dict[str, object] = {}
    so3, rejected = so3_homomorphisms(gd, table)
    homs["su(2)"] = so3
    hom_reports["su(2)"] = {
        "adjoint_group": "SO(3)",
        "needed": "su(2)" in needed_types,
        "homomorphisms": [_hom_json(h) for h in so3],
        "rejected_orthogonal_representations": rejected,
    }
    if "su(3)" in needed_types:
        psu3 = psu3_homomorphisms(gd, table)
        homs["su(3)"] = psu3
        hom_reports["su(3)"] = {
            "adjoint_group": "PSU(3)",
            "needed": True,
            "homomorphisms": [_hom_json(h) for h in psu3],
            "lift_to_linear_representation": "declared: H^2(A5, Z/3) = 0, so every homomorphism into PSU(3) lifts to SU(3)",
        }
    else:
        hom_reports["su(3)"] = {
            "adjoint_group": "PSU(3)",
            "needed": False,
            "reason": f"no candidate of dimension {cx.vertex_count} with centre of dimension at most one has an su(3) ideal",
        }
    hom_reports["so(5)"] = {
        "adjoint_group": "SO(5)",
        "needed": "so(5)" in needed_types,
        "reason": f"no candidate of dimension {cx.vertex_count} with centre of dimension at most one has an so(5) ideal",
    }
    _require("so(5)" not in needed_types, "so(5) is never needed")

    evaluations = [evaluate_candidate(gd, c, homs) for c in candidates]
    survivors = [
        {"candidate": e["candidate"], "homomorphisms": s}
        for e in evaluations
        for s in e["survivors"]  # type: ignore[union-attr]
    ]
    pairing = inverse_pairing_report(gd)

    return {
        "complex": cx,
        "group": gd,
        "spectral": spectral,
        "table": table,
        "chi_p_decomposition": chi_p_decomposition,
        "outer": outer,
        "candidates": candidates,
        "empty_candidates": empty,
        "hom_reports": hom_reports,
        "evaluations": evaluations,
        "survivors": survivors,
        "pairing": pairing,
    }


def _hom_json(h: AdjointHom) -> dict[str, object]:
    return {
        "label": h.label,
        "target": h.target,
        "constituents": list(h.constituents),
        "representation_character": cf_json(h.representation_character),
        "determinant_trivial": None if h.determinant is None else h.determinant == tuple(ONE for _ in h.determinant),
        "adjoint_character": cf_json(h.adjoint_character),
        "adjoint_decomposition": h.adjoint_decomposition,
        "fixed_dimension": h.fixed_dimension,
    }


def _class_json(cls: ConjugacyClass, gd: GroupData, index: int) -> dict[str, object]:
    return {
        "key": cls.key,
        "name": cls.name,
        "size": cls.size,
        "element_order": cls.element_order,
        "fixed_vertices": cls.fixed_vertices,
        "link_step": cls.link_step,
        "representative": list(cls.representative),
        "axis": cls.axis,
        "square_class": gd.classes[gd.powers[2][index]].key,
        "cube_class": gd.classes[gd.powers[3][index]].key,
    }


def _irrep_json(r: RealIrrep) -> dict[str, object]:
    return {
        "name": r.name,
        "dim": r.dim,
        "kind": r.kind,
        "norm": r.norm,
        "values": cf_json(r.values),
        "frobenius_schur": q5_json(r.frobenius_schur),
        "origin": r.origin,
        "in_permutation_module": r.in_permutation_module,
    }


VERDICTS = {
    "tetrahedron": (
        "u(1)+su(2) with su(2) carrying the vector representation 3 is the sole "
        "assignment consistent with the character constraint; no fixed-point-free "
        "involution of the four ports commutes with A4, so no equivariant inverse "
        "pairing exists"
    ),
    "octahedron": (
        "su(2)+su(2) is the sole candidate and no assignment of homomorphisms "
        "S4 -> SO(3) satisfies the character constraint; witness: the edge "
        "half-turn class, where the permutation character vanishes and every "
        "candidate trace lies in {6, 2, -2}"
    ),
    "icosahedron": (
        "u(1)+su(2)+su(3) with su(2) on 3 and su(3) on 3'+5, together with its "
        "image under the outer automorphism (su(2) on 3', su(3) on 3+5), are the "
        "sole surviving assignments; su(2)^4 fails; the antipodal map is the "
        "unique fixed-point-free involution commuting with the sixty rotations"
    ),
}


def carrier_json(name: str) -> dict[str, object]:
    a = analyze(name)
    cx: Complex = a["complex"]  # type: ignore[assignment]
    gd: GroupData = a["group"]  # type: ignore[assignment]
    table: list[RealIrrep] = a["table"]  # type: ignore[assignment]
    chi_p = gd.permutation_character()
    return {
        "faces": [list(f) for f in cx.faces],
        "edges": [list(e) for e in cx.edges],
        "counts": euler_count_matches(cx),
        "complex_checks": cx.checks,
        "incidence_automorphism_order": len(gd.automorphisms),
        "rotation_group_order": gd.order,
        "rotation_group_name": GROUP_NAMES[name],
        "group_axioms": group_axiom_checks(gd.rotations),
        "classes": [_class_json(cls, gd, i) for i, cls in enumerate(gd.classes)],
        "permutation_character": {cls.key: q5_json(x) for cls, x in zip(gd.classes, chi_p, strict=True)},
        "trivial_isotypic_dimension": _trivial_multiplicity(gd, chi_p),
        "adjacency_spectrum": a["spectral"],
        "real_irreducible_characters": [_irrep_json(r) for r in table],
        "character_table_checks": {
            "orthogonality": True,
            "completeness_sum": str(_completeness_sum(table)),
            "group_order": gd.order,
            "class_keys": [cls.key for cls in gd.classes],
        },
        "permutation_character_decomposition": a["chi_p_decomposition"],
        "outer_automorphism": a["outer"],
        "lie_candidates": a["candidates"],
        "semisimple_dimensions_without_candidate": a["empty_candidates"],
        "adjoint_homomorphisms": a["hom_reports"],
        "candidate_evaluations": a["evaluations"],
        "survivors": a["survivors"],
        "equivariant_inverse_pairing": a["pairing"],
        "verdict": VERDICTS[name],
    }


# ==========================================================================
# Receipt
# ==========================================================================


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def implementation_pins() -> dict[str, dict[str, str]]:
    pins = {
        "producer": {"path": str(PRODUCER_PATH.relative_to(ROOT)), "sha256": _sha256_file(PRODUCER_PATH)},
        "verifier": {"path": str(VERIFIER_PATH.relative_to(ROOT)), "sha256": _sha256_file(VERIFIER_PATH)},
        "test": {"path": str(TEST_PATH.relative_to(ROOT)), "sha256": _sha256_file(TEST_PATH)},
        "carrier_module": {
            "path": str(CARRIER_MODULE_PATH.relative_to(ROOT)),
            "sha256": _sha256_file(CARRIER_MODULE_PATH),
        },
    }
    return pins


SCOPE = {
    "declared": [
        "carrier class: simplicial triangulations of the sphere of uniform vertex degree (the A1 carrier class); "
        "uniqueness of the d-regular triangulation for d = 3, 4, 5 is classical and declared",
        "classification table: the compact simple Lie algebras of dimension at most twelve are su(2) (3), su(3) (8) "
        "and so(5) = sp(2) (10); the next simple algebra is g2 (14)",
        "inner implementers: the transport clause of A2 acts on g = D(R^P) by inner automorphisms, so every simple "
        "ideal is preserved and carries a homomorphism into its adjoint group (SO(3) for su(2), PSU(3) for su(3)) "
        "and the centre is fixed pointwise",
        "homomorphism classes: conjugacy classes of homomorphisms G -> SO(3) are the three-dimensional real "
        "orthogonal representations with trivial determinant, with adjoint character the exterior square; every "
        "homomorphism A5 -> PSU(3) lifts to a linear three-dimensional representation (H^2(A5, Z/3) = 0), with "
        "adjoint character |chi|^2 - 1",
        "hypotheses of the proposition: a complete reversible response D of A1 (an isomorphism R^P -> g of "
        "modules) and the transport clause of A2 for every rotation",
    ],
    "not_claimed": [
        "derivation of the carrier class from records",
        "Lean check of the proposition (work in progress)",
        "the proof of the flagship forcing theorem; the receipt re-derives its character-constraint outcome on the icosahedron",
        "selection outside the uniform spherical triangulations",
    ],
}

CLAIM_BOUNDARY = (
    "Within the declared class of uniform simplicial triangulations of the sphere, with the declared Lie "
    "classification table and inner implementers, the character constraint of an A1-A2 equivariant response "
    "excludes the octahedron, the equivariant inverse pairing excludes the tetrahedron, and the icosahedron "
    "carries u(1)+su(2)+su(3) with su(2) on the vector representation up to the outer automorphism of A5. "
    "The carrier class, the classification table and the implementer clause are supplied; the receipt is a "
    "machine check of the paper's proposition and derives no carrier from records."
)


def build_receipt() -> dict[str, object]:
    """The receipt as a JSON-ready object."""

    carriers = {name: carrier_json(name) for name in CARRIER_NAMES}
    summary = {
        "selected_carrier": "icosahedron",
        "exclusions": {
            "octahedron": "A1-A2 equivariance: no reductive candidate satisfies the character constraint (edge half-turn trace)",
            "tetrahedron": "equivariant inverse pairing: the centralizer of A4 in Sym(4) is trivial",
        },
        "icosahedron_algebra": "u(1)+su(2)+su(3)",
        "icosahedron_survivors": carriers["icosahedron"]["survivors"],
        "tetrahedron_survivors": carriers["tetrahedron"]["survivors"],
        "octahedron_survivors": carriers["octahedron"]["survivors"],
        "declared_inputs": [
            "carrier class = uniform-degree simplicial triangulations of the sphere",
            "compact simple Lie algebra classification table",
            "inner implementers of the A2 transport clause",
        ],
    }
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "status": STATUS,
        "paper_reference": dict(PAPER_REFERENCE),
        "euler_count": {
            "formulas": {"F": "4d/(6-d)", "E": "3F/2", "V": "3F/d"},
            "table": euler_count_table(),
            "uniqueness": "declared: the d-regular simplicial triangulation of the sphere is unique up to isomorphism for d = 3, 4, 5",
        },
        "lie_classification_table": {
            "simple_compact_dimension_at_most_twelve": {name: dim for name, dim in SIMPLE_COMPACT_LIE_ALGEBRAS},
            "first_excluded": {FIRST_EXCLUDED_SIMPLE[0]: FIRST_EXCLUDED_SIMPLE[1]},
            "adjoint_groups": ADJOINT_GROUP,
            "declared": True,
        },
        "carriers": carriers,
        "summary": summary,
        "scope": SCOPE,
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation_pins": implementation_pins(),
    }
    receipt["receipt_sha256"] = "sha256:" + hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    return receipt


def write_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, object]:
    receipt = build_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(receipt))
    return receipt


def check_receipt(path: Path = DEFAULT_RECEIPT) -> list[str]:
    """Differences between the stored receipt and a fresh rebuild (empty when frozen)."""

    if not path.exists():
        return [f"missing receipt {path}"]
    stored = path.read_bytes()
    fresh = canonical_bytes(build_receipt())
    if stored == fresh:
        return []
    differences = ["canonical bytes differ"]
    try:
        stored_obj = json.loads(stored)
        fresh_obj = json.loads(fresh)
    except json.JSONDecodeError as error:
        return differences + [f"stored receipt is not JSON: {error}"]
    for key in sorted(set(stored_obj) | set(fresh_obj)):
        if stored_obj.get(key) != fresh_obj.get(key):
            differences.append(f"top-level key differs: {key}")
    return differences


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write the receipt")
    parser.add_argument("--check", action="store_true", help="rebuild and compare with the stored receipt")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    if args.write:
        receipt = write_receipt(args.receipt)
        print(f"wrote {args.receipt} ({receipt['receipt_sha256']})")
        return 0
    if args.check:
        differences = check_receipt(args.receipt)
        if differences:
            for line in differences:
                print(line)
            return 1
        print(f"receipt frozen: {args.receipt}")
        return 0
    receipt = build_receipt()
    for name in CARRIER_NAMES:
        entry = receipt["carriers"][name]  # type: ignore[index]
        print(f"{name}: |Rot| = {entry['rotation_group_order']}, survivors = {entry['survivors']}")
        print(f"  {entry['verdict']}")
    return 0


__all__ = [
    "SCHEMA",
    "STATUS",
    "DEFAULT_RECEIPT",
    "CARRIER_NAMES",
    "TETRAHEDRON_FACES",
    "OCTAHEDRON_FACES",
    "SIMPLE_COMPACT_LIE_ALGEBRAS",
    "Complex",
    "GroupData",
    "ConjugacyClass",
    "RealIrrep",
    "AdjointHom",
    "build_complex",
    "build_group",
    "euler_count_table",
    "euler_count_matches",
    "incidence_automorphisms",
    "orientation_preserving",
    "conjugacy_classes",
    "minimal_polynomial",
    "roots_in_q_sqrt5",
    "spectral_projectors",
    "derive_permutation_module_constituents",
    "close_character_table",
    "name_irreps",
    "decompose",
    "reductive_candidates",
    "so3_homomorphisms",
    "psu3_homomorphisms",
    "evaluate_candidate",
    "perfect_matchings",
    "centralizer_in_symmetric_group",
    "inverse_pairing_report",
    "analyze",
    "carrier_json",
    "build_receipt",
    "write_receipt",
    "check_receipt",
    "canonical_bytes",
    "implementation_pins",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
