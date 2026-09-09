"""Tests for the exact carrier-class selection lane (L3).

Producer: ``oph_exact/carrier_class.py``.  Verifier:
``oph_exact/verify_carrier_class_independent.py``.  Receipt:
``data/exact/carrier_class_selection_receipt.json``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oph_exact import carrier
from oph_exact import carrier_class as cc
from oph_exact import verify_carrier_class_independent as verifier

RECEIPT = cc.DEFAULT_RECEIPT


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_group_orders() -> None:
    expected = {"tetrahedron": (24, 12), "octahedron": (48, 24), "icosahedron": (120, 60)}
    for name, (automorphisms, rotations) in expected.items():
        gd = cc.analyze(name)["group"]
        assert len(gd.automorphisms) == automorphisms
        assert gd.order == rotations
        assert all(cc.group_axiom_checks(gd.rotations).values())


def test_class_sizes_and_geometric_invariants() -> None:
    expected = {
        "tetrahedron": {"o1f4": 1, "o2f0": 3, "o3f1a": 4, "o3f1b": 4},
        "octahedron": {"o1f6": 1, "o2f2": 3, "o2f0": 6, "o3f0": 8, "o4f2": 6},
        "icosahedron": {"o1f12": 1, "o2f0": 15, "o3f0": 20, "o5f2s1": 12, "o5f2s2": 12},
    }
    for name, sizes in expected.items():
        gd = cc.analyze(name)["group"]
        assert {c.key: c.size for c in gd.classes} == sizes
        edge_half_turns = [c for c in gd.classes if c.key == "o2f0"]
        assert len(edge_half_turns) == 1
        assert edge_half_turns[0].fixed_vertices == 0
        assert edge_half_turns[0].axis["type"] == "edge_midpoint_axis"


def test_permutation_character_decompositions() -> None:
    assert cc.analyze("tetrahedron")["chi_p_decomposition"] == {"1": 1, "3": 1}
    assert cc.analyze("octahedron")["chi_p_decomposition"] == {"1": 1, "2": 1, "3": 1}
    assert cc.analyze("icosahedron")["chi_p_decomposition"] == {"1": 1, "3": 1, "3'": 1, "5": 1}
    a = cc.analyze("icosahedron")
    gd = a["group"]
    three = next(r for r in a["table"] if r.name == "3")
    three_prime = next(r for r in a["table"] if r.name == "3'")
    phi = cc.GOLDEN
    one_minus_phi = cc.q5_sub(cc.ONE, phi)
    assert gd.value_by_key(three.values, "o5f2s1") == phi
    assert gd.value_by_key(three.values, "o5f2s2") == one_minus_phi
    assert gd.value_by_key(three_prime.values, "o5f2s1") == one_minus_phi
    assert gd.value_by_key(three_prime.values, "o5f2s2") == phi
    assert [r.name for r in a["table"]] == ["1", "3", "3'", "4", "5"]
    assert a["spectral"]["eigenvalues"] == [
        {"eigenvalue": ["-1", "0"], "dimension": 5},
        {"eigenvalue": ["5", "0"], "dimension": 1},
        {"eigenvalue": ["0", "1"], "dimension": 3},
        {"eigenvalue": ["0", "-1"], "dimension": 3},
    ]


def test_octahedral_trace_witness() -> None:
    a = cc.analyze("octahedron")
    assert [c["label"] for c in a["candidates"]] == ["su(2)+su(2)"]
    assert a["empty_candidates"] == [
        {"centre_dim": 1, "semisimple_dim": 5, "reason": "no compact semisimple Lie algebra of this dimension"}
    ]
    (evaluation,) = a["evaluations"]
    assert evaluation["assignment_count"] == 10
    assert evaluation["survivors"] == []
    witness = evaluation["uniform_witness_class"]
    assert witness["class"] == "o2f0"
    assert witness["permutation_character"] == ["0", "0"]
    assert witness["candidate_values"] == [["-2", "0"], ["2", "0"], ["6", "0"]]
    assert a["survivors"] == []
    labels = [h["label"] for h in a["hom_reports"]["su(2)"]["homomorphisms"]]
    assert sorted(labels) == sorted(["3", "sgn+2", "1+1+1", "1+sgn+sgn"])


def test_tetrahedral_survivor_and_centralizer() -> None:
    a = cc.analyze("tetrahedron")
    assert a["survivors"] == [{"candidate": "u(1)+su(2)", "homomorphisms": ["3"]}]
    assert a["empty_candidates"][0]["semisimple_dim"] == 4
    pairing = a["pairing"]
    assert pairing["perfect_matching_count"] == 3
    assert pairing["commuting_count"] == 0
    assert pairing["centralizer_order"] == 1
    assert pairing["centralizer_elements"] == [[0, 1, 2, 3]]


def test_icosahedral_survivors_and_antipode() -> None:
    a = cc.analyze("icosahedron")
    assert a["survivors"] == [
        {"candidate": "u(1)+su(2)+su(3)", "homomorphisms": ["3", "3'"]},
        {"candidate": "u(1)+su(2)+su(3)", "homomorphisms": ["3'", "3"]},
    ]
    by_label = {e["candidate"]: e for e in a["evaluations"]}
    four = by_label["su(2)+su(2)+su(2)+su(2)"]
    assert four["assignment_count"] == 15
    assert four["survivors"] == []
    assert four["trivial_multiplicity_values"] == [0, 3, 6, 9, 12]
    mixed = by_label["u(1)+su(2)+su(3)"]
    assert mixed["assignment_count"] == 9
    survivors = [x["by_ideal"] for x in mixed["assignments"] if x["survives"]]
    assert survivors == [["su(2):3", "su(3):3'"], ["su(2):3'", "su(3):3"]]
    su3 = {h["label"]: h["adjoint_decomposition"] for h in a["hom_reports"]["su(3)"]["homomorphisms"]}
    assert su3["3"] == {"3": 1, "5": 1} and su3["3'"] == {"3'": 1, "5": 1}
    assert a["outer"]["abstract_outer_automorphism"]["swaps_three_and_three_prime"] is True
    assert a["outer"]["reflection_conjugation"]["acts_trivially_on_classes"] is True
    pairing = a["pairing"]
    assert pairing["perfect_matching_count"] == 10395
    assert pairing["commuting_count"] == 1
    assert tuple(pairing["commuting_fixed_point_free_involutions"][0]) == carrier.antipode()
    assert pairing["centralizer_order"] == 2
    assert pairing["graph_distance_of_pairs"] == [3]


def test_euler_count_table() -> None:
    rows = {row["degree"]: row for row in cc.euler_count_table()}
    assert rows[1]["status"] == "excluded_non_integral" and rows[1]["F"] == "4/5"
    assert rows[2]["status"] == "excluded_non_simplicial" and rows[2]["F"] == "2"
    assert rows[3]["V"] == "4" and rows[4]["V"] == "6" and rows[5]["V"] == "12"
    assert rows[6]["status"] == "excluded_positivity" and rows[7]["status"] == "excluded_positivity"


def test_frozen_receipt(receipt: dict) -> None:
    assert receipt["schema"] == cc.SCHEMA
    assert cc.check_receipt(RECEIPT) == []
    faces = receipt["carriers"]["icosahedron"]["faces"]
    assert [tuple(f) for f in faces] == list(carrier.oriented_faces())
    assert receipt["scope"]["declared"] and receipt["scope"]["not_claimed"]
    assert receipt["claim_boundary"]
    assert sorted(receipt["implementation_pins"]) == ["carrier_module", "producer", "test", "verifier"]
    reference = receipt["paper_reference"]
    assert reference["euler_proposition"] == "Uniform triangulations of the sphere"
    assert reference["carrier_proposition"] == "Carrier selection among uniform spherical triangulations"
    assert reference["forcing_theorem_label"] == "thm:forcing"
    assert "sha256" not in reference


def test_verifier_passes() -> None:
    assert verifier.verify(RECEIPT) == []


def _write(tmp_path: Path, obj: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(cc.canonical_bytes(obj))
    return path


def _redigest(obj: dict) -> dict:
    body = {k: v for k, v in obj.items() if k != "receipt_sha256"}
    body["receipt_sha256"] = "sha256:" + hashlib.sha256(cc.canonical_bytes(body)).hexdigest()
    return body


def test_verifier_fails_on_mutation(tmp_path: Path, receipt: dict) -> None:
    mutated = copy.deepcopy(receipt)
    mutated["carriers"]["octahedron"]["survivors"] = [{"candidate": "su(2)+su(2)", "homomorphisms": ["3", "3"]}]
    failures = verifier.verify(_write(tmp_path, _redigest(mutated), "survivor.json"))
    assert any("survivor" in f for f in failures)

    mutated = copy.deepcopy(receipt)
    mutated["carriers"]["icosahedron"]["classes"][1]["size"] = 14
    failures = verifier.verify(_write(tmp_path, _redigest(mutated), "class.json"))
    assert any("class data" in f for f in failures)

    mutated = copy.deepcopy(receipt)
    pairing = mutated["carriers"]["icosahedron"]["equivariant_inverse_pairing"]
    pairing["commuting_fixed_point_free_involutions"] = [list(range(1, 12)) + [0]]
    failures = verifier.verify(_write(tmp_path, _redigest(mutated), "pairing.json"))
    assert any("commuting involutions" in f for f in failures)

    mutated = copy.deepcopy(receipt)
    mutated["carriers"]["tetrahedron"]["survivors"] = []
    failures = verifier.verify(_write(tmp_path, mutated, "digest.json"))
    assert any("self-digest" in f for f in failures)
