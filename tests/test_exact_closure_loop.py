from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from oph_exact import carrier
from oph_exact import closure_loop as cl
from oph_exact.verify_closure_loop_independent import (
    IndependentVerificationError,
    recover_invariants as verifier_recover,
    verify_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data" / "exact" / "closure_loop_receipt.json"
LOG_DIR = ROOT / "data" / "exact" / "closure_loop_logs"


def _roundtrip(log: dict) -> dict:
    return json.loads(cl.log_text(log).decode("ascii"))


def _rehash(receipt: dict) -> dict:
    receipt = copy.deepcopy(receipt)
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = "sha256:" + hashlib.sha256(cl.canonical_bytes(receipt)).hexdigest()
    return receipt


@pytest.fixture(scope="module")
def fresh_icosahedral_recovery() -> dict:
    log = _roundtrip(cl.run_source(cl.icosahedron_spec(), seed=7))
    assert set(log) == {"schema", "ports", "carriers", "ports_per_carrier", "initial", "events", "final", "probe"}
    return cl.recover_specification(log)


def test_log_carries_no_structure_description() -> None:
    log = cl.run_source(cl.icosahedron_spec(), seed=3, events=40)
    text = cl.log_text(log).decode("ascii")
    for word in ("seam", "rotation", "icosahedron", "mean", "uniform", "law", "group"):
        assert word not in text.lower().replace("seams", "")
    assert all(len(e["changed"]) == 2 for e in log["events"])


def test_recovery_of_seams_pairing_group_rule_gram(fresh_icosahedral_recovery: dict) -> None:
    rec = fresh_icosahedral_recovery
    comp = rec["component"]
    inv = rec["invariants"]
    assert rec["components"] == 1
    assert inv["ports"] == 12 and inv["seams"] == 30 and inv["degree_sequence"] == [5] * 12
    assert sorted(tuple(s) for s in comp["seams"]) == sorted(carrier.seams())
    assert inv["pairing_type"] == "distance_three_involution"
    assert tuple(comp["pairing"]["distance_three_pairing"]) == carrier.antipode()
    assert inv["automorphism_order"] == 120 and inv["rotation_order"] == 60
    assert comp["orientation"]["orientation_classes"] == 2
    assert comp["orientation"]["rotation_group_identical_for_both_classes"] is True
    assert inv["rule_class"] == "seam_mean" and not inv["nonconservative_events_present"]
    assert inv["schedule_law"] == "uniform"
    assert inv["gram_rank"] == 3 and inv["gram_step_index"] == 4
    assert inv["gram_eigenvalues"] == [4.0, 4.0, 4.0] + [0.0] * 9
    kernel = np.array(comp["gram"]["kernel"])
    assert np.allclose(kernel, carrier.intrinsic_gram(), atol=1e-9)
    assert comp["gram"]["rank_by_step"] == {"1": 11, "5": 11, "30": 11, "100": 11, "300": 3}
    assert comp["gram"]["prediction_within_readback_tolerance"] is True
    assert inv["strict_descent_violation_count"] == 0
    assert comp["descent"]["laplacian_form_increase_events"] > 0
    assert inv["terminal_invariant_type"] == "state"


def test_lie_type_character_test(fresh_icosahedral_recovery: dict) -> None:
    lie = fresh_icosahedral_recovery["component"]["lie_type"]
    assert lie["split"] == "1+3+8" and lie["assignment_count"] == 2
    assert lie["trivial_isotypic_dimension"] == 1 and lie["commutant_dimension"] == 4
    assert [c["label"] for c in lie["candidate_splits"]] == ["3+3+3+3", "1+3+8"]
    dims = sorted(b["dimension"] for b in lie["bands"])
    assert dims == [1, 3, 3, 5]
    slow = next(b for b in lie["bands"] if b["laplacian_eigenvalue"] == ["5", "-1"])
    assert sorted(slow["character_values_by_element_order"]["5"]) == [["1/2", "-1/2"], ["1/2", "1/2"]]
    assert slow["character_values_by_element_order"]["2"] == [["-1", "0"]]
    witnesses = {a[1]["witness"][0] for a in lie["surviving_splits"][0]["assignments"]}
    assert all(w.startswith("adjoint_su3_via_so3_rep_") for w in witnesses)


def test_fixed_point_true_for_canonical_and_false_for_overwrite() -> None:
    canonical = cl.run_loop(cl.icosahedron_spec(), seed=11, iterations=1)
    assert canonical["invariant_vectors_agree"] and canonical["fixed_point"]
    assert canonical["fixed_point_level"] == "state" and canonical["receipt"] == cl.FIXED_POINT_RECEIPT
    overwrite = cl.run_loop(cl.icosahedron_spec(cl.RULE_OVERWRITE), seed=12, iterations=1)
    first = overwrite["stages"][0]["recovered"]
    assert first["invariants"]["rule_class"] == "overwrite"
    assert first["component"]["rule"]["nonconservative_events"] > 0
    assert first["invariants"]["terminal_invariant_type"] == "none"
    assert not first["invariants"]["terminal_schedule_independent"]
    assert not overwrite["fixed_point"] and overwrite["receipt"] == "NO_FIXED_POINT"


def test_negative_controls_recover_their_own_specifications() -> None:
    tet = cl.run_loop(cl.tetrahedron_spec(), seed=21, iterations=1)
    inv = tet["stages"][0]["recovered"]["invariants"]
    assert (inv["ports"], inv["seams"], inv["automorphism_order"], inv["rotation_order"]) == (4, 6, 24, 12)
    assert inv["pairing_type"] == "none" and inv["farthest_pairing_distance"] is None
    assert inv["gram_rank"] == 3 and inv["lie_split"] == "1+3"
    assert tet["fixed_point"]
    octa = cl.run_loop(cl.octahedron_spec(), seed=22, iterations=1)
    inv = octa["stages"][0]["recovered"]["invariants"]
    comp = octa["stages"][0]["recovered"]["component"]
    assert (inv["ports"], inv["seams"], inv["automorphism_order"], inv["rotation_order"]) == (6, 12, 48, 24)
    assert inv["pairing_type"] == "none" and inv["farthest_pairing_distance"] == 2
    assert comp["pairing"]["farthest_port_pairing"]["pairing"] == [1, 0, 3, 2, 5, 4]
    assert inv["gram_rank"] == 3 and inv["lie_split"] == "none"
    assert [c["label"] for c in comp["lie_type"]["candidate_splits"]] == ["3+3"]
    assert octa["fixed_point"]
    integer = cl.run_loop(cl.icosahedron_spec(cl.RULE_INTEGER), seed=23, iterations=1)
    inv = integer["stages"][0]["recovered"]["invariants"]
    rule = integer["stages"][0]["recovered"]["component"]["rule"]
    assert inv["rule_class"] == "integer_nearest_agreement" and inv["tie_law"] == "uniform"
    assert 0.35 < rule["tie_ceiling_to_lower_index_frequency_float"] < 0.65
    assert inv["terminal_invariant_type"] == "quotient_multiset"
    assert integer["fixed_point"] and integer["fixed_point_level"] == "quotient_multiset"


def test_matrix_power_routes_agree() -> None:
    lap = carrier.laplacian().tolist()
    m = tuple(tuple((60 if i == j else 0) - lap[i][j] for j in range(12)) for i in range(12))
    for exponent in (1, 2, 7, 40, 200):
        assert cl.integer_matrix_power(m, exponent) == cl.integer_matrix_power_naive(m, exponent)
    assert cl.readback_string(1, 12, 6) == "833333E-7"
    assert cl.readback_string(-25, 100, 4) == "-2500E-4"
    assert cl.readback_string(0, 5, 4) == "0"


def test_verifier_recovery_matches_producer(fresh_icosahedral_recovery: dict) -> None:
    log = _roundtrip(cl.run_source(cl.icosahedron_spec(), seed=7))
    assert verifier_recover(log) == fresh_icosahedral_recovery["invariants"]


def test_frozen_receipt_matches_a_rebuilt_canonical_loop() -> None:
    receipt = cl.load_receipt(RECEIPT)
    assert receipt["schema"] == cl.SCHEMA
    assert _rehash(receipt)["receipt_sha256"] == receipt["receipt_sha256"]
    assert cl.canonical_bytes(receipt) == RECEIPT.read_bytes()
    for label, pin in receipt["pins"].items():
        assert cl.sha256_file(ROOT / pin["path"]) == pin["sha256"], label
    loop = cl.run_loop(cl.icosahedron_spec(), cl.LOOP_SEEDS["icosahedron_seam_mean"], cl.LOOP_ITERATIONS["icosahedron_seam_mean"])
    frozen = receipt["canonical_loop"]
    for stage, frozen_stage in zip(loop["stages"], frozen["stages"]):
        assert stage["log_sha256"] == frozen_stage["log_sha256"]
        assert stage["recovered"]["invariants"] == frozen_stage["invariants"]
    for role, text in loop["_logs"]:
        assert (LOG_DIR / f"icosahedron_seam_mean__{role}.json").read_bytes() == text
    assert frozen["fixed_point"] and frozen["receipt"] == cl.FIXED_POINT_RECEIPT
    assert receipt["loops"]["icosahedron_overwrite"]["fixed_point"] is False
    assert receipt["negative_controls"]["gram_rank_alone_does_not_select_icosahedron"] is True
    assert receipt["federation"]["components"] == 20
    assert receipt["federation"]["component_invariants_equal_single_carrier"] is True


def test_verifier_passes_and_fails_on_mutation(tmp_path: Path) -> None:
    receipt = cl.load_receipt(RECEIPT)
    summary = verify_receipt(receipt, ROOT, LOG_DIR)
    assert summary["recomputed_logs"] >= 15
    mutated = copy.deepcopy(receipt)
    mutated["loops"]["icosahedron_seam_mean"]["stages"][0]["invariants"]["gram_rank"] = 4
    with pytest.raises(IndependentVerificationError):
        verify_receipt(_rehash(mutated), ROOT, LOG_DIR)
    mutated = copy.deepcopy(receipt)
    mutated["canonical_loop"]["fixed_point"] = False
    with pytest.raises(IndependentVerificationError):
        verify_receipt(_rehash(mutated), ROOT, LOG_DIR)
    with pytest.raises(IndependentVerificationError):
        verify_receipt(mutated, ROOT, LOG_DIR)
    corrupt_dir = tmp_path / "logs"
    corrupt_dir.mkdir()
    for path in LOG_DIR.glob("*.json"):
        (corrupt_dir / path.name).write_bytes(path.read_bytes())
    target = corrupt_dir / "icosahedron_seam_mean__inhabited_structure.json"
    log = json.loads(target.read_text(encoding="ascii"))
    log["events"][0]["changed"] = {k: [v[0], v[0]] for k, v in log["events"][0]["changed"].items()}
    target.write_bytes(cl.log_text(log))
    with pytest.raises(IndependentVerificationError):
        verify_receipt(receipt, ROOT, corrupt_dir)
