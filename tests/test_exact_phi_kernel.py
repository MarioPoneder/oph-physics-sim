from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from oph_exact import phi_kernel as pk
from oph_exact import verify_phi_kernel_independent as verifier

V = pk.PRIMARY_POTENTIAL
PHI = pk.SECONDARY_POTENTIAL


def test_sector_sizes_and_absorbing_sets() -> None:
    three = pk.build_sector(3)
    four = pk.build_sector(4)
    assert three.size == 364 and four.size == 1365
    assert len(three.absorbing_indices()) == 220
    assert len(four.absorbing_indices()) == 495
    assert len(pk.phi_minimal_indices(three)) == 20
    assert len(pk.phi_minimal_indices(four)) == 30
    assert min(three.phi) == 9 and max(three.phi) == 45
    assert min(four.phi) == 10 and max(four.phi) == 80
    assert min(three.v) == 3 and max(three.v) == 9
    assert min(four.v) == 4 and max(four.v) == 16


@pytest.mark.parametrize("total", [3, 4])
def test_absorbing_set_equals_v_minimal_set(total: int) -> None:
    sector = pk.build_sector(total)
    assert set(sector.absorbing_indices()) == set(pk.v_minimal_indices(sector))
    assert len(pk.v_minimal_indices(sector)) == len(sector.absorbing_indices())
    assert not set(pk.phi_minimal_indices(sector)) == set(sector.absorbing_indices())


@pytest.mark.parametrize("total", [3, 4])
@pytest.mark.parametrize("potential", [V, PHI])
def test_decrement_nonpositive_zero_exactly_on_waits(total: int, potential: str) -> None:
    summary = pk.descent_summary(pk.build_sector(total), potential)
    assert summary["all_nonpositive"]
    assert summary["zero_iff_wait"]
    assert summary["state_changing_all_negative"]
    assert summary["state_label_pairs"] == pk.build_sector(total).size * 60


@pytest.mark.parametrize("total", [3, 4])
def test_v_unit_transfer_identity_and_closed_form(total: int) -> None:
    summary = pk.unit_transfer_summary(pk.build_sector(total))
    assert summary["verified"]
    assert summary["unit_transfer_identity_dv_equals_minus_two_d_minus_one"]
    assert summary["declared_move_closed_form_dv_equals_minus_two_floor_ceil"]
    assert summary["declared_move_dv_equals_sum_of_unit_transfer_decrements"]
    assert summary["unit_transfer_pairs_checked"] > 0
    assert pk.closed_form_dv(2) == -2 and pk.closed_form_dv(3) == -4 and pk.closed_form_dv(4) == -8
    assert pk.closed_form_dv(1) == 0 and pk.closed_form_dv(0) == 0
    x = tuple([3] + [0] * 11)
    assert pk.squared_norm(pk.apply_label(x, (1, 0))) - pk.squared_norm(x) == -4


def test_literal_gap_one_swap_is_v_neutral_and_raises_phi() -> None:
    literal = pk.literal_swap_summary(pk.build_sector(3))
    assert literal["literal_dv_zero_on_every_differing_pair"]
    assert literal["literal_dphi_positive"] == 900
    assert literal["max_literal_dphi"] == 4
    assert literal["pairs_where_literal_differs_from_declared"] == 3360
    state = tuple(literal["example_positive"]["state"])
    label = tuple(literal["example_positive"]["label"])
    assert pk.apply_label(state, label) == state
    assert pk.phi(pk.apply_label_literal(state, label)) - pk.phi(state) == 4
    assert pk.squared_norm(pk.apply_label_literal(state, label)) == pk.squared_norm(state)


@pytest.mark.parametrize("total", [3, 4])
def test_beta_zero_kernels_coincide(total: int) -> None:
    assert pk.build_kernel(total, 0, V).weights == pk.build_kernel(total, 0, PHI).weights
    assert all(z == 60 for z in pk.build_kernel(total, 0, V).partition)


@pytest.mark.parametrize("total", [3, 4])
@pytest.mark.parametrize("beta", [0, 1, 2])
@pytest.mark.parametrize("potential", [V, PHI])
def test_kernel_rows_sum_to_one_exactly(total: int, beta: int, potential: str) -> None:
    sector = pk.build_sector(total)
    kernel = pk.build_kernel(total, beta, potential)
    for k in range(sector.size):
        assert sum(kernel.row(k), Fraction(0)) == 1
    assert all(sum(row.values(), Fraction(0)) == 1 for row in pk.aggregated_rows(total, beta, potential))


@pytest.mark.parametrize("beta", [0, 1, 2])
@pytest.mark.parametrize("potential", [V, PHI])
def test_derived_action_on_length_two_paths(beta: int, potential: str) -> None:
    result = pk.derived_action_check(3, beta, potential, 2)
    assert result["verified"]
    assert result["label_path_count"] == 3 * 3600
    assert result["tilt_partition_constant"] == "1/3600"
    assert result["lean_partition_constant"] == "1/3600"
    assert all(entry["path_law_sum"] == "1" for entry in result["per_initial_state"])


@pytest.mark.parametrize("potential", [V, PHI])
def test_gauge_characterization_on_length_two_paths(potential: str) -> None:
    tilted = pk.gauge_check(3, 1, potential, 2)
    assert tilted["verified"]
    assert tilted["fitted_action_minus_action_exp_neg"] == "1/3"
    assert tilted["action_nonconstant"]
    outcomes = {c["candidate"]: c["tilt_equals_path_law"] for c in tilted["candidates"]}
    assert outcomes["S + c with exp(-c) = 3"]
    assert outcomes["S / 2 at multiplier 2"]
    assert not outcomes["S + f, exp(-f) = 2 on paths whose first label is label 0"]
    assert not outcomes["2 S at multiplier 1"]
    assert all(c["characterization_holds"] for c in tilted["candidates"])
    uniform = pk.gauge_check(3, 0, potential, 2)
    assert not uniform["action_nonconstant"]
    assert {c["candidate"]: c["tilt_equals_path_law"] for c in uniform["candidates"]}["2 S at multiplier 1"]


@pytest.mark.parametrize("potential", [V, PHI])
def test_non_identifiability_family_on_realized_records(potential: str) -> None:
    result = pk.non_identifiability_check(3, 1, potential, 2)
    assert result["verified"]
    assert result["correction_vanishes_on_every_realized_record"]
    assert result["corpus_midpoint_gap"] == "1/8"
    by_configuration = {u["configuration"]: u for u in result["unrealized_configurations"]}
    assert by_configuration["y_m = 2"]["difference_a2_minus_a1"] == "1"
    assert by_configuration["y_m = 1/2"]["corrections_by_a"]["1"] == "-1/8"
    assert pk.enrichment_correction(Fraction(7), pk.realized_record(17)) == 0


@pytest.mark.parametrize("potential", [V, PHI])
def test_expected_absorption_beta_zero_s3(potential: str) -> None:
    times = pk.expected_moves_to_absorption(3, 0, potential)
    sector = pk.build_sector(3)
    assert sum(times, Fraction(0)) / sector.size == Fraction(261, 91)
    top = tuple([3] + [0] * 11)
    assert times[sector.index[top]] == Fraction(27, 2)
    assert all(times[k] == 0 for k in sector.absorbing_indices())
    assert all(times[k] > 0 for k in sector.transient_indices())


def test_v_terminal_is_v_minimal_with_probability_one() -> None:
    for beta in (0, 1, 2):
        _g_phi, g_v, _h_phi, h_v = pk.terminal_functionals(3, beta, V)
        assert all(value == 1 for value in g_v)
        assert all(value == 3 for value in h_v)
        summary = pk.absorption_summary(3, beta, V)
        assert summary["terminal_v_minimal_probability_uniform_initial"] == "1"
        assert summary["expected_terminal_v_uniform_initial"] == "3"


@pytest.mark.parametrize("potential", [V, PHI])
def test_most_probable_paths_report_ties(potential: str) -> None:
    result = pk.most_probable_paths(3, 1, potential, 2)
    assert result["verified"]
    first = result["per_initial_state"][0]["per_length"][0]
    assert first["argmax_label_path_count"] == 10
    assert first["argmax_terminal_state_count"] == 10
    second = result["per_initial_state"][0]["per_length"][1]
    assert second["witness_terminal_v"] == 3
    uniform = pk.most_probable_paths(3, 0, potential, 1)["per_initial_state"][0]["per_length"][0]
    assert uniform["argmax_label_path_count"] == 60


def test_frozen_receipt_matches_replay() -> None:
    path = pk.DEFAULT_RECEIPT
    assert path.exists()
    stored = json.loads(path.read_text(encoding="ascii"))
    assert stored["schema"] == pk.SCHEMA
    assert stored["verified"] is True
    assert stored["declared"]["potentials"]["primary"] == V
    assert stored["declared"]["potentials"]["secondary"] == PHI
    assert pk.canonical_bytes(stored) == path.read_bytes()
    for key, rel in pk.PINNED_SIM_FILES.items():
        assert stored["pins"][key]["sha256"] == pk.file_sha256(pk.SIM_ROOT / rel), key
    for key, rel in pk.PINNED_META_FILES.items():
        target = pk.META_ROOT / rel
        if not target.exists():
            pytest.skip(f"theory file absent: {rel}")
        assert stored["pins"][key]["sha256"] == pk.file_sha256(target), key
    assert "instantiation_paper_tex" not in stored["pins"]
    reference = stored["paper_reference"]
    assert reference["hashing"] == "none"
    assert reference["repository"] == "oph-meta"
    assert reference["path"] == pk.PAPER_REFERENCE["path"]
    assert reference["subsection"] == "(M5) A source-selected action"
    assert pk.check_receipt(path) == []


def test_independent_verifier_passes_and_fails_on_mutation(tmp_path: Path) -> None:
    summary = verifier.verify(pk.DEFAULT_RECEIPT)
    assert summary["sectors"] == {"3": 364, "4": 1365}
    assert summary["potentials"] == [V, PHI]
    stored = json.loads(pk.DEFAULT_RECEIPT.read_text(encoding="ascii"))
    mutated = json.loads(json.dumps(stored))
    mutated["sectors"]["3"]["potentials"][V]["absorption"]["1"]["expected_moves_uniform_initial"] = "1/2"
    bad = tmp_path / "mutated.json"
    bad.write_bytes(pk.canonical_bytes(mutated))
    with pytest.raises(verifier.PhiKernelVerificationError):
        verifier.verify(bad)
    pinned = json.loads(json.dumps(stored))
    pinned["pins"]["carrier"]["sha256"] = "sha256:" + "0" * 64
    bad_pin = tmp_path / "bad_pin.json"
    bad_pin.write_bytes(pk.canonical_bytes(pinned))
    with pytest.raises(verifier.PhiKernelVerificationError):
        verifier.verify(bad_pin)
