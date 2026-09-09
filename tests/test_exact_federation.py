"""Tests for lane L1: the exact federation receipt and its independent verifier."""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from oph_exact import carrier
from oph_exact import federation as F
from oph_exact import verify_federation_independent as V

RECEIPT = F.RECEIPT_PATH


@pytest.fixture(scope="module")
def receipt() -> dict:
    if not RECEIPT.exists():
        pytest.skip("receipt not built")
    return F.load_receipt(RECEIPT)


def _without_state(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "state"}


def test_engines_are_bit_identical() -> None:
    fed = F.build_federation(1, "port_pair")
    loads = F.initial_loads(1, fed.ports)
    rng = np.random.default_rng(11)
    seq = rng.integers(0, fed.seams, size=60_000, dtype=np.int64)
    coin = rng.integers(0, 2, size=seq.size, dtype=np.int64)
    engines = ["python", "numpy"] + (["native"] if F.native_kernel() is not None else [])
    reference = None
    for name in engines:
        x = loads.astype(float).copy()
        tally = F.MeanTally()
        F.MEAN_ENGINES[name](x, fed.graph, seq, tally)
        xi = loads.astype(np.int64).copy()
        itally = F.IntegerTally()
        v, first = F.INTEGER_ENGINES[name](xi, fed.graph, seq, coin, int(np.dot(xi, xi)), fed.integer_minimum_descent(loads), itally)
        assert v == int(np.dot(xi, xi))
        assert tally.descent_violations == 0 and itally.decrement_violations == 0
        assert tally.phi_raising_moves > 0 and tally.first_raise is not None
        if reference is None:
            reference = (x, tally, xi, itally, first)
        else:
            assert np.array_equal(x, reference[0]), name
            ref = reference[1]
            assert (tally.waits, tally.descent_violations, tally.phi_raising_moves) == (ref.waits, ref.descent_violations, ref.phi_raising_moves), name
            assert tally.first_raise == ref.first_raise, name
            assert abs(tally.ledger - ref.ledger) <= 1e-9 * ref.ledger, name
            assert np.array_equal(xi, reference[2])
            assert itally == reference[3] and first == reference[4]


def test_isolated_synchronous_block_and_kernel_equal_carrier() -> None:
    fed = F.build_federation(1, "isolated")
    assert fed.mean_denominator == Fraction(60)
    so = F.synchronous_operator_receipt(fed)
    assert so["isolated_block_equals_carrier_repair_mean_exact"] is True
    for cell in (0, 37, 79):
        kernels = F.response_kernels(fed, cell)
        for n in F.KERNEL_STEPS:
            assert np.max(np.abs(kernels[n] - carrier.normalized_response_kernel(n))) < 1e-12
    limit = kernels[300]
    assert np.max(np.abs(limit - carrier.intrinsic_gram())) < 1e-12
    assert np.linalg.matrix_rank(limit, tol=1e-9) == 3


def test_terminal_equals_component_mean_across_schedules() -> None:
    fed = F.build_federation(0, "port_pair")
    loads = F.initial_loads(0, fed.ports)
    expected = fed.expected_terminal_quotient_hash(loads)
    means = fed.component_means(loads)
    hashes = set()
    for s in range(3):
        result = F.run_mean_law(fed, loads, F.schedule_seed(0, "port_pair", s))
        assert result["terminated"]
        assert result["strict_descent_violations"] == 0
        assert result["descent_ledger_relative_error_below_1e-9"]
        assert result["centered_norm_terminal"] < 1e-12
        assert result["phi_raising_moves"] > 0 and result["phi_raising_counterexample"]["delta_phi"] > 0
        assert np.max(np.abs(result["state"] - means[fed.component_of_port])) < 1e-9
        hashes.add(result["terminal_quotient_hash"])
    assert hashes == {expected}


def test_integer_quotient_unique_modulo_odd_tie_orbit() -> None:
    fed = F.build_federation(1, "port_pair")
    loads = F.initial_loads(1, fed.ports)
    expected = fed.expected_integer_quotient_hash(loads)
    states = []
    for s in range(4):
        result = F.run_integer_law(fed, loads, F.schedule_seed(1, "port_pair", s))
        assert result["terminated"] and result["descent_ledger_exact"]
        assert result["unit_transfer_decrement_identity_violations"] == 0 and result["unit_transfers"] >= result["descents"]
        assert result["quotient_hash"] == expected
        assert result["max_seam_difference_at_termination"] <= 1
        states.append(result["state"])
    # The terminal states differ as vectors (different odd-tie placements) and agree as multisets.
    assert any(not np.array_equal(states[0], s) for s in states[1:])
    assert all(np.array_equal(np.sort(states[0]), np.sort(s)) for s in states[1:])


def test_exact_run_descent_and_conservation() -> None:
    fed = F.build_federation(0, "isolated")
    loads = F.initial_loads(0, fed.ports)
    result = F.run_mean_law_exact(fed, loads, F.schedule_seed(0, "isolated", 0))
    assert result["terminated"] and result["global_descent_identity_exact"]
    assert result["total_conservation_exact"] and result["component_conservation_exact"]
    assert result["terminal_quotient_hash"] == fed.expected_terminal_quotient_hash(loads)
    assert result["float_vs_exact_max_abs_deviation"] < 1e-12


def test_phi_witness_and_descent_functional() -> None:
    witness = F.phi_nonmonotone_witness()
    assert witness["phi_rises"] is True
    assert witness["descent_drop_equals_half_square_difference"] is True


def test_receipt_frozen_and_deterministic(receipt: dict) -> None:
    assert receipt["schema"] == F.SCHEMA
    assert RECEIPT.read_text(encoding="ascii") == F.canonical_json(receipt)
    assert receipt["pins"] == F.file_pins()
    # Full rebuild of the isolated L0 block (all laws, all schedules, exact arithmetic).
    stored = receipt["rungs"]["L0/isolated"]
    rebuilt = F.build_rung_block(0, "isolated", schedules=F.SCHEDULES, engine="auto", workers=1)
    assert F.canonical_json(rebuilt) == F.canonical_json(stored)
    # Float, integer, kernel and confluence sub-blocks of the glued L0 rung.
    stored = receipt["rungs"]["L0/port_pair"]
    rebuilt = F.build_rung_block(0, "port_pair", schedules=F.SCHEDULES, engine="auto", workers=1, laws=("float", "integer"))
    for key in ("federation", "loads", "synchronous_operator", "mean_law_float", "integer_law", "local_confluence", "response_kernel"):
        assert F.canonical_json(rebuilt[key]) == F.canonical_json(stored[key]), key
    assert "mean_law_exact" in stored
    assert receipt["conventions"]["phi_single_move_witness"] == F.phi_nonmonotone_witness()


def test_receipt_verdicts_and_numbers(receipt: dict) -> None:
    assert all(receipt["verdicts"].values())
    rungs = receipt["rungs"]
    assert set(rungs) == {f"L{level}/{g}" for level in F.RECEIPT_LEVELS for g in F.GLUINGS}
    for key, block in rungs.items():
        f = block["mean_law_float"]
        assert f["schedules"] == F.SCHEDULES and f["all_terminated"] and f["unique_terminal_hash_count"] == 1
        assert f["terminal_hash_equals_expected"] and f["strict_descent_violations_total"] == 0
        i = block["integer_law"]
        assert i["unique_terminal_hash_count"] == 1 and i["all_terminated"]
        k = block["response_kernel"]
        assert k["pentagonal_cells_in_sample"] >= 1 and len(k["sampled_cells"]) >= 8
        if block["gluing"] == "isolated":
            assert k["isolated_kernel_matches_carrier_within_1e-12"] is True
            assert k["summary"]["300"]["slow_band_share"]["min"] > 1 - 1e-9
        if block["level"] in F.EXACT_LEVELS:
            e = block["mean_law_exact"]
            assert e["all_terminated"] and e["descent_identity_exact_all"] and e["conservation_exact_all"]
            assert e["unique_terminal_hash_count"] == 1


def test_verifier_passes_on_receipt_and_fails_on_mutation(receipt: dict, tmp_path: Path) -> None:
    report = V.verify(RECEIPT)
    assert report["ok"] and report["pins_verified"] == 4
    mutated = copy.deepcopy(receipt)
    entry = mutated["rungs"]["L1/port_pair"]["mean_law_float"]["entries"][3]
    entry["terminal_quotient_hash"] = "0" * 64
    path = tmp_path / "mutated.json"
    path.write_text(F.canonical_json(mutated), encoding="ascii")
    with pytest.raises(V.VerificationError):
        V.verify(path, replay=False)
    mutated = copy.deepcopy(receipt)
    mutated["rungs"]["L0/isolated"]["integer_law"]["entries"][0]["waits"] += 1
    path.write_text(F.canonical_json(mutated), encoding="ascii")
    with pytest.raises(V.VerificationError):
        V.verify(path)
    mutated = copy.deepcopy(receipt)
    mutated["pins"]["oph_exact/federation.py"] = "0" * 64
    path.write_text(F.canonical_json(mutated), encoding="ascii")
    with pytest.raises(V.VerificationError):
        V.verify(path, replay=False)
