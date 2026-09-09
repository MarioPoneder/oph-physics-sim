"""Tests for the exact-federation evidence archive builder and its simulator-free verifier."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from oph_exact import federation as F
from oph_exact import federation_archive as A

ARCHIVE_ID = "exact_federation_L0_test"


@pytest.fixture(scope="module")
def mini_archive(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("archive") / ARCHIVE_ID
    result = A.build_archive(0, out, archive_id=ARCHIVE_ID, float_sweeps=4, kernel_count=64, workers=2, log=lambda *_: None)
    return out, result


def run_verifier(archive: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(archive / "verify_archive.py")], capture_output=True, text=True, cwd=str(archive))


def test_mini_archive_builds_and_verifies(mini_archive: tuple[Path, dict]) -> None:
    out, result = mini_archive
    assert result["verify_returncode"] == 0, result["verify_output"]
    manifest = json.loads((out / "archive_manifest.json").read_text(encoding="ascii"))
    config = json.loads((out / "config.json").read_text(encoding="ascii"))
    assert manifest["schema"] == config["schema"] == A.ARCHIVE_SCHEMA
    assert manifest["archive_id"] == config["archive_id"] == ARCHIVE_ID
    assert manifest["source"]["revision"] == "uncommitted-working-tree"
    assert set(manifest["source"]["modules"]) == {"oph_exact/federation.py", "oph_exact/carrier.py"}
    assert manifest["source"]["builder"]["oph_exact/federation_archive.py"] == F.sha256_file(A.BUILDER_PATH)
    listed = {row["path"] for row in manifest["inventory"]}
    on_disk = {p.name for p in out.iterdir() if p.is_file()} - set(A.CONTROL_FILES)
    assert listed == on_disk
    assert {"config.json", "primitives.npz", "integer_law_port_pair.json", "integer_terminal_port_pair.npz", "float_law_port_pair.json",
            "float_terminal_port_pair.npz", "float_law_isolated.json", "spectrum_port_pair.json", "fiedler_port_pair.npz",
            "kernel_readout.json", "kernel_matrices.npz"} <= listed
    assert manifest["exact_verification"]["verifier_sha256"] == F.sha256_file(out / "verify_archive.py")
    assert config["carriers"] == 20 and config["seams"] == {**config["seams"], "intra": 600, "inter": 30, "total": 630}
    assert config["schedule_count"] == 16 and config["rng"]["bit_generator"] == "PCG64"
    integer = json.loads((out / "integer_law_port_pair.json").read_text(encoding="ascii"))
    assert integer["unique_quotient_hash_count"] == 1 and integer["quotient_hash_equals_expected_all"] is True
    assert len(integer["schedules"]) == 16 and all(s["terminated"] for s in integer["schedules"])
    assert "draw_sha256_per_sweep" in integer["schedules"][0]
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "PASS:" in readme and "—" not in readme and "---" not in readme
    proc = run_verifier(out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("PASS")


def test_mini_archive_matches_lane_receipt(mini_archive: tuple[Path, dict]) -> None:
    if not F.RECEIPT_PATH.exists():
        pytest.skip("lane receipt not built")
    out, _ = mini_archive
    receipt = F.load_receipt(F.RECEIPT_PATH)
    integer = json.loads((out / "integer_law_port_pair.json").read_text(encoding="ascii"))
    assert integer["expected_quotient_hash"] == receipt["rungs"]["L0/port_pair"]["integer_law"]["expected_terminal_hash"]
    lane_entries = receipt["rungs"]["L0/port_pair"]["integer_law"]["entries"]
    assert [s["attempts_to_balanced_class"] for s in integer["schedules"]] == [e["attempts_to_balanced_class"] for e in lane_entries]
    isolated = json.loads((out / "float_law_isolated.json").read_text(encoding="ascii"))
    assert isolated["expected_terminal_quotient_hash"] == receipt["rungs"]["L0/isolated"]["mean_law_float"]["expected_terminal_hash"]
    spectrum = json.loads((out / "spectrum_port_pair.json").read_text(encoding="ascii"))
    assert abs(spectrum["laplacian_lambda_2"] - receipt["rungs"]["L0/port_pair"]["synchronous_operator"]["laplacian_lambda_2"]) < 1e-9


def test_verifier_imports_no_simulator(mini_archive: tuple[Path, dict]) -> None:
    out, _ = mini_archive
    source = (out / "verify_archive.py").read_text(encoding="utf-8")
    assert source == A.VERIFIER_SOURCE
    tree = ast.parse(source)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".")[0])
    assert modules <= {"__future__", "hashlib", "json", "sys", "time", "pathlib", "typing", "numpy"}
    assert "oph_exact" not in source and "oph_fpe" not in source and "scipy" not in source


def _mutant(mini_archive: tuple[Path, dict], tmp_path: Path) -> Path:
    out, _ = mini_archive
    mutant = tmp_path / "mutant"
    shutil.copytree(out, mutant)
    return mutant


def test_verifier_fails_on_one_flipped_byte(mini_archive: tuple[Path, dict], tmp_path: Path) -> None:
    mutant = _mutant(mini_archive, tmp_path)
    target = mutant / "integer_terminal_port_pair.npz"
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0x01
    target.write_bytes(bytes(data))
    proc = run_verifier(mutant)
    assert proc.returncode == 1
    assert proc.stderr.startswith("FAIL:")


def test_verifier_fails_on_a_ledger_edit_behind_a_fresh_manifest(mini_archive: tuple[Path, dict], tmp_path: Path) -> None:
    mutant = _mutant(mini_archive, tmp_path)
    path = mutant / "integer_law_port_pair.json"
    block = json.loads(path.read_text(encoding="ascii"))
    block["schedules"][0]["V_ledger"][1] -= 2
    A.write_json(path, block)
    manifest = json.loads((mutant / "archive_manifest.json").read_text(encoding="ascii"))
    rows, digest, total = A.inventory(mutant)
    manifest["inventory"] = rows
    manifest["curated_archive"]["inventory_sha256"] = digest
    manifest["curated_archive"]["total_bytes"] = total
    A.write_json(mutant / "archive_manifest.json", manifest)
    proc = run_verifier(mutant)
    assert proc.returncode == 1
    assert "replayed V differs" in proc.stderr or "ledger" in proc.stderr
