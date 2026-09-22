"""Native binding, independent replay and adversarial primitive-read checks."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oph_fpe.bulk import primitive_source_reads as produce
from oph_fpe.bulk import verify_primitive_source_reads_independent as check


@pytest.fixture(scope="module")
def packet():
    return produce.produce()


def reseal(value):
    value["sha256"] = hashlib.sha256(check.canonical({k:v for k,v in value.items() if k != "sha256"})).hexdigest()
    return value


def test_native_driver_binding_and_complete_replay(packet):
    receipt = check.verify(packet, produce.ROOT)
    assert receipt["source_bytes_checked"] is True
    assert [r["commits"] for r in receipt["cases"]] == [8, 16, 24, 24, 48, 96, 24]
    assert [r["ideal_real_read_algebra"]["observation_rank"] for r in receipt["cases"]] == [17, 19, 23, 23, 23, 23, 23]
    assert receipt["quantum"]["certified_nonzero_unitary_entries"] == 144
    assert F(receipt["quantum"]["equal_input_intensities_output_gap_lower_bound"]) > 0
    assert receipt["M1_derived"] is False


@pytest.mark.parametrize("path,value", [
    (("scope", "M1_derived"), True),
    (("cases", 0, "carriers"), True),
    (("cases", 0, "commits"), 0),
    (("cases", 0, "noops"), 1),
    (("cases", 0, "native_log_sha256"), "sha256:"+"0"*64),
    (("cases", 2, "final_hex", 0), "0x0.0p+0"),
    (("cases", 4, "order"), []),
    (("cases", 4, "seams", 0, 1), 0),
    (("cases", 1, "initial_hex", 0), "nan"),
    (("cases", 2, "observer_records", 0, 3), 11),
    (("cases", 2, "observer_records", 0, 4, 0), "0x0.0p+0"),
    (("cases", 2, "observer_records"), []),
    (("quantum", "unitary_hex", 0, 0), ["0x0.0p+0", "0x0.0p+0"]),
    (("quantum", "phase_output_hex", 1), "0x1.0000000000000p-1"),
    (("quantum", "laplacian", 0, 0), 4),
    (("quantum", "uniform_snapshot_feedback", "native_next_port"), 0),
    (("quantum", "uniform_snapshot_feedback", "stabilizer_permutation"), list(range(12))),
    (("cases",), []),
])
def test_resealed_forgery(packet, path, value):
    bad = deepcopy(packet)
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises((ValueError, KeyError)):
        check.verify(reseal(bad))


def test_smallest_propagator_entries_cannot_be_forged_to_zero(packet):
    bad = deepcopy(packet)
    matrix = bad["quantum"]["unitary_hex"]
    i, j = min(((i,j) for i in range(12) for j in range(12)),
               key=lambda pair: sum(abs(float.fromhex(v)) for v in matrix[pair[0]][pair[1]]))
    matrix[i][j] = ["0x0.0p+0", "0x0.0p+0"]
    with pytest.raises(ValueError, match="Taylor"):
        check.verify(reseal(bad))


def test_normalization_recovers_an_unobserved_twelfth_coordinate():
    # Arbitrary matching diagnostic, not an exported registered federation.
    mate = {i:i+12 for i in range(12)} | {i+12:i for i in range(12)}
    mate.update({i:i+12 for i in range(24,36)})
    mate.update({i+12:i for i in range(24,36)})
    result = check.read_algebra(4, mate, set(range(11)))
    assert 23 not in result["directly_observed_slots"]
    assert 23 in result["recoverable_slots_with_normalization"]
    assert result["observation_rank"] == 22


def test_two_hidden_coordinates_are_an_actual_observation_kernel():
    n = 4
    # A full captured source topology, with every permitted operation included.
    row = produce.case(n, 16, 1)
    mate = {a:b for _,a,b in row["seams"]} | {b:a for _,a,b in row["seams"]}
    result = check.read_algebra(n, mate, set(range(12*n)))
    p, q = result["hidden_pair"]
    assert p//12 == q//12 != 0
    assert result["hidden_dimension"] == 21
    assert result["global_hash_distinguishes"] is True


def test_verifier_runs_without_any_simulator_or_numerical_import(packet, tmp_path):
    artifact = tmp_path/"packet.json"
    artifact.write_bytes(check.canonical(packet))
    script = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'oph_fpe','numpy','scipy'}:
            raise AssertionError('forbidden import: '+fullname)
sys.meta_path.insert(0, Block())
module=runpy.run_path(sys.argv[1],run_name='independent')
print(module['verify'](module['strict_load'](sys.argv[2]))['M1_derived'])
'''
    result = subprocess.run([sys.executable, "-c", script, str(Path(check.__file__)), str(artifact)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("raw", ['{"x":0,"x":1}', '{"x":NaN}', '{"x":1.0}', '{}'])
def test_real_cli_rejects_trash(raw, tmp_path):
    artifact = tmp_path/"bad.json"
    artifact.write_text(raw, encoding="ascii")
    result = subprocess.run([sys.executable, str(Path(check.__file__)), str(artifact)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert '"packet_sha256"' not in result.stdout


def test_source_custody_rejects_real_file_tampering(packet, tmp_path):
    for name in packet["source"]["files"]:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((produce.ROOT/name).read_bytes())
    target = tmp_path/"oph_fpe/core/echosahedral_dynamics.py"
    target.write_bytes(target.read_bytes()+b"\n# changed source\n")
    with pytest.raises(ValueError, match="source custody"):
        check.verify(packet, tmp_path)


def test_skipped_pairs_are_not_committed_record_parents(packet):
    # A valid uniform preparation at the same operation interface, independent
    # of the Gaussian seed. Every attempt is a no-op, so no committed writer
    # can appear in the actual record graph even though the ideal word visits it.
    row = deepcopy(packet["cases"][2])
    row["initial_hex"] = row["final_hex"] = [(1/12).hex()]*48
    row["commits"], row["noops"] = 0, 32
    row["cycle_counts"] = [[0, 2]]*16
    row["native_log_sha256"] = "sha256:"+hashlib.sha256(b"").hexdigest()
    port = 0
    values = [check.clean(1/12)]*12
    for record in row["observer_records"]:
        record[3], record[4] = port, [v.hex() for v in values]
        port = (port+1+abs(round(1e6*sum((j+1)*v for j,v in enumerate(values))))%11)%12
    result = check.case(row, (4,16,1))
    assert result["observer_payload"]["repair_to_record_edges"] == []
    assert result["maximum_version"] == 0
