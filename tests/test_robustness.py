"""What keeps a multi-day run from failing quietly: the protocol's keys, locks, killed processes, exit codes,
broken files (final review B2-B11), and the cross-dataset table of the stage-1 report (H51)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import pytest
import torch

from seqrec_eval import cli
from seqrec_eval.protocol import ProtocolError, load_protocol
from seqrec_eval.report import _across_datasets
from seqrec_eval.results import read_json
from seqrec_eval.runner import RunLock
from test_smoke import PROTOCOL


def _protocol(tmp_path, old: str, new: str):
    text = PROTOCOL.replace(old, new, 1)
    assert text != PROTOCOL
    (tmp_path / "protocol.toml").write_text(text)
    return load_protocol(tmp_path / "protocol.toml")


@pytest.mark.parametrize("old, new, named", [
    ("[models.ease]\nfamily = \"matrix\"\nmax_items = 10", "[models.ease]\nfamily = \"matrix\"\nmax_item = 10",
     "'max_item' (did you mean 'max_items'?)"),
    ("max_val_users = 30", "max_val_user = 30", "'max_val_user' (did you mean 'max_val_users'?)"),
    ("requests_per_bin = 5", "requests_per_bins = 5", "'requests_per_bins'"),
    ("new_item_diagnostic = true", "new_item_diagnostics = true", "'new_item_diagnostics'"),
    ("[baselines.markov]\nkind = \"markov\"", "[baselines.markov]\nkind = \"markov\"\nspace_ = 1", "'space_'"),
])
def test_a_misspelt_key_is_refused_by_name(tmp_path, old, new, named):
    # a misspelt key once fell back to its default: `max_item` let EASE run unbounded, `scop` refitted everything
    with pytest.raises(ProtocolError, match="unknown key") as refused:
        _protocol(tmp_path, old, new)
    assert named in str(refused.value)


def test_a_misspelt_sweep_key_is_refused(tmp_path):
    (tmp_path / "protocol.toml").write_text(PROTOCOL + '\n[ablations.h]\ntransform = "history_length"\n'
                                            'levels = [2]\nscop = "inference"\n')
    with pytest.raises(ProtocolError, match="'scop' \\(did you mean 'scope'\\?\\)"):
        load_protocol(tmp_path / "protocol.toml")


@pytest.mark.parametrize("old, new, message", [
    ("seeds = [0, 1]", "seeds = [-1, 0]", "non-negative"),
    ("history_bins = [1, 10, 30]", "history_bins = [1, 30, 10]", "increasing order"),
    ("history_bins = [1, 10, 30]", "history_bins = [0, 10, 30]", "positive integers"),
])
def test_seeds_and_latency_bins_are_checked_when_read(tmp_path, old, new, message):
    with pytest.raises(ProtocolError, match=message):
        _protocol(tmp_path, old, new)


def test_the_repository_protocols_hold_only_known_keys():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("protocol.toml", "protocol.local.toml"):
        load_protocol(root / name)


def test_a_lock_dies_with_its_process(tmp_path):
    # review B2: a flock is released by the kernel when its holder ends, however it ends
    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(os.path.dirname(cli.__file__) + '/..')!r})
        from seqrec_eval.runner import RunLock
        assert RunLock({str(tmp_path)!r}).acquire()
        print("held", flush=True)
        time.sleep(60)
    """)
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        lock = RunLock(tmp_path)
        assert lock.held_by_other() and not lock.acquire()
        child.send_signal(signal.SIGKILL)  # killed: no cleanup runs
        child.wait(timeout=10)
        assert not lock.held_by_other()
        assert lock.acquire()
        lock.release()
    finally:
        if child.poll() is None:
            child.kill()


def test_a_lock_names_its_last_owner_and_two_claims_in_one_process_fail(tmp_path):
    first, second = RunLock(tmp_path), RunLock(tmp_path)
    assert first.acquire()
    assert not second.acquire()  # one claim per run, whoever asks
    owner = json.loads((tmp_path / ".lock").read_text())
    assert owner["pid"] == os.getpid()
    first.release()
    assert second.acquire()
    second.release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_run_on_an_explicit_gpu_starts_its_memory_counters_in_a_fresh_process():
    # 2026-10-01, the first DGX run: `--device cuda:0` failed every run, since the counters refuse an explicit
    # device until CUDA has started. In-process tests had always started it already, so only a fresh one shows it.
    code = "from seqrec_eval.runner import _reset_peak_memory; print(_reset_peak_memory('cuda:0'))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "True", out.stderr


def test_exit_codes_tell_failure_from_unfinished_work():
    assert cli._exit_code({"done": 3, "cached": 2, "skipped": 1}) == 0
    assert cli._exit_code({"done": 3, "running-elsewhere": 1}) == 3
    assert cli._exit_code({"waiting-for-stage1": 1}) == 3
    assert cli._exit_code({"done": 3, "failed": 1, "not-ready": 1}) == 1
    assert cli._exit_code({"stale-selection": 1}) == 1


def test_a_file_cut_short_is_named_when_read(tmp_path):
    (tmp_path / "done.json").write_text("")
    with pytest.raises(ValueError, match="done.json is not valid JSON"):
        read_json(tmp_path / "done.json")


def test_two_prepares_of_one_dataset_do_not_run_at_once(tmp_path):
    from seqrec_eval.splits import prepare_split

    (tmp_path / "protocol.toml").write_text(PROTOCOL)
    protocol = load_protocol(tmp_path / "protocol.toml")
    held = RunLock(tmp_path / "work" / "splits" / ".synth.prepare")
    assert held.acquire()
    try:
        with pytest.raises(RuntimeError, match="another process is preparing synth"):
            prepare_split(protocol, "synth", data_dir=tmp_path / "data", work_dir=tmp_path / "work")
    finally:
        held.release()


def test_library_temp_files_go_to_the_work_dir(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)
    (tmp_path / "protocol.toml").write_text(PROTOCOL)
    assert cli.main(["--protocol", str(tmp_path / "protocol.toml"), "--work-dir", str(tmp_path / "work"),
                     "plan"]) == 0
    assert tempfile.gettempdir() == str(tmp_path / "work" / "tmp")


def test_the_cross_dataset_table_counts_wins_without_pooling():
    outcomes = {"a": {"gru": (0.01, 0.001), "sasrec": (-0.02, 0.01)},
                "b": {"gru": (0.004, 0.30), "sasrec": (-0.01, 0.02)},
                "c": {"gru": (0.02, 0.0001)}}
    table = _across_datasets(outcomes, ["a", "b", "c"], "elsa")
    assert "| gru | +* | + | +* | 2 | 1 | 0 |" in table
    assert "| sasrec | −* | −* | — | 0 | 0 | 2 |" in table
