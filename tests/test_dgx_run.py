"""scripts/dgx-run.sh: the whole suite in order, several processes per step (review-plan/plans/week-budget.md,
B8). Run against a fake seqrec-eval that records each call and exits with the codes it is told to."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "dgx-run.sh"
SWEEPS = "history_length_inference shuffle"
FAKE = f"""#!{sys.executable}
import fcntl, json, os, sys, time
args = sys.argv[1:]
rest = args[args.index("--work-dir") + 2:]
folder = os.environ["FAKE_DIR"]
with open(os.path.join(folder, "lock"), "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    codes = os.path.join(folder, rest[0] + ".codes")
    code = 0
    if os.path.exists(codes):
        queue = open(codes).read().split()
        if queue:
            code = int(queue[0])
            open(codes, "w").write(" ".join(queue[1:]))
    with open(os.path.join(folder, "calls.jsonl"), "a") as out:
        out.write(json.dumps({{"args": rest, "code": code, "pid": os.getpid(), "start": time.time()}}) + "\\n")
if rest[0] in os.environ.get("FAKE_SLOW", "").split():
    time.sleep(float(os.environ.get("FAKE_SECONDS", "30")))
with open(os.path.join(folder, "ends.jsonl"), "a") as out:
    out.write(json.dumps({{"pid": os.getpid(), "end": time.time()}}) + "\\n")
sys.exit(code)
"""


def _run(tmp_path, *, codes=None, background=False, **env):
    fake = tmp_path / "seqrec-eval"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    for command, queue in (codes or {}).items():
        (tmp_path / f"{command}.codes").write_text(queue)
    environment = {**os.environ, "FAKE_DIR": str(tmp_path), "SEQREC_EVAL": str(fake), "WORK": str(tmp_path / "w"),
                   "DATA_DIR": str(tmp_path / "d"), "GPUS": "cuda:0 cuda:1", "SWEEPS": SWEEPS, **env}
    if background:
        return subprocess.Popen(["bash", str(SCRIPT)], env=environment, stdout=subprocess.PIPE, text=True)
    return subprocess.run(["bash", str(SCRIPT)], env=environment, capture_output=True, text=True, timeout=120)


def _calls(tmp_path):
    path, ends = tmp_path / "calls.jsonl", tmp_path / "ends.jsonl"
    end = {e["pid"]: e["end"] for e in map(json.loads, ends.read_text().splitlines())} if ends.exists() else {}
    calls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [{**call, "end": end.get(call["pid"])} for call in calls]


def _index(calls, command, *extra):
    return [i for i, call in enumerate(calls) if call["args"][0] == command and all(e in call["args"] for e in extra)]


def test_the_suite_runs_in_order_with_every_step_shared(tmp_path):
    done = _run(tmp_path, FAKE_SLOW="analyse", FAKE_SECONDS="0.5")  # the analyses beside the GPU steps take a while
    assert done.returncode == 0, done.stdout + done.stderr
    calls = _calls(tmp_path)
    # each shared step: one process per GPU device, the GPU models; one CPU process, EASE and popularity
    for step, extra in (("search", ()), ("final", ()), ("ablate", ("history_length_inference",)),
                        ("ablate", ("shuffle",))):
        group = [calls[i]["args"] for i in _index(calls, step, *extra)]
        devices = sorted(args[args.index("--device") + 1] for args in group)
        assert devices == ["cpu", "cuda:0", "cuda:1"], (step, extra)
        for args in group:
            models = args[args.index("--model") + 1:args.index("--device")]
            assert models == (["popularity", "ease"] if "cpu" in args else ["elsa", "gru", "sasrec", "bert4rec"])
    # the order: prepare, stage 1, its report, the sweeps in order, latency alone, then the reports
    last = lambda command, *e: max(_index(calls, command, *e))  # noqa: E731
    first = lambda command, *e: min(_index(calls, command, *e))  # noqa: E731
    assert last("prepare") < first("search") and last("search") < first("final")
    assert first("analyse", "none") < last("final")  # the full data's analysis beside stage 1
    assert last("analyse", "none") < first("report") and last("final") < first("report")
    assert last("ablate", "history_length_inference") < first("ablate", "shuffle")
    assert first("latency") > max(last("ablate", "shuffle"), last("analyse", "shuffle"))
    # latency starts alone, once every analysis beside the sweeps has finished; the stage-1 report waits for the
    # full data's
    starts = lambda command: [calls[i]["start"] for i in _index(calls, command)]  # noqa: E731
    assert min(starts("latency")) >= max(call["end"] for call in calls if call["args"][0] == "analyse")
    assert min(starts("report")) >= calls[first("analyse", "none")]["end"]
    assert first("latency") < first("ablation-report") < first("status") == len(calls) - 1
    assert "done: reports in" in done.stdout


def test_a_step_with_work_left_over_is_repeated(tmp_path):
    done = _run(tmp_path, codes={"search": "3"})  # one process found a run held by another
    assert done.returncode == 0, done.stdout
    assert len(_index(_calls(tmp_path), "search")) == 6  # two rounds of three processes
    assert "work left over" in done.stdout


def test_work_still_left_after_every_round_stops_the_suite(tmp_path):
    done = _run(tmp_path, codes={"final": "3 3 3 3 3 3"}, ROUNDS="2")
    assert done.returncode == 3
    assert "work still left after 2 rounds" in done.stdout
    assert not _index(_calls(tmp_path), "ablate")


def test_a_failed_run_stops_the_suite_before_the_next_step(tmp_path):
    done = _run(tmp_path, codes={"final": "1"})
    assert done.returncode == 1
    assert "stopped with exit code 1" in done.stdout and "final-r1-*.log" in done.stdout
    calls = _calls(tmp_path)
    assert _index(calls, "final") and not _index(calls, "ablate") and not _index(calls, "report")


def test_a_stop_reaches_every_process_it_started(tmp_path):
    running = _run(tmp_path, background=True, FAKE_SLOW="search analyse")
    try:
        deadline = time.time() + 20
        while len(_index(_calls(tmp_path), "search")) < 3 and time.time() < deadline:
            time.sleep(0.1)  # prepare and plan first, then three search processes and the analysis beside them
        assert len(_index(_calls(tmp_path), "search")) == 3
        running.send_signal(signal.SIGTERM)
        assert running.wait(timeout=10) == 130
    finally:
        if running.poll() is None:
            running.kill()
    for call in _calls(tmp_path):
        with pytest.raises(ProcessLookupError):
            os.kill(call["pid"], 0)  # gone, none left behind


def test_settings_are_checked_first(tmp_path):
    done = _run(tmp_path, WORK="", SEQREC_EVAL_WORK="")
    assert done.returncode == 2 and "set WORK" in done.stderr
    assert not _calls(tmp_path)


def test_stage1_alone_runs_no_sweep(tmp_path):
    # the ablation datasets are chosen after stage 1 has run (review-plan/plans/week-budget.md, B4)
    done = _run(tmp_path, SWEEPS="none")
    assert done.returncode == 0, done.stdout + done.stderr
    calls = _calls(tmp_path)
    assert not _index(calls, "ablate") and not _index(calls, "ablation-report")
    assert [call["args"][1:] for call in calls if call["args"][0] == "analyse"] == [["--sweep", "none"]]
    assert _index(calls, "final") and _index(calls, "latency") and calls[-1]["args"][0] == "status"
