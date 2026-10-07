"""Models from other installed packages, and the extra protocol files that hold their sections (DECISIONS §39).

A plugin here is a real package in a temporary directory on ``sys.path``, with the ``*.dist-info`` metadata an
install writes, so the entry point is found the way an installed plugin's is.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from seqrec_eval import cli, models
from seqrec_eval.models import model_provenance, model_spec
from seqrec_eval.protocol import ProtocolError, load_protocol
from seqrec_eval.splits import _source_hash, code_provenance
from test_smoke import DEVICE, PROTOCOL, workspace  # noqa: F401  (the shared synthetic workspace)

PLUGIN = '''
from compresso_recsys.models import PopularityBaseline, PopularityBaselineConfig

from seqrec_eval.models import register


class Pop(PopularityBaseline):
    provenance_packages = ("fakelib",)
    calls = 0

    def predict_on_batch(self, source, **kwargs):
        self.calls += 1
        return super().predict_on_batch(source, **kwargs)

    def run_diagnostics(self):
        return {"predict_calls": self.calls}


@register("fakepop", family="matrix", cls=Pop)
def _build(params, *, n_items, device, seed):
    return Pop(PopularityBaselineConfig(**params))
'''

EXTRA = '[models.fakepop]\nfamily = "matrix"\n'


@pytest.fixture
def isolated(monkeypatch):
    """A copy of the registry and the plugin errors, restored afterwards, and the fake modules forgotten."""
    monkeypatch.setattr(models, "REGISTRY", dict(models.REGISTRY))
    errors = dict(models.PLUGIN_ERRORS)
    yield monkeypatch
    models.PLUGIN_ERRORS.clear()
    models.PLUGIN_ERRORS.update(errors)
    for name in ("fakeplug", "fakelib", "brokenplug"):
        sys.modules.pop(name, None)


def _install(root, monkeypatch, name: str, source: str, *, entry: str | None = None) -> None:
    """``name`` as an installed distribution: the module and its ``*.dist-info`` with one model entry point."""
    (root / name).mkdir(parents=True)
    (root / name / "__init__.py").write_text(source)
    if entry is not None:
        info = root / f"{name}-0.1.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1\n")
        (info / "entry_points.txt").write_text(f"[{models.PLUGIN_GROUP}]\n{name} = {entry}\n")
    monkeypatch.syspath_prepend(str(root))


def _fake_plugin(root, monkeypatch) -> None:
    _install(root, monkeypatch, "fakeplug", PLUGIN, entry="fakeplug")
    _install(root, monkeypatch, "fakelib", "VALUE = 1\n")
    models._load_plugins()


def test_a_plugin_registers_its_model_through_its_entry_point(tmp_path, isolated):
    assert "fakepop" not in models.REGISTRY
    _fake_plugin(tmp_path, isolated)
    spec = model_spec("fakepop")
    assert (spec.family, spec.cls.__name__) == ("matrix", "Pop")
    # the plugin's code and the library it names are recorded; the suite's own models add nothing
    import fakelib
    import fakeplug
    assert model_provenance("fakepop") == {"plugins": {
        "fakeplug": {"version": "0.1", "code_sha256": _source_hash(fakeplug), "path": str(tmp_path / "fakeplug")},
        "fakelib": {"version": None, "code_sha256": _source_hash(fakelib), "path": str(tmp_path / "fakelib")}}}
    assert model_provenance("ease") == model_provenance("sasrec") == {}
    # and every process finds it as the registry loads, with nothing but the install
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")])}
    found = subprocess.run([sys.executable, "-c", "from seqrec_eval.models import model_spec; "
                            "print(model_spec('fakepop').cls.__module__)"],
                           env=env, capture_output=True, text=True, check=True)
    assert found.stdout.strip() == "fakeplug"


def test_a_plugin_that_fails_to_load_is_named_and_leaves_the_suite_usable(tmp_path, isolated, capsys):
    _install(tmp_path, isolated, "brokenplug", 'raise RuntimeError("its library is not installed")\n',
             entry="brokenplug")
    models._load_plugins()
    assert "plugin 'brokenplug' (brokenplug) failed to load: RuntimeError: its library is not installed" \
        in capsys.readouterr().err
    assert model_spec("ease").name == "ease"
    with pytest.raises(KeyError, match="plugin 'brokenplug' failed to load: RuntimeError"):
        model_spec("brokenpop")
    (tmp_path / "protocol.toml").write_text(PROTOCOL)
    assert cli.main(["--protocol", str(tmp_path / "protocol.toml"), "--work-dir", str(tmp_path / "work"),
                     "plan"]) == 0
    assert "⚠ model plugin 'brokenplug' failed to load" in capsys.readouterr().out


def _base_and_extra(tmp_path, extra: str = EXTRA):
    (tmp_path / "protocol.toml").write_text(PROTOCOL)
    (tmp_path / "extra.toml").write_text(extra)
    return tmp_path / "protocol.toml", tmp_path / "extra.toml"


def test_an_extra_protocol_adds_its_models_and_moves_no_other(tmp_path):
    base, extra = _base_and_extra(tmp_path, EXTRA + '[models.fakepop.fixed]\ncount = "users"\n')
    alone, added = load_protocol(base), load_protocol(base, extra=[extra])
    assert list(added.models) == [*alone.models, "fakepop"]
    assert added.model("fakepop").raw == {"family": "matrix", "fixed": {"count": "users"}}
    assert added.extras == (extra,) and alone.extras == ()
    for model in alone.models:
        assert added.run_fingerprint("synth", model) == alone.run_fingerprint("synth", model)
    # the model's own fingerprint is its section's, wherever the section is written
    (tmp_path / "inline.toml").write_text(PROTOCOL + "\n" + extra.read_text())
    assert load_protocol(tmp_path / "inline.toml").run_fingerprint("synth", "fakepop") == \
        added.run_fingerprint("synth", "fakepop")


@pytest.mark.parametrize("extra, message", [
    ('[protocol]\nseeds = [5]\n' + EXTRA, r"may only add \[models.\*\] sections; it also has \['protocol'\]"),
    ('[datasets.other]\nbuilder = "x"\n', r"it also has \['datasets'\]"),
    ('[models.ease]\nfamily = "matrix"\n', r"defines \[models.ease\], which the protocol already defines"),
])
def test_an_extra_protocol_may_only_add_models(tmp_path, extra, message):
    base, path = _base_and_extra(tmp_path, extra)
    with pytest.raises(ProtocolError, match=message):
        load_protocol(base, extra=[path])


def test_two_extras_may_not_define_the_same_model(tmp_path):
    base, first = _base_and_extra(tmp_path)
    (second := tmp_path / "second.toml").write_text(EXTRA)
    with pytest.raises(ProtocolError, match=f"defines \\[models.fakepop\\], which {first} already defines"):
        load_protocol(base, extra=[first, second])


def test_the_extra_protocols_come_from_the_option_or_else_the_environment(tmp_path, monkeypatch, capsys):
    base, extra = _base_and_extra(tmp_path)
    common = ["--protocol", str(base), "--work-dir", str(tmp_path / "work")]
    assert cli._extras(None) == []
    monkeypatch.setenv(cli.EXTRA_ENV, f"{extra}:{tmp_path / 'second.toml'}")
    assert cli._extras(None) == [str(extra), str(tmp_path / "second.toml")]
    assert cli._extras(["given.toml"]) == ["given.toml"]  # the option replaces the environment
    monkeypatch.setenv(cli.EXTRA_ENV, str(extra))
    assert cli.main(common + ["plan"]) == 0
    out = capsys.readouterr().out
    assert f"with the models of {extra}" in out and "fakepop" in out
    monkeypatch.delenv(cli.EXTRA_ENV)
    assert cli.main(common + ["plan"]) == 0
    assert "fakepop" not in capsys.readouterr().out


def test_a_plugin_model_runs_through_the_suite_and_its_code_is_recorded(workspace, tmp_path, isolated):
    root, _ = workspace
    work = tmp_path / "work"
    shutil.copytree(root / "work", work)
    _fake_plugin(tmp_path / "site", isolated)
    (extra := tmp_path / "private.toml").write_text(EXTRA)
    common = ["--protocol", str(root / "protocol.toml"), "--work-dir", str(work), "--protocol-extra", str(extra)]
    for step in (["search"], ["final"], ["diversity"]):
        assert cli.main(common + step + ["--model", "fakepop", "--device", DEVICE]) == 0
    assert cli.main(common + ["report", "--reference", "elsa"]) == 0

    protocol = load_protocol(root / "protocol.toml", extra=[extra])
    finals = sorted(work.glob("runs/synth/fakepop/*/final-seed*/done.json"))
    assert len(finals) == len(protocol.seeds)
    import fakeplug
    for path in finals:
        code = json.loads(path.read_text())["code"]
        assert {key: code[key] for key in ("library", "suite")} == code_provenance()
        assert code["plugins"]["fakeplug"]["code_sha256"] == _source_hash(fakeplug)
        assert json.loads(path.read_text())["model_diagnostics"]["predict_calls"] > 0
        assert (path.parent / "diversity.json").exists()  # reloaded through the plugin's class
    report = (work / "reports" / "stage1.md").read_text()
    assert f"with the models of `{extra.resolve()}`" in report
    assert f"Model plugin: fakeplug 0.1 (source {_source_hash(fakeplug)}), for all " in report
    assert "| fakepop |" in report
    # the suite's models are one build still: the plugin's runs do not split them
    assert "different code builds" not in report
