"""The gap plot, and the order it lays the levels out in."""

from __future__ import annotations

import pytest

from seqrec_eval.ablation_report import level_order
from seqrec_eval.protocol import load_protocol
from test_ablations import PROTOCOL


def test_levels_follow_the_knee_axis_with_the_full_data_last(tmp_path):
    path = tmp_path / "protocol.toml"
    path.write_text(PROTOCOL)
    protocol = load_protocol(path)
    assert level_order(protocol, "history") == ["1", "2", "5", "full"]
    assert level_order(protocol, "repeats") == ["1.0", "full"]  # more removal is further from the full data
    assert level_order(protocol, "shuffle") == ["3", "all", "full"]  # no axis: the protocol's order


def test_the_gap_plot_renders_a_png():
    pytest.importorskip("matplotlib")
    from seqrec_eval.plots import gap_figure

    rows = [{"dataset": dataset, "condition": condition, "model": model, "gap": gap,
             "ci_low": gap - 0.01, "ci_high": gap + 0.01}
            for dataset in ("a", "b") for model, base in (("gru", 0.01), ("sasrec", 0.012))
            for condition, gap in zip(("2", "5", "full"), (base, base + 0.005, base + 0.006))]
    png = gap_figure(rows, sweep="history", order=["2", "5", "full"], candidates=["gru", "sasrec"], metric="ndcg@10")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_no_rows_no_plot():
    from seqrec_eval.plots import gap_figure

    assert gap_figure([], sweep="history", order=["full"], candidates=["gru"], metric="ndcg@10") is None
