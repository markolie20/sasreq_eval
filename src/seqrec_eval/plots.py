"""The gap plot of an ablation sweep: each sequential model minus the comparator (ELSA), per level.

One panel per dataset, each on its own y-axis: datasets differ in absolute
NDCG, and a shared axis would flatten the small ones. The x-axis lists the
levels in the sweep's order -- along the knee's axis where it has one, with
the full data at the end -- evenly spaced, since the levels are chosen, not
measured. Each sequential model is a line with its seed-aware 95% t interval
as a band, and a dashed zero line marks "no better than the comparator" (or,
in a sweep without it, than the best non-sequential model at that level). The
floor -- the strongest non-learned baseline at that level -- is drawn on the
same scale as a grey dashed line: a model below it has not beaten the floor,
whatever its gap."""

from __future__ import annotations

import io
import math
from typing import Any

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE, INK, INK_SECONDARY, MUTED, GRID, BASELINE = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                                      "#c3c2b7")


def available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _label_ends(figure, axis, ends: list[tuple[str, float, float]], spacing: float = 12.0) -> None:
    """Name each line at its last point, pushed apart vertically so no two labels overlap."""
    if not ends:
        return
    figure.canvas.draw()  # fix the axis limits, so data and screen positions agree
    to_screen, to_data = axis.transData, axis.transData.inverted()
    placed = sorted(((to_screen.transform((x, y))[1], name, x, y) for name, x, y in ends))
    heights = []
    for screen_y, *_ in placed:
        heights.append(max(screen_y, heights[-1] + spacing) if heights else screen_y)
    shift = (heights[-1] - placed[-1][0]) / 2.0  # centre the stack on the lines rather than pushing it all up
    for height, (_, name, x, y) in zip(heights, placed):
        screen_x = to_screen.transform((x, y))[0]
        label_y = to_data.transform((screen_x, height - shift))[1]
        axis.text(x + 0.12, label_y, name, va="center", fontsize=8, color=INK_SECONDARY, clip_on=False)


def gap_figure(gap_rows: list[dict[str, Any]], *, sweep: str, order: list[str], candidates: list[str],
               metric: str, floor_rows: list[dict[str, Any]] | None = None,
               against: str = "best non-sequential") -> bytes | None:
    """A PNG of ``gap_rows``, or ``None`` when there is nothing to plot or no matplotlib."""
    if not gap_rows or not available():
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(row["dataset"] for row in gap_rows))
    columns = min(len(datasets), 3)
    rows = math.ceil(len(datasets) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(max(4.2 * columns, 7.0), 3.2 * rows + 0.8), squeeze=False,
                                facecolor=SURFACE)
    colour = {model: SERIES[i % len(SERIES)] for i, model in enumerate(candidates)}
    x = {label: i for i, label in enumerate(order)}

    for axis in axes.flat[len(datasets):]:
        axis.set_visible(False)
    for axis, dataset in zip(axes.flat, datasets):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(BASELINE)
        axis.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.axhline(0.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ends = []
        for model in candidates:
            points = sorted((x[r["condition"]], r) for r in gap_rows
                            if r["dataset"] == dataset and r["model"] == model and r["condition"] in x)
            if not points:
                continue
            xs = [p for p, _ in points]
            axis.fill_between(xs, [r["ci_low"] for _, r in points], [r["ci_high"] for _, r in points],
                              color=colour[model], alpha=0.16, linewidth=0)
            axis.plot(xs, [r["gap"] for _, r in points], color=colour[model], linewidth=1.5, marker="o",
                      markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=model)
            # a level scored on too few users is descriptive only: drawn hollow
            low = [(p, r["gap"]) for p, r in points if r.get("below_min_users")]
            if low:
                axis.plot([p for p, _ in low], [g for _, g in low], linestyle="none", marker="o", markersize=5,
                          markerfacecolor=SURFACE, markeredgecolor=colour[model], markeredgewidth=1.5)
            ends.append((model, points[-1][0], points[-1][1]["gap"]))
        floor = sorted((x[r["condition"]], r["gap"]) for r in floor_rows or []
                       if r["dataset"] == dataset and r["condition"] in x)
        if floor:
            axis.plot([p for p, _ in floor], [g for _, g in floor], color=INK_SECONDARY, linewidth=1.2,
                      linestyle=(0, (2, 2)), marker="s", markersize=4, markeredgecolor=SURFACE, label="floor")
            ends.append(("floor", floor[-1][0], floor[-1][1]))
        axis.set_xticks(range(len(order)), order, rotation=0 if len(order) <= 6 else 45)
        axis.set_xlim(-0.4, len(order) - 0.4 + 0.6)
        _label_ends(figure, axis, ends)
        axis.set_title(dataset, loc="left", fontsize=10, color=INK)
        axis.set_xlabel("level", fontsize=8, color=MUTED)
        axis.set_ylabel(f"gap in {metric}", fontsize=8, color=MUTED)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle(f"{sweep}: sequential − {against}, seed-aware 95% t interval "
                    "(hollow: too few users, descriptive)",
                    x=0.01, y=0.99, ha="left", fontsize=11, color=INK)
    top = 0.93
    if len(handles) >= 2:  # one series needs no legend: the title names it
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=len(handles),
                      frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
        top = 0.88
    figure.tight_layout(rect=(0, 0, 1, top))
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return buffer.getvalue()
