"""How diverse the final models' recommendations are: catalogue coverage and intra-list diversity.

Research design §3.5 and §5.3. Diagnostics only: nothing is selected on them. They exist so that a gain in
accuracy bought by narrowing what is recommended is visible rather than silent.

Each stage-1 final model is scored once more on its test users, exactly as its metrics were -- the same users,
the same exclusion of seen items -- through :func:`~seqrec_eval.evaluate.evaluate_phase`, which hands back the
ranked lists it scored. From those:

- **Catalogue coverage@K**: the share of the items a final model can recommend (the training catalogue it was
  fitted on) that appear in at least one user's top K. Exact.
- **Intra-list diversity@K**: per user, the mean dissimilarity 1 − cos(i, j) over the pairs of items in the top
  K, averaged over users (Ziegler et al., 2005; Vargas & Castells, 2011). Two items are similar when the same
  users interacted with them in the training data the finals were fitted on: co-occurrence, chosen
  2026-10-06 because not every dataset has item attributes (OTTO has none) and those that exist differ in kind
  between datasets. The exact cosine of two items' user columns does not scale to catalogues of 100,000 items
  and more, so it is taken on the best rank-``RANK`` approximation of the binarised training matrix: the cosine
  of the approximation's columns, ``S v_i`` against ``S v_j``. An item with no training interaction is
  dissimilar to every other.

- **Novelty@K** (DECISIONS §54): the mean self-information of the recommended items, ``−log2(p)`` with ``p`` the
  share of training users who had the item (Zhou et al., 2010): higher means less popular items.
- **Long-tail share@K** (DECISIONS §54): the share of recommended items outside the head, the most popular items
  that together hold :data:`HEAD_SHARE` of the training interactions.

The same pass scores each final on every new item its users went on to have in the test window
(``test_window_new``, :func:`~seqrec_eval.analysis.discovery_views`), which the stage-1 report compares.

Written per final run as ``diversity.json``; :func:`diversity_table` summarises them for the stage-1 report.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from .analysis import discovery_views, score_view
from .evaluate import evaluate_phase
from .models import model_spec
from .protocol import Protocol
from .results import evaluation_exists, read_json, save_evaluation, write_json
from .runner import final_seeds, run_root, stale_final_seeds
from .splits import Split

#: the record's format and method; a change to either re-measures every final. 2 (2026-10-20, DECISIONS §54):
#: novelty and the long-tail share, and the finals scored on later new items
DIVERSITY_VERSION = 2
#: rank of the factorisation the item similarity is taken on
RANK = 64
#: the share of the training interactions the head holds; every other item is the long tail (DECISIONS §54)
HEAD_SHARE = 0.8


def item_users(training: csr_matrix) -> np.ndarray:
    """How many training users had each item: the column counts of the binarised training matrix."""
    binary = csr_matrix(training, copy=True)
    binary.sum_duplicates()
    binary.eliminate_zeros()
    return np.diff(binary.tocsc().indptr).astype(np.float64)


def tail_items(users: np.ndarray, head_share: float = HEAD_SHARE) -> np.ndarray:
    """Per item, whether it is in the long tail: outside the most popular items that together hold ``head_share``
    of the interactions (ties in training users broken by the lower index, so the split is fixed)."""
    order = np.lexsort((np.arange(users.size), -users))
    held = np.cumsum(users[order]) / users.sum() if users.sum() else np.ones(users.size)
    head = order[:int(np.searchsorted(held, head_share)) + 1]
    tail = np.ones(users.size, dtype=bool)
    tail[head] = False
    return tail


def item_vectors(training: csr_matrix, rank: int = RANK) -> np.ndarray:
    """Unit item vectors whose dot products are the cosines of the columns of the best rank-``rank``
    approximation of the binarised ``training`` matrix (users × items); zero for an item no user had."""
    binary = csr_matrix(training, dtype=np.float64, copy=True)
    binary.data[:] = 1.0
    rank = min(rank, min(binary.shape) - 1)
    if rank < 1:
        raise ValueError(f"a training matrix of shape {binary.shape} has no factorisation to measure on")
    _, singular, vt = svds(binary, k=rank, random_state=0)
    vectors = (vt * singular[:, None]).T  # items × rank: S v_i
    norms = np.linalg.norm(vectors, axis=1)
    return np.divide(vectors, norms[:, None], out=np.zeros_like(vectors), where=norms[:, None] > 1e-12)


def list_diversity(lists: np.ndarray, vectors: np.ndarray, cutoffs, users: np.ndarray | None = None,
                   n_users: int | None = None) -> dict[str, dict[str, float]]:
    """Coverage and intra-list diversity of ranked ``lists`` (users × K item indices) at each cutoff; given
    ``users`` (:func:`item_users` of the training matrix) and ``n_users`` (its rows), novelty and the long-tail
    share too.

    Items past the end of ``vectors`` (none a final model can recommend) count as uncovered and dissimilar, and as
    tail items no training user had (as novel as an item is counted: one user of ``n_users``).
    """
    n_items = vectors.shape[0]
    padded = np.vstack([vectors, np.zeros((1, vectors.shape[1]))])  # the row for any index out of range
    if users is not None:
        if not n_users or users.size != n_items:
            raise ValueError(f"novelty needs every item's training users ({n_items}) and the number of users")
        # an item no training user had counts as had by one, so its information is finite and the largest
        information = np.append(-np.log2(np.maximum(users, 1.0) / n_users), -np.log2(1.0 / n_users))
        tail = np.append(tail_items(users), True)
    out = {}
    for k in sorted(cutoffs):
        top = lists[:, :k]
        inside = top[(top >= 0) & (top < n_items)]
        coverage = float(np.unique(inside).size / n_items)
        if k < 2 or top.shape[0] == 0:
            ild = float("nan")
        else:
            safe = np.where((top >= 0) & (top < n_items), top, n_items)
            similarity, n_lists = 0.0, top.shape[0]
            for start in range(0, n_lists, 4096):
                rows = padded[safe[start:start + 4096]]  # users × k × rank
                total = rows.sum(axis=1)
                # Σ over ordered pairs i ≠ j of cos(i, j) = ‖Σ_i u_i‖² − Σ_i ‖u_i‖²
                pairs = np.einsum("ur,ur->u", total, total) - np.einsum("ukr,ukr->u", rows, rows)
                similarity += float(pairs.sum())
            ild = 1.0 - similarity / (n_lists * k * (k - 1))
        out[str(k)] = {"coverage": coverage, "intra_list_diversity": ild}
        if users is not None:
            safe = np.where((top >= 0) & (top < n_items), top, n_items)
            empty = top.size == 0
            out[str(k)]["novelty"] = float("nan") if empty else float(information[safe].mean())
            out[str(k)]["tail_share"] = float("nan") if empty else float(tail[safe].mean())
    return out


def diversity_path(directory: Path) -> Path:
    return Path(directory) / "diversity.json"


def measure_final(protocol: Protocol, split: Split, model: str, directory: Path, *, device: str,
                  vectors: np.ndarray, users: np.ndarray | None = None) -> dict[str, Any]:
    """Score one saved final model on test again, keep its lists, and measure them against ``vectors``
    (:func:`item_vectors` of the split's training matrix) and ``users`` (:func:`item_users` of it); then score it on
    every view of new items it has no result for yet (:func:`~seqrec_eval.analysis.discovery_views`)."""
    registered = model_spec(model)
    trainer = registered.cls.load(Path(directory) / "model.zip", device=device)
    lists: list[np.ndarray] = []
    started = time.perf_counter()
    result = evaluate_phase(trainer, split, "test", family=registered.family, protocol=protocol,
                            exclude_seen=protocol.dataset(split.dataset).exclude_seen, rows=split.test_rows,
                            lists=lists)
    ranked = np.vstack(lists) if lists else np.zeros((0, max(protocol.cutoffs)), dtype=np.int64)
    for targets, view in discovery_views(protocol, split.dataset):
        if not evaluation_exists(Path(directory) / view):
            save_evaluation(score_view(protocol, split, trainer, targets, family=registered.family),
                            Path(directory) / view)
    training = split.data["x_train"]
    users = item_users(training) if users is None else users
    return {
        "version": DIVERSITY_VERSION,
        "basis": f"cosine of the columns of the rank-{min(RANK, min(training.shape) - 1)} "
                 "approximation of the binarised training interactions",
        "catalogue": int(vectors.shape[0]),
        "head_share": HEAD_SHARE,
        "n_users": int(ranked.shape[0]),
        "n_scored_rows": int(result.n_scored_rows),
        "cutoffs": list_diversity(ranked, vectors, protocol.cutoffs, users=users, n_users=int(training.shape[0])),
        "seconds": time.perf_counter() - started,
    }


def measure(protocol: Protocol, work_dir: Path, split: Split, model: str, *, device: str, force: bool = False,
            log=print) -> int:
    """Measure every finished, current final of ``model`` on ``split`` (a final split) that has no record yet.
    None of a latency-only model, whose weights mean nothing (DECISIONS §48)."""
    if not protocol.model(model).scored_on(split.dataset):
        return 0
    root = run_root(work_dir, split.dataset, model, protocol.run_fingerprint(split.dataset, model))
    stale = stale_final_seeds(protocol, work_dir, split.dataset, model)
    vectors, users = None, None  # factorised and counted once, when the first final needs them
    done = 0
    for seed in final_seeds(protocol, work_dir, split.dataset):
        directory = root / f"final-seed{seed}"
        if seed in stale or not (directory / "done.json").exists() or not (directory / "model.zip").exists():
            continue
        path = diversity_path(directory)
        if path.exists() and not force and read_json(path).get("version") == DIVERSITY_VERSION:
            continue
        if vectors is None:
            vectors, users = item_vectors(split.data["x_train"]), item_users(split.data["x_train"])
        record = measure_final(protocol, split, model, directory, device=device, vectors=vectors, users=users)
        write_json(path, record)
        primary = protocol.primary_metric.split("@")[1]
        cut = record["cutoffs"][primary]
        log(f"[{split.dataset}/{model}] final-seed{seed}: coverage@{primary} {cut['coverage']:.3f}, "
            f"intra-list diversity@{primary} {cut['intra_list_diversity']:.3f}, novelty@{primary} "
            f"{cut['novelty']:.2f}, long-tail share@{primary} {cut['tail_share']:.3f} over {record['n_users']:,} users")
        done += 1
    return done


def diversity_table(protocol: Protocol, work_dir: Path, datasets: list[str], models: list[str]) -> str:
    """Per dataset and model, coverage, intra-list diversity, novelty and the long-tail share at the primary cutoff,
    mean ± sd over seeds."""
    k = protocol.primary_metric.split("@")[1]
    names = ("coverage", "intra_list_diversity", "novelty", "tail_share")
    lines = [f"| dataset | model | seeds | coverage@{k} | intra-list diversity@{k} | novelty@{k} | "
             f"long-tail share@{k} |", "|---|---|---|---|---|---|---|"]
    for dataset in datasets:
        for model in models:
            if not protocol.model(model).scored_on(dataset):
                continue
            root = run_root(work_dir, dataset, model, protocol.run_fingerprint(dataset, model))
            stale = stale_final_seeds(protocol, work_dir, dataset, model)
            records = [read_json(diversity_path(root / f"final-seed{seed}"))
                       for seed in final_seeds(protocol, work_dir, dataset)
                       if seed not in stale and diversity_path(root / f"final-seed{seed}").exists()]
            records = [r for r in records if r.get("version") == DIVERSITY_VERSION]
            if not records:
                continue
            cells = []
            for name in names:
                values = np.array([r["cutoffs"][k][name] for r in records])
                spread = f" ± {values.std(ddof=1):.3f}" if values.size > 1 else ""
                cells.append(f"{values.mean():.3f}{spread}")
            lines.append(f"| {dataset} | {model} | {len(records)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
