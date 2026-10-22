# seqrec-eval: overview

seqrec-eval benchmarks sequential recommendation models against Recombee's production models (ELSA and SANSA) and
simple non-learned baselines. It runs on five public datasets and measures next-item accuracy, recommendation
diversity and CPU inference latency. Within-dataset ablations then tie each model's behaviour to individual data
properties: history length, item order and repeat consumption.

It is stage 1 of the internship project:
1. **Public benchmark (this suite):** which model works best, under which data conditions, at what latency.
2. **Client selection:** a Recombee client whose data falls inside the region where the chosen model performs and
   meets the latency requirement.
3. **Client evaluation:** tuning on that client's data, an offline gate, Recombee's production load test, and an
   online A/B test.

## 1. What is tested

**Main question:** under which conditions does a sequential model outperform ELSA and SANSA on next-item
prediction, while keeping model inference under 100 ms at P95?

The suite answers it with:
- **Accuracy:** each model's next-item ranking quality per dataset, compared with ELSA and with the strongest
  simple baseline (the floor), with significance tests.
- **Data dependence:** how accuracy changes when one property of the data is changed (ablations).
- **Latency:** each model's CPU inference time against history length and catalogue size.
- **Diversity:** how broad and varied each model's recommendations are.

## 2. Pipeline

The suite runs these steps in order; every step skips work that is already done.

1. **Prepare:** builds each dataset's temporal split and recovers every event's timestamp.
2. **Analyse:** profiles the data, runs the non-learned baselines and sets the floor, and measures how much
   signal item order carries.
3. **Search:** 10 random-search trials per model, scored on validation.
4. **Final:** the selected configuration, retrained on training plus validation data, run with 3 seeds and scored
   on test.
5. **Diversity:** coverage and intra-list diversity of the final models' recommendations.
6. **Ablations:** the final models retrained or rescored on transformed data.
7. **Latency:** single-request CPU timing of every final model.
8. **Reports:** stage-1 results, data analysis, one report per ablation sweep, and repeat strata.

## 3. Datasets

| dataset | domain | used | validation and test window | training users | items | median history | repeat events | already-seen items excluded | test users scored |
|---|---|---|---|---|---|---|---|---|---|
| MovieLens-20M | movie ratings (3 stars and up) | all users | 339 days | 134,138 | 21,557 | 59 | 0% | yes | 2,942 |
| Amazon Toys & Games (2023) | product reviews | 25% of users | 339 days | 123,390 | 275,514 | 7 | 0% | yes | 15,672 |
| Music4All-Onion | music listening, 2014 | all users | 30 days | 50,491 | 36,797 | 239 | 71% | no | 30,180 |
| Yambda-50M | music listening (Yandex Music) | 50% of users | 30 days | 4,364 | 134,066 | 2,506 | 72% | no | 3,840 |
| OTTO | e-commerce sessions, clicks | 2% of sessions | 2 days | 78,641 | 73,674 | 14 | 35% | no | 14,605 |

- **Interactions** are implicit: every kept event counts as one interaction.
- **Filtering:** users need at least 5 interactions. Items need at least 5 on Music4All, Yambda and OTTO, and 1 on
  MovieLens-20M and Amazon.
- **Samples** are drawn by user (or session) with every history kept whole.
- **The columns:**
  - *training users*, *items*, *median history* and *repeat events* describe the training data the final models
    are fitted on;
  - *repeat events* is the share of interactions with an item the user had before;
  - *test users scored* is the number of test users the metrics are computed over.

## 4. Evaluation protocol

**Split:** global temporal split. The last two windows of each log (the window lengths in the table above) are
validation and test; everything before is training data.

**Training data:** every user with interactions before the cutoff, not only recently active users.

**Selection and testing:**
- Search trials are trained on the training window and scored on a fixed sample of up to 20,000 validation users.
- The selected configuration is then retrained on training plus validation data and scored once on test.
- Test data is never used for selection.

**Target:** the user's next interaction after their history ends (every item at that first timestamp).
Everything the user does in the test window is reported beside it as a diagnostic.

**Users scored:** those whose next item exists in the model's training catalogue. Users whose next item is new
score zero for every model; they are counted and reported, not scored. Where items the user already had are
excluded, users whose next item is one of those are left out as well.

**Ranking:**
- Every model ranks the full catalogue; there are no sampled candidates.
- On MovieLens-20M and Amazon, items the user already had are excluded from the ranking.
- On the music datasets and OTTO they stay in, so repeats can be recommended.

**Metrics** at cutoffs 5, 10 and 20: **NDCG (primary, at 10)**, recall, hit rate, calibrated recall, precision,
MAP and MRR.

## 5. Models

**Sequential models** (read the history in order):

| model | what it is | searched over | datasets |
|---|---|---|---|
| GRU | recurrent network (GRU cell), full softmax over the catalogue | embedding and hidden size 64–512, 1–2 layers, dropout, learning rate, batch size, 3–40 epochs, history window 50–200 | all 5 |
| SASRec | causal self-attention, sampled-negative training (as published) | width 16–128, 1–3 blocks, 1–2 heads, dropout, learning rate, batch size, 10–100 epochs, 1–256 negatives, history window 25–400 | all 5 |
| BERT4Rec | bidirectional self-attention, masked-item training (as published) | width 32–128, 1–3 blocks, 1–4 heads, dropout, mask share 0.2–0.6, learning rate, batch size, 1–10 epochs (11 passes each), history window 25–400 | all 5 |
| SASRec v2 | SASRec's network trained with cross-entropy over the whole catalogue (the user's own history among the negatives), with input masking and co-occurrence augmentation | SASRec's ranges without negatives, plus mask share 0–0.3 and co-occurrence replacement and insertion 0–0.1 | all 5 |
| LRURec | linear recurrent units (Yue et al., WSDM 2024); timed for latency only: the paper's architecture (width 64, 2 blocks) at a 400-item window, not trained for accuracy | not searched | all 5 (latency) |

**Non-sequential models** (read the history as a set):

| model | what it is | searched over | datasets |
|---|---|---|---|
| ELSA | Recombee's production linear autoencoder (low-rank); **the reference every model is compared with** | factors 128–2048, batch size, learning rate, 3–40 epochs | all 5 |
| SANSA | Recombee's production sparse approximation of EASE | the settings Recombee tunes per client | all 5 |
| EASE | closed-form linear autoencoder | regularisation | MovieLens-20M, Music4All (catalogues up to 40,000 items) |
| Popularity | most popular training items | — | all 5 |

**Non-learned baselines** (searched on validation like the models):

| baseline | what it does | searched over |
|---|---|---|
| popularity | the most popular items | counting events or users |
| time-decayed popularity | popularity with each event weighted by its age | half-life from 6 hours to a year |
| replay | the user's own history | most recent or most frequent first |
| Markov | first-order transitions from the user's last item | nothing |

The strongest baseline on test is the **floor**: the level a model must exceed to show it learns more than a
simple rule.

## 6. Model selection

- **Search:** 10 random-search trials per model, the same budget for every model. Trial *i* is the same
  configuration on every dataset. Selection is on validation NDCG@10.
- **Finals:** the selected configuration, retrained on training plus validation, with seeds 0, 1 and 2. Test
  values are the mean ± standard deviation over the three seeds.
- **Extra epochs:** a selection whose training loss still fell by more than 5% over its last tenth of epochs is
  also trained at 2× and 4× its epochs, and validation picks among the three. This applied to SASRec and GRU on
  Yambda.
- **Baselines:** searched the same way; a small grid is searched in full.

## 7. Statistics

- **Per dataset:** every model against ELSA on test NDCG@10, the comparison fixed in advance.
  - **The difference:** of the per-user scores, averaged over seeds.
  - **The test:** a t-test whose standard error includes both the variation between users and the spread between
    seeds.
  - **Multiple comparisons:** Holm correction across the models of a dataset, with 95% intervals.
- **Against the floor:** every model, with the same test, in its own Holm family.
- **Across datasets:** no pooling. A table shows, per dataset, whether each model is better, not different or
  worse than ELSA.
- **For comparison:** the paired t-test over users only is reported beside it.
- **On new items too:** the same two comparisons on the next items users never had before, against ELSA and
  against a floor of the simple rules scored the same way, with a table across datasets. This matches a feed that
  never shows what the user already had.
- **Against ELSA at its best window:** ELSA (and SANSA) is also served only the last *n* events of each history, *n*
  chosen on validation, and every model is compared with it there too.

## 8. Diagnostics

Reported beside the results, never used for selection:
- **New items only:** the next items the user never had before, on the repeat-heavy datasets. It shows what a
  model adds beyond repeating the user's history.
- **Window targets:** everything in the test window, beside the next item.
- **Later new items:** every new item the user went on to have in the test window, so a list that holds several
  counts each; compared like new items, with its own floor.
- **Diversity:**
  - catalogue coverage@K: the share of the catalogue that appears in any user's top K;
  - intra-list diversity@K: the average dissimilarity between items in a list, with item similarity taken from
    co-occurrence in the training data;
  - novelty@K: how unpopular the recommended items are (their mean self-information);
  - long-tail share@K: the share of recommended items outside the most popular items that hold 80% of interactions.
- **Freshness:** each final against the same configuration trained one window earlier, and the share of next items
  that such an older model has never seen.
- **Convergence:** flags a selected run whose training loss was still falling at its last epoch.
- **All-cold histories:** the number of users whose history holds only items unseen in training.
- **Repeat strata:** accuracy by history length and by the share of repeats in the user's history.
- **Sequence signal:** a Markov model fitted on shuffled and on reversed histories, against the normal one. It
  shows how much order and direction matter in each dataset.

## 9. Ablations

Each ablation changes one property of the data, at several levels, within one dataset. Every model keeps its
stage-1 configuration, and each level is compared with the full data.

**Retraining sweeps.**
- **Where:** Music4All and OTTO, seed 0.
- **Models:** ELSA, GRU, SASRec, SANSA, popularity, BERT4Rec (OTTO only), and SASRec v2 (since 22 October, §57).
- **How:** each model is retrained on the transformed data and scored on the same test users.

| sweep | what changes | levels |
|---|---|---|
| history length | every history is cut to its last *n* interactions, in training and as input | 5, 20, 100, 500 |
| shuffle | the order within histories is randomised, the contents kept | in blocks of 2, in blocks of 10, the whole history |
| repeat removal | repeat interactions are removed with probability *p* | 50%, 100% |

**SASRec v2's parts.** On Music4All, Yambda and OTTO, seed 0: its selected configuration retrained with one part
changed at a time (no augmentation, each co-occurrence augmentation alone, masking added, half the width, one block),
each against the stage-1 model.

**Serving-time history length.**
- **Where:** all five datasets, all models but EASE, 3 seeds.
- **How:** the trained final models are not retrained; they score with only the last *n* items of each history.
- **Levels:** 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000.
- **It answers** how much history a model needs at serving time.

**How the sweeps are read:**
- **Curves:** each sweep gives a curve per model against the full data, plus the gap between each sequential
  model and ELSA.
- **The knee:** the smallest level whose score is no more than 10% below the full data's, with the test's power.
- **Manipulation check:** the data's characteristics are measured at every level, to check that the transform
  changed what it should and nothing else.
- **Re-scored at every level:** the baselines and the floor, and the new-items view.

## 10. Latency

- **What is timed:** single-request inference of every final model on the CPU, 4 threads, producing a top-20 over
  the full catalogue.
- **By history length:** 200 timed requests per bin (1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000+ items), after
  20 warm-up requests.
- **Reported:** P50, P95 and P99, per model and dataset.
- **Requirement:** model inference under 100 ms at P95.
- **Same machine:** all models are timed on one machine, so they compare directly. Recombee's production load test
  confirms the requirement in stage 3.

## 11. Outputs

- **Stage-1 report:**
  - per dataset: the data split, every model's validation and test scores, and the comparisons with ELSA and with
    the floor;
  - the cross-dataset table;
  - the latency and diversity tables;
  - all values as CSV.
- **Analysis report:** data profiles, baselines and the floor, sequence signal.
- **Ablation reports,** one per sweep: the curves, the gap to ELSA, the knees, manipulation checks and the
  new-items view.
- **Repeat-strata report.**

## 12. Reproducibility

- **One protocol file** fixes every setting: datasets, models, search spaces, seeds, metrics and sweeps.
- **Fingerprints:** every result is stored under a fingerprint of the settings that determine it, and records the
  code version that produced it.
- **Resumable:** runs can be stopped and resumed, and several processes can share the work without duplicating it.
- **Tests:** 330 automated tests on synthetic data, end to end on CPU and GPU. Each model was also checked on data
  with a planted signal whose best possible score is known.
- **Hardware:** an NVIDIA V100 server (most of stage 1) and an RTX 2060 workstation (the rest). Latency is
  measured on one machine.

## 13. Status (22 October 2026)

- **Done:**
  - stage 1 for every model on all five datasets (EASE where it fits): search, finals, epoch extensions, diversity,
    latency;
  - baselines, floor and data analysis;
  - the ablations (EASE is in none);
  - the new-items floor and tests, ELSA and SANSA at their serving window, freshness, novelty and later new items,
    SASRec v2's parts, and SASRec on Yambda at 8 times its epochs.
- **Running:** SASRec v2 in the retraining sweeps and the serving-time history sweep (§57).
- **Next:** the client selection with Recombee.
