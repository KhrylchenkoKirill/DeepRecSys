from __future__ import annotations
from typing import Dict, List, Callable, Any

import numpy as np
import polars as pl


class Constants:
    temporal_threshold: int = 25395195
    train_size: int = 7755988
    test_size: int = 254174
    num_test_users: int = 37446


def check_data_split(
    train: pl.DataFrame,
    test: pl.DataFrame,
    embeddings: pl.DataFrame,
    artists: pl.DataFrame,
    test_targets: Dict[int, List[int]],
) -> None:
    # required columns
    for col in ["uid", "item_id", "timestamp"]:
        assert col in train.columns, f"train missing column {col}"
        assert col in test.columns, f"test missing column {col}"

    assert embeddings.height > 0, "embeddings is empty"
    assert "item_id" in embeddings.columns, "embeddings must have item_id"
    assert "embed" in embeddings.columns, "embeddings must have embed"
    assert artists.height > 0, "artists is empty"
    assert "item_id" in artists.columns, "artists must have item_id"
    assert "artist_id" in artists.columns, "artists must have artist_id"

    # test users in train
    train_users = set(train["uid"].unique().to_list())
    test_users = set(test["uid"].unique().to_list())
    assert test_users.issubset(train_users), "test contains users absent in train"

    # targets cover test users
    assert len(test_targets) > 0, "test_targets is empty"
    assert set(test_targets.keys()) == test_users, "test_targets keys must match test users"

    # items in train have embeddings (after filtering)
    emb_items = set(embeddings["item_id"].unique().to_list())
    train_items = set(train["item_id"].unique().to_list())
    assert train_items.issubset(emb_items), "Some train items missing embeddings after filtering"

    # timestamp ordering (train < test threshold) – soft check
    assert train["timestamp"].max() <= test["timestamp"].max(), "Unexpected timestamps"

    assert train["timestamp"].max() < Constants.temporal_threshold, "Train-test temporal split is incorrect"
    assert train.height == Constants.train_size, "Train size is incorrect"
    assert test.height == Constants.test_size, "Test size is incorrect"
    assert len(test_targets) == Constants.num_test_users, "Test targets size is incorrect"
    print('All good! :)')


def check_metrics(
    get_metrics: Callable[[List[int], List[int], int], Dict[str, float]],
    evaluate: Callable[[Dict[int, List[int]], Dict[int, List[int]], int, int], Dict[str, float]],
) -> None:
    # toy example
    targets = [1, 2, 3]
    cands = [10, 2, 30, 1]
    m = get_metrics(targets, cands, topk=4)
    assert 0.0 <= m["hitrate"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["ndcg"] <= 1.0
    assert np.isclose(m["hitrate"], 1.0) and np.isclose(m["recall"], 2. / 3) and np.isclose(m["ndcg"], 0.49818925746641285), \
    "'get_metrics' outputs incorrect values"

    targets_by_user = {7: [1, 2], 8: [9]}
    cands_by_user = {7: [1, 3, 4], 8: [10, 9, 11]}
    out = evaluate(targets_by_user, cands_by_user, catalog_size=100, topk=3)
    for k in ["hitrate", "recall", "ndcg", "coverage"]:
        assert k in out, f"missing metric {k}"
    assert (
        np.isclose(out["hitrate"], 1.0) 
        and np.isclose(out["recall"], 0.75) 
        and np.isclose(out["ndcg"], 0.622038473168458) 
        and np.isclose(out["coverage"], 0.06)
     ), "'evaluate' outputs incorrect values"
    print('All good! :)')
    

def check_all_metrics(metrics, hitrate, recall, ndcg, coverage):
    assert np.isclose(metrics["recall"], recall), "Incorrect recall value"
    assert np.isclose(metrics["hitrate"], hitrate), "Incorrect hitrate value"
    assert np.isclose(metrics["ndcg"], ndcg), "Incorrect ndcg value"
    assert np.isclose(metrics["coverage"], coverage), "Incorrect coverage value"


def check_all_metrics_geq(metrics, hitrate, recall, ndcg, coverage):
    assert metrics["recall"] >= recall, "Too low recall value"
    assert metrics["hitrate"] >= hitrate, "Too low hitrate value"
    assert metrics["ndcg"] >= ndcg, "Too low ndcg value"
    assert metrics["coverage"] >= coverage, "Too low coverage value"


def check_top_pop(metrics):
    check_all_metrics(
        metrics,
        hitrate=0.11237515355445174,
        recall=0.03036588814350686,
        ndcg=0.010997994080361367,
        coverage=0.0006363063687904452
    )
    print('All good! :)')


def check_artist_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.2,
        recall=0.05,
        ndcg=0.02,
        coverage=0.58
    )
    print('All good! :)')


def check_i2i_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.12,
        recall=0.03,
        ndcg=0.01,
        coverage=0.9
    )
    print('All good! :)')


def check_w2v_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.18,
        recall=0.04,
        ndcg=0.017,
        coverage=0.7
    )


def check_cf_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.17,
        recall=0.05,
        ndcg=0.02,
        coverage=0.65
    )
    print('All good! :)')


def check_tfidf_recs(metrics):
    check_all_metrics_geq(
        metrics,
        hitrate=0.26,
        recall=0.08,
        ndcg=0.03,
        coverage=0.78
    )
    print('All good! :)')


def check_als_recs(raw_metrics, tfidf_metrics):
    check_all_metrics_geq(
        raw_metrics,
        hitrate=0.26,
        recall=0.075,
        ndcg=0.028,
        coverage=0.05
    )
    check_all_metrics_geq(
        tfidf_metrics,
        hitrate=0.29,
        recall=0.09,
        ndcg=0.03,
        coverage=0.1
    )
    print('All good! :)')
