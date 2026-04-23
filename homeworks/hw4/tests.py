import math

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_single_linear(
    module: nn.Module, in_features: int, out_features: int
) -> nn.Linear:
    matches = [
        m
        for m in module.modules()
        if isinstance(m, nn.Linear)
        and m.in_features == in_features
        and m.out_features == out_features
    ]
    assert len(matches) == 1, (
        f"Expected exactly one Linear({in_features}, {out_features}), got {len(matches)}"
    )
    return matches[0]


def _find_optional_single_gelu(module: nn.Module) -> nn.GELU | None:
    matches = [m for m in module.modules() if isinstance(m, nn.GELU)]
    if len(matches) == 0:
        return None
    assert len(matches) == 1, f"expected at most one nn.GELU, got {len(matches)}"
    return matches[0]


def _assert_mlp_matches_gelu_between_linears(
    mlp_cls,
    mlp: nn.Module,
    fc_up: nn.Linear,
    fc_down: nn.Linear,
    *,
    d_model: int,
) -> None:
    """If MLP has no nn.GELU, verify nonlinearity between the two linears is GELU (eval, no dropout)."""
    mlp0 = mlp_cls(d_model=d_model, dropout=0.0)
    mlp0.load_state_dict(mlp.state_dict())
    mlp0.eval()
    x = torch.randn(2, 3, d_model)
    with torch.no_grad():
        h = F.linear(x, fc_up.weight, fc_up.bias)
        expected = F.linear(F.gelu(h), fc_down.weight, fc_down.bias)
        got = mlp0(x)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4), (
        "Between the d_model→4d and 4d→d Linear blocks, the activation should be GELU "
        "(or expose it as a single nn.GELU())"
    )


def _find_single_modulelist(module: nn.Module) -> nn.ModuleList:
    matches = [m for m in module.children() if isinstance(m, nn.ModuleList)]
    assert len(matches) == 1, f"Expected one top-level ModuleList, got {len(matches)}"
    return matches[0]


def _find_top_level_by_type(module: nn.Module, tp):
    matches = [m for m in module.children() if isinstance(m, tp)]
    assert len(matches) == 1, (
        f"Expected one top-level {tp.__name__}, got {len(matches)}"
    )
    return matches[0]


def _find_named_child_by_predicate(module: nn.Module, predicate):
    matches = [
        (name, child) for name, child in module.named_children() if predicate(child)
    ]
    assert len(matches) == 1, f"Expected one matching child, got {len(matches)}"
    return matches[0]


def _reference_causal_self_attention(
    x: torch.Tensor,
    mask: torch.Tensor,
    qkv: nn.Linear,
    out: nn.Linear,
    n_heads: int,
) -> torch.Tensor:
    """
    Spec numerics: QKV (one Linear d→3d), head split, key-padding ∧ causal mask,
    scaled dot-product attention, out Linear. Dropout=0.
    (Matches both hand-written softmax and F.scaled_dot_product_attention for this mask.)
    """
    B, T, d_model = x.shape
    head_dim = d_model // n_heads
    qkv_t = F.linear(x, qkv.weight, qkv.bias)
    q, k, v = qkv_t.chunk(3, dim=-1)
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    key_padding = mask[:, None, None, :]
    causal = torch.ones((T, T), device=x.device, dtype=torch.bool).tril()
    am = key_padding & causal[None, None, :, :]
    if hasattr(F, "scaled_dot_product_attention"):
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=am, dropout_p=0.0, is_causal=False
        )
    else:
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(~am, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        y = probs @ v
    y = y.transpose(1, 2).contiguous().view(B, T, d_model)
    return F.linear(y, out.weight, out.bias)


def _is_attn_subtree(module: nn.Module, d_model: int) -> bool:
    has_qkv = any(
        isinstance(m, nn.Linear)
        and m.in_features == d_model
        and m.out_features == 3 * d_model
        for m in module.modules()
    )
    has_4d = any(
        isinstance(m, nn.Linear)
        and (m.in_features == 4 * d_model or m.out_features == 4 * d_model)
        for m in module.modules()
    )
    return has_qkv and not has_4d


def _is_mlp_subtree(module: nn.Module, d_model: int) -> bool:
    try:
        _find_single_linear(module, d_model, 4 * d_model)
        _find_single_linear(module, 4 * d_model, d_model)
    except AssertionError:
        return False
    return True


def _find_block_attention_and_mlp(
    block: nn.Module, d_model: int
) -> tuple[nn.Module, nn.Module]:
    """Direct children: LayerNorms / Dropout are excluded; the rest must be (attn, mlp) by weight graph."""
    non_meta = [
        c for c in block.children() if not isinstance(c, (nn.LayerNorm, nn.Dropout))
    ]
    atts = [c for c in non_meta if _is_attn_subtree(c, d_model)]
    mlps = [c for c in non_meta if _is_mlp_subtree(c, d_model)]
    assert len(atts) == 1 and len(mlps) == 1, (
        f"Block must have one attention sublayer (QKV d→3d) and one MLP; "
        f"found {len(atts)} attn-like, {len(mlps)} mlp-like among non-LN non-Dropout children"
    )
    return atts[0], mlps[0]


def _block_dropouts_in_order(
    block: nn.Module,
) -> tuple[nn.Dropout, nn.Dropout]:
    douts = [c for c in block.children() if isinstance(c, nn.Dropout)]
    if len(douts) == 1:
        return douts[0], douts[0]
    if len(douts) == 2:
        return douts[0], douts[1]
    raise AssertionError(
        f"Expected 1 (shared) or 2 Dropout modules as direct children, got {len(douts)}"
    )


def _reference_block_pre_ln(
    x: torch.Tensor,
    mask: torch.Tensor,
    ln1: nn.LayerNorm,
    ln2: nn.LayerNorm,
    d1: nn.Dropout,
    d2: nn.Dropout,
    attn: nn.Module,
    mlp: nn.Module,
) -> torch.Tensor:
    """pre-LN residual: x1 = x + d1(attn(ln1(x), mask)); out = x1 + d2(mlp(ln2(x1)))."""
    h = x + d1(attn(ln1(x), mask))
    return h + d2(mlp(ln2(h)))


def _reference_gpt_stack(
    x: torch.Tensor,
    mask: torch.Tensor,
    top_drop: nn.Dropout,
    blocks: nn.ModuleList,
    ln_f: nn.LayerNorm,
) -> torch.Tensor:
    h = top_drop(x)
    for block in blocks:
        h = block(h, mask)
    return ln_f(h)


def _find_non_embedding_backbone(
    user_encoder: nn.Module,
) -> tuple[str, nn.Module]:
    """Exactly one top-level child that is not an Embedding (stack over item + position)."""
    matches = [
        (n, c)
        for n, c in user_encoder.named_children()
        if not isinstance(c, nn.Embedding)
    ]
    assert len(matches) == 1, (
        f"UserEncoder: expected one non-Embedding child, got {len(matches)}: {[m[0] for m in matches]}"
    )
    return matches[0]


def _user_encoder_padded_to_unpadded(
    item_embeddings: torch.Tensor,
    pos_emb: nn.Embedding,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    item_embeddings: flat (sum(L), D) in batch order. Returns x before encoder, mask.
    x has padding zeroed; mask is (B, T) bool.
    """
    device = item_embeddings.device
    dtype = item_embeddings.dtype
    emb_dim = item_embeddings.shape[1]
    batch_size = lengths.numel()
    seq_len = int(lengths.max().item())
    positions = torch.arange(seq_len, device=device)
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    padded = torch.zeros(batch_size, seq_len, emb_dim, device=device, dtype=dtype)
    padded[mask] = item_embeddings
    pos = torch.arange(seq_len, device=device, dtype=torch.long)
    x = padded + pos_emb(pos).unsqueeze(0)
    x = x.clone()
    x[~mask] = 0
    return x, mask


def _logq_shift_from_q_counts(
    q_counts: torch.Tensor, item_ids: torch.Tensor, eps: float
) -> torch.Tensor:
    """log q(i) from raw counts, without materializing a full q vector: log c_i - log sum c."""
    log_total = torch.log(q_counts.sum().clamp_min(eps))
    return torch.log(q_counts[item_ids].clamp_min(eps)) - log_total


def test_create_masked_tensor(func):
    data = torch.tensor([1, 2, 3, 4, 5, 6])
    lengths = torch.tensor([2, 3, 1])

    padded, mask = func(data, lengths)

    expected_padded = torch.tensor(
        [
            [1, 2, 0],
            [3, 4, 5],
            [6, 0, 0],
        ]
    )
    expected_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, False, False],
        ]
    )

    assert torch.equal(padded, expected_padded)
    assert torch.equal(mask, expected_mask)
    assert (padded[~mask] == 0).all()
    assert torch.equal(padded[mask], data)

    data2 = torch.arange(5 * 4).view(5, 4)
    lengths2 = torch.tensor([2, 3])

    padded2, mask2 = func(data2, lengths2)

    assert padded2.shape == (2, 3, 4)
    assert mask2.shape == (2, 3)
    assert (padded2[~mask2] == 0).all()
    assert torch.equal(padded2[mask2], data2)

    print("All good! :)")


def test_yambda_train_dataset(dataset_cls):
    histories = {
        "u_empty": [],
        "u_len1": [42],
        "u_len2": [10, 11],
        "u_exact_max": [7, 8, 9],
        "u_long": [1, 2, 3, 4, 5, 6],
    }
    max_seq_len = 3
    ds = dataset_cls(histories=histories, max_seq_len=max_seq_len)

    # u_empty and u_len1 should not produce any sample because history becomes empty after shift.
    expected_samples = [
        {"uid": "u_len2", "history": [10], "targets": [11], "length": 1},
        {"uid": "u_exact_max", "history": [7, 8], "targets": [8, 9], "length": 2},
        {"uid": "u_long", "history": [1, 2, 3], "targets": [2, 3, 4], "length": 3},
        {"uid": "u_long", "history": [4, 5], "targets": [5, 6], "length": 2},
    ]

    assert len(ds) == len(expected_samples), f"Unexpected number of samples: {len(ds)}"

    for idx, expected in enumerate(expected_samples):
        sample = ds[idx]
        assert set(sample.keys()) == {"uid", "history", "targets", "length"}
        assert sample["uid"] == expected["uid"], f"Wrong uid at idx={idx}"
        assert sample["history"] == expected["history"], f"Wrong history at idx={idx}"
        assert sample["targets"] == expected["targets"], f"Wrong targets at idx={idx}"
        assert sample["length"] == expected["length"], f"Wrong length at idx={idx}"

        assert isinstance(sample["history"], list)
        assert isinstance(sample["targets"], list)
        assert isinstance(sample["length"], int)
        assert len(sample["history"]) == sample["length"]
        assert len(sample["history"]) == len(sample["targets"])
        assert sample["length"] <= max_seq_len

    for sample in ds:
        assert sample["history"][1:] == sample["targets"][:-1], (
            "targets must be history shifted by one position"
        )

    ds_step = dataset_cls(histories={"u": [5, 6, 7, 8]}, max_seq_len=1)
    assert len(ds_step) == 3
    assert ds_step[0] == {"uid": "u", "history": [5], "targets": [6], "length": 1}
    assert ds_step[1] == {"uid": "u", "history": [6], "targets": [7], "length": 1}
    assert ds_step[2] == {"uid": "u", "history": [7], "targets": [8], "length": 1}

    print("All good! :)")


def test_yambda_eval_dataset(dataset_cls):
    histories = {
        "u_empty": [],
        "u_short": [10, 11],
        "u_exact": [1, 2, 3],
        "u_long": [4, 5, 6, 7, 8],
        "u_not_in_targets": [100, 101],
    }
    targets = {
        "u_short": [999],
        "u_exact": [888],
        "u_long": [777],
        "u_missing_history": [666],
        "u_empty": [555],
    }

    max_seq_len = 3
    ds = dataset_cls(histories=histories, targets=targets, max_seq_len=max_seq_len)

    expected_samples = [
        {"uid": "u_short", "history": [10, 11], "length": 2},
        {"uid": "u_exact", "history": [1, 2, 3], "length": 3},
        {"uid": "u_long", "history": [6, 7, 8], "length": 3},
    ]

    assert len(ds) == len(expected_samples), f"Unexpected number of samples: {len(ds)}"

    for idx, expected in enumerate(expected_samples):
        sample = ds[idx]
        assert set(sample.keys()) == {"uid", "history", "length"}
        assert sample["uid"] == expected["uid"], f"Wrong uid at idx={idx}"
        assert sample["history"] == expected["history"], f"Wrong history at idx={idx}"
        assert sample["length"] == expected["length"], f"Wrong length at idx={idx}"

        assert isinstance(sample["history"], list)
        assert isinstance(sample["length"], int)
        assert len(sample["history"]) == sample["length"]
        assert sample["length"] > 0
        assert sample["length"] <= max_seq_len

    uids = {ds[i]["uid"] for i in range(len(ds))}
    assert uids == {"u_short", "u_exact", "u_long"}
    assert "u_not_in_targets" not in uids
    assert "u_missing_history" not in uids
    assert "u_empty" not in uids

    ds_tail = dataset_cls(histories={"u": [5, 6, 7]}, targets={"u": [1]}, max_seq_len=1)
    assert len(ds_tail) == 1
    assert ds_tail[0] == {"uid": "u", "history": [7], "length": 1}

    ds_empty = dataset_cls(
        histories={"u1": [], "u2": [1]},
        targets={"u3": [10], "u1": [11]},
        max_seq_len=5,
    )
    assert len(ds_empty) == 0

    print("All good! :)")


def test_collate_fn(func):
    batch_train = [
        {"uid": 7, "history": [1, 2], "targets": [2, 3], "length": 2},
        {"uid": 8, "history": [10], "targets": [11], "length": 1},
        {"uid": 9, "history": [5, 6, 7], "targets": [6, 7, 8], "length": 3},
    ]
    out = func(batch_train)

    assert set(out.keys()) == {"uid", "length", "history", "targets"}
    assert torch.equal(out["uid"], torch.tensor([7, 8, 9], dtype=torch.long))
    assert torch.equal(out["length"], torch.tensor([2, 1, 3], dtype=torch.long))
    assert torch.equal(
        out["history"], torch.tensor([1, 2, 10, 5, 6, 7], dtype=torch.long)
    )
    assert torch.equal(
        out["targets"], torch.tensor([2, 3, 11, 6, 7, 8], dtype=torch.long)
    )

    assert out["uid"].dtype == torch.long
    assert out["length"].dtype == torch.long
    assert out["history"].dtype == torch.long
    assert out["targets"].dtype == torch.long
    assert out["uid"].numel() == out["length"].numel() == len(batch_train)
    assert out["history"].numel() == out["targets"].numel() == int(out["length"].sum())

    batch_eval = [
        {"uid": 101, "history": [4, 5, 6], "length": 3},
        {"uid": 102, "history": [9], "length": 1},
    ]
    out2 = func(batch_eval)

    assert set(out2.keys()) == {"uid", "length", "history"}
    assert "targets" not in out2
    assert torch.equal(out2["uid"], torch.tensor([101, 102], dtype=torch.long))
    assert torch.equal(out2["length"], torch.tensor([3, 1], dtype=torch.long))
    assert torch.equal(out2["history"], torch.tensor([4, 5, 6, 9], dtype=torch.long))
    assert out2["history"].numel() == int(out2["length"].sum())

    print("All good! :)")


def test_split_fields(data, train_df, test_df, catalog_size):
    assert isinstance(data, pl.DataFrame), "data must be a polars DataFrame"
    assert isinstance(train_df, pl.DataFrame), "train_df must be a polars DataFrame"
    assert isinstance(test_df, pl.DataFrame), "test_df must be a polars DataFrame"

    required_cols = {"uid", "item_id", "timestamp"}
    assert required_cols.issubset(set(data.columns)), (
        f"data must contain columns: {required_cols}"
    )
    assert required_cols.issubset(set(train_df.columns)), (
        f"train_df must contain columns: {required_cols}"
    )
    assert required_cols.issubset(set(test_df.columns)), (
        f"test_df must contain columns: {required_cols}"
    )

    assert train_df.height + test_df.height == data.height, (
        "train_df and test_df sizes must sum to data size"
    )

    if train_df.height > 0 and test_df.height > 0:
        assert train_df["timestamp"].max() <= test_df["timestamp"].min(), (
            "train_df timestamps should not be later than test_df timestamps"
        )

    expected_catalog_size = data["item_id"].n_unique()
    assert catalog_size == expected_catalog_size, (
        f"catalog_size={catalog_size}, expected {expected_catalog_size}"
    )
    assert train_df["item_id"].n_unique() <= catalog_size
    assert test_df["item_id"].n_unique() <= catalog_size

    integer_dtypes = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }
    assert data["item_id"].dtype in integer_dtypes, (
        "item_id must be integer after preprocessing"
    )
    assert data["item_id"].min() >= 0, "item_id must be non-negative"

    print("All good! :)")


def test_dataloaders(train_dataloader, eval_dataloader):
    assert train_dataloader is not None, "train_dataloader must be initialized"
    assert eval_dataloader is not None, "eval_dataloader must be initialized"
    assert hasattr(train_dataloader, "dataset"), "train_dataloader must have dataset"
    assert hasattr(eval_dataloader, "dataset"), "eval_dataloader must have dataset"
    assert len(train_dataloader.dataset) > 0, "train dataset must not be empty"
    assert len(eval_dataloader.dataset) > 0, "eval dataset must not be empty"

    train_batch = next(iter(train_dataloader))
    assert set(train_batch.keys()) == {"uid", "length", "history", "targets"}
    assert train_batch["uid"].dtype == torch.long
    assert train_batch["length"].dtype == torch.long
    assert train_batch["history"].dtype == torch.long
    assert train_batch["targets"].dtype == torch.long
    assert train_batch["uid"].ndim == 1
    assert train_batch["length"].ndim == 1
    assert train_batch["history"].ndim == 1
    assert train_batch["targets"].ndim == 1
    assert train_batch["uid"].numel() == train_batch["length"].numel()
    assert train_batch["history"].numel() == int(train_batch["length"].sum().item())
    assert train_batch["targets"].numel() == int(train_batch["length"].sum().item())
    assert (train_batch["length"] > 0).all()

    eval_batch = next(iter(eval_dataloader))
    assert set(eval_batch.keys()) == {"uid", "length", "history"}
    assert "targets" not in eval_batch
    assert eval_batch["uid"].dtype == torch.long
    assert eval_batch["length"].dtype == torch.long
    assert eval_batch["history"].dtype == torch.long
    assert eval_batch["uid"].ndim == 1
    assert eval_batch["length"].ndim == 1
    assert eval_batch["history"].ndim == 1
    assert eval_batch["uid"].numel() == eval_batch["length"].numel()
    assert eval_batch["history"].numel() == int(eval_batch["length"].sum().item())
    assert (eval_batch["length"] > 0).all()

    print("All good! :)")


def test_causal_self_attention(attn_cls):
    """
    Fixed spec: one Linear d→3d (QKV), one d→d (out), masked causal MHA. Plus behavioral invariants.
    """
    torch.manual_seed(42)
    d_model = 8
    n_heads = 2
    B, T = 2, 4

    attn = attn_cls(d_model=d_model, n_heads=n_heads, dropout=0.0)
    attn.eval()

    qkv_linear = _find_single_linear(attn, d_model, 3 * d_model)
    out_linear = _find_single_linear(attn, d_model, d_model)

    x = torch.randn(B, T, d_model)
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )

    out = attn(x, mask)
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all(), "output must be finite"

    with torch.no_grad():
        expected = _reference_causal_self_attention(
            x, mask, qkv_linear, out_linear, n_heads
        )
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4), (
        "Forward must match the HW multi-head attention numerics (QKV split, padding ∧ causal, SDPA/softmax, proj)"
    )

    # 3) Causality check: changing future token must not change past outputs
    mask_all_true = torch.ones((1, T), dtype=torch.bool)
    x_base = torch.randn(1, T, d_model)
    y_base = attn(x_base, mask_all_true)

    x_future_changed = x_base.clone()
    x_future_changed[:, -1, :] += 100.0  # strong perturbation at future position
    y_future_changed = attn(x_future_changed, mask_all_true)

    assert torch.allclose(y_base[:, :-1, :], y_future_changed[:, :-1, :], atol=1e-6), (
        "Past outputs changed after modifying a future token (causality broken)"
    )

    # 4) Key-padding check: changing masked key token must not affect valid queries
    mask_with_last_padded = torch.tensor([[True, True, True, False]], dtype=torch.bool)
    x_pad_base = torch.randn(1, T, d_model)
    y_pad_base = attn(x_pad_base, mask_with_last_padded)

    x_pad_changed = x_pad_base.clone()
    x_pad_changed[:, -1, :] += 100.0  # masked key position
    y_pad_changed = attn(x_pad_changed, mask_with_last_padded)

    assert torch.allclose(y_pad_base[:, :3, :], y_pad_changed[:, :3, :], atol=1e-6), (
        "Valid query outputs changed after modifying a masked key token"
    )

    print("All good! :)")


def test_mlp(mlp_cls):
    """Structure: two Linears, GELU (nn.GELU or F.gelu path), output checks (shape, pos-wise, eval vs train)."""
    torch.manual_seed(42)
    d_model = 6
    B, T = 3, 4
    dropout = 0.3

    mlp = mlp_cls(d_model=d_model, dropout=dropout)
    fc_up = _find_single_linear(mlp, d_model, 4 * d_model)
    fc_down = _find_single_linear(mlp, 4 * d_model, d_model)
    if _find_optional_single_gelu(mlp) is None:
        _assert_mlp_matches_gelu_between_linears(
            mlp_cls, mlp, fc_up, fc_down, d_model=d_model
        )
    x = torch.randn(B, T, d_model)

    mlp.eval()
    out = mlp(x)
    assert out.shape == x.shape, "output shape must match input"
    assert torch.isfinite(out).all(), "output must be finite"
    out_again = mlp(x)
    assert torch.allclose(out, out_again, atol=1e-6, rtol=1e-5), (
        "in eval, forward on same input should be deterministic"
    )

    mlp.train()
    y1, y2 = mlp(x), mlp(x)
    assert y1.shape == x.shape and y2.shape == x.shape
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    if dropout > 0:
        assert not torch.allclose(y1, y2, atol=0.0, rtol=0.0), (
            "with dropout>0, two consecutive train forwards on same x should not match bit-for-bit"
        )

    mlp.eval()
    x_ref = torch.randn(1, T, d_model)
    y_ref = mlp(x_ref)
    x_changed = x_ref.clone()
    x_changed[:, 2, :] += 10.0
    y_changed = mlp(x_changed)
    assert torch.allclose(y_ref[:, 0, :], y_changed[:, 0, :], atol=1e-5)
    assert torch.allclose(y_ref[:, 1, :], y_changed[:, 1, :], atol=1e-5)
    assert torch.allclose(y_ref[:, 3, :], y_changed[:, 3, :], atol=1e-5)

    print("All good! :)")


def test_block(block_cls):
    """
    Spec: pre-LN residual, exactly two top-level LayerNorms, 1 or 2 top-level Dropouts;
    one attention child (QKV d→3d) and one MLP child (d↔4d), names arbitrary.
    """
    torch.manual_seed(42)
    d_model = 8
    n_heads = 2
    B, T = 2, 5

    block = block_cls(d_model=d_model, n_heads=n_heads, dropout=0.0)
    block.eval()
    layer_norms = [m for m in block.children() if isinstance(m, nn.LayerNorm)]
    assert len(layer_norms) == 2, (
        "Block must contain exactly two top-level LayerNorm modules"
    )
    ln1, ln2 = layer_norms[0], layer_norms[1]
    d1, d2 = _block_dropouts_in_order(block)
    attn_module, mlp_module = _find_block_attention_and_mlp(block, d_model)

    x = torch.randn(B, T, d_model)
    mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )

    out = block(x, mask)
    assert out.shape == x.shape, "Block must preserve input shape"
    assert torch.isfinite(out).all(), "output must be finite"

    with torch.no_grad():
        expected = _reference_block_pre_ln(
            x, mask, ln1, ln2, d1, d2, attn_module, mlp_module
        )
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4), (
        "Block forward must match pre-LN: x+drop(attn(ln1(x),m)); +drop(mlp(ln2(...))) (dropout=0 in test)"
    )

    # 3) mask=None must be equivalent to all-True mask
    mask_all_true = torch.ones((B, T), dtype=torch.bool)
    out_none = block(x, None)
    out_all_true = block(x, mask_all_true)
    assert torch.allclose(out_none, out_all_true, atol=1e-6, rtol=1e-5), (
        "mask=None should behave like all-True mask"
    )

    # 4) Residual identity check when sublayers return zeros.
    class ZeroAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_mask = None

        def forward(self, x, mask):
            self.last_mask = mask
            return torch.zeros_like(x)

    class ZeroMLP(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    block_zero = block_cls(d_model=d_model, n_heads=n_heads, dropout=0.0)
    block_zero.eval()
    attn_m0, mlp_m0 = _find_block_attention_and_mlp(block_zero, d_model)
    z_attn, z_mlp = ZeroAttn(), ZeroMLP()
    for name, c in list(block_zero.named_children()):
        if c is attn_m0:
            setattr(block_zero, name, z_attn)
        elif c is mlp_m0:
            setattr(block_zero, name, z_mlp)

    y_zero = block_zero(x, None)
    assert torch.allclose(y_zero, x, atol=1e-7), (
        "With zero sublayers and residuals, output must equal input"
    )
    assert z_attn.last_mask is not None
    assert z_attn.last_mask.dtype == torch.bool
    assert z_attn.last_mask.shape == (B, T)
    assert z_attn.last_mask.all(), "Generated default mask must be all True"

    print("All good! :)")


def test_gpt(gpt_cls):
    torch.manual_seed(42)
    max_seq_len = 6
    n_layers = 3
    d_model = 8
    n_heads = 2
    B, T = 2, 5

    model = gpt_cls(
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.0,
    )
    model.eval()
    top_drop = _find_top_level_by_type(model, nn.Dropout)
    blocks = _find_single_modulelist(model)
    ln_final = _find_top_level_by_type(model, nn.LayerNorm)
    assert len(blocks) == n_layers, "Number of blocks must equal n_layers"

    x = torch.randn(B, T, d_model)
    mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    out = model(x, mask)
    assert out.shape == x.shape, "GPT must preserve (B, T, D) shape"
    assert torch.isfinite(out).all(), "output must be finite"

    with torch.no_grad():
        expected = _reference_gpt_stack(x, mask, top_drop, blocks, ln_final)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4), (
        "GPT must be: drop -> block(x,m) for each block -> final LayerNorm"
    )

    # 3) max_seq_len guard
    x_too_long = torch.randn(B, max_seq_len + 1, d_model)
    mask_too_long = torch.ones((B, max_seq_len + 1), dtype=torch.bool)
    try:
        model(x_too_long, mask_too_long)
        raise AssertionError("Expected assertion when T > max_seq_len")
    except AssertionError:
        pass

    # 4) Check that mask is passed unchanged to every block
    class SpyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_mask = None
            self.calls = 0

        def forward(self, x, mask):
            self.seen_mask = mask.clone()
            self.calls += 1
            return x + 1.0

    model_spy = gpt_cls(
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.0,
    )
    model_spy.eval()
    spies = [SpyBlock() for _ in range(n_layers)]
    modlist = _find_single_modulelist(model_spy)
    for name, c in list(model_spy.named_children()):
        if c is modlist:
            setattr(model_spy, name, nn.ModuleList(spies))
            break
    else:
        raise AssertionError("ModuleList of blocks not found on GPT")

    x0 = torch.zeros(B, T, d_model)
    y0 = model_spy(x0, mask)

    # Each spy adds +1, then final LayerNorm should still keep shape.
    assert y0.shape == x0.shape
    for spy in spies:
        assert spy.calls == 1, "Each block must be called exactly once"
        assert torch.equal(spy.seen_mask, mask), (
            "Mask must be passed unchanged to blocks"
        )

    # 5) Dropout behavior sanity: deterministic in eval, stochastic in train.
    model_do = gpt_cls(
        max_seq_len=max_seq_len,
        n_layers=1,
        d_model=d_model,
        n_heads=n_heads,
        dropout=0.5,
    )
    x_do = torch.randn(B, T, d_model)
    m_do = torch.ones((B, T), dtype=torch.bool)

    model_do.eval()
    y_eval_1 = model_do(x_do, m_do)
    y_eval_2 = model_do(x_do, m_do)
    assert torch.allclose(y_eval_1, y_eval_2, atol=1e-7), (
        "Eval mode must be deterministic"
    )

    model_do.train()
    torch.manual_seed(1)
    y_train_1 = model_do(x_do, m_do)
    torch.manual_seed(2)
    y_train_2 = model_do(x_do, m_do)
    assert not torch.allclose(y_train_1, y_train_2), (
        "Train mode with dropout should produce stochastic outputs"
    )

    print("All good! :)")


def test_user_encoder(encoder_cls):
    """Batch to UserEncoder: keys ``history`` (flat item ids) and ``length`` (per user)."""
    torch.manual_seed(42)
    num_items = 30
    emb_dim = 8
    max_seq_len = 5

    model = encoder_cls(
        num_items=num_items,
        embedding_dim=emb_dim,
        max_seq_len=max_seq_len,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
    )
    model.eval()
    emb_layers = [m for m in model.modules() if isinstance(m, nn.Embedding)]
    assert len(emb_layers) >= 2, (
        "UserEncoder must contain item and positional embeddings"
    )
    item_emb = next(
        (
            m
            for m in emb_layers
            if m.num_embeddings == num_items and m.embedding_dim == emb_dim
        ),
        None,
    )
    pos_emb = next(
        (
            m
            for m in emb_layers
            if m.num_embeddings == max_seq_len and m.embedding_dim == emb_dim
        ),
        None,
    )
    assert item_emb is not None, "Item embedding layer not found"
    assert pos_emb is not None, "Positional embedding layer not found"

    class SpyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_x = None
            self.last_mask = None

        def forward(self, x, mask):
            self.last_x = x.detach().clone()
            self.last_mask = mask.detach().clone()
            return x + 0.5

    spy = SpyEncoder()
    enc_name, _ = _find_non_embedding_backbone(model)
    setattr(model, enc_name, spy)

    history = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    lengths = torch.tensor([3, 2, 1], dtype=torch.long)
    out = model({"history": history, "length": lengths})

    with torch.no_grad():
        item_embeddings = item_emb(history)
        x_pad, mask = _user_encoder_padded_to_unpadded(
            item_embeddings, pos_emb, lengths
        )
        expected = (x_pad + 0.5)[mask]

    assert out.shape == expected.shape == (int(lengths.sum().item()), emb_dim)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5), (
        "UserEncoder forward: expected item_emb + pad + pos (padded=0) -> backbone -> [mask] out"
    )

    # Ensure mask is passed to encoder correctly.
    assert spy.last_mask is not None
    assert spy.last_mask.dtype == torch.bool
    assert torch.equal(spy.last_mask, mask)

    # Ensure padded positions are zeroed out before encoder.
    assert spy.last_x is not None
    assert torch.allclose(spy.last_x[~mask], torch.zeros_like(spy.last_x[~mask]))

    # Constructor invariant: embedding_dim divisible by n_heads.
    try:
        encoder_cls(num_items=num_items, embedding_dim=10, n_heads=4)
        raise AssertionError(
            "Expected assertion when embedding_dim is not divisible by n_heads"
        )
    except AssertionError:
        pass

    print("All good! :)")


def test_train_nip_model(model_cls):
    torch.manual_seed(42)
    num_items = 50
    emb_dim = 6
    num_negatives = 4

    q_counts = torch.arange(1, num_items + 1, dtype=torch.float32)

    model = model_cls(
        num_items=num_items,
        embedding_dim=emb_dim,
        num_negatives=num_negatives,
        q_counts=q_counts,
        max_seq_len=5,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
    )

    class DummyEncoder(nn.Module):
        def __init__(self, n_items: int, d_model: int):
            super().__init__()
            self.num_items = n_items
            self.item_embeddings = nn.Embedding(n_items, d_model)
            self.last_inputs = None

        def forward(self, inputs):
            self.last_inputs = {k: v.detach().clone() for k, v in inputs.items()}
            return self.item_embeddings(inputs["history"])

    dummy_encoder = DummyEncoder(num_items, emb_dim)
    with torch.no_grad():
        weight = torch.arange(num_items * emb_dim, dtype=torch.float32).view(
            num_items, emb_dim
        )
        dummy_encoder.item_embeddings.weight.copy_(weight / 100.0)
    model.encoder = dummy_encoder

    inputs = {
        "history": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        "length": torch.tensor([3, 2], dtype=torch.long),
        "targets": torch.tensor([2, 3, 4, 5, 6], dtype=torch.long),
    }

    torch.manual_seed(123)
    loss = model(inputs)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0, (
        "TrainNIPModel.forward must return scalar loss"
    )
    assert torch.isfinite(loss), "Loss must be finite"

    # Forward should pass flattened history + lengths into UserEncoder.
    assert dummy_encoder.last_inputs is not None
    assert torch.equal(dummy_encoder.last_inputs["history"], inputs["history"])
    assert torch.equal(dummy_encoder.last_inputs["length"], inputs["length"])

    # Numerical check for compute_loss against manual reference with the same seed.
    queries = dummy_encoder.item_embeddings(inputs["history"])
    target_ids = inputs["targets"]
    n = target_ids.shape[0]

    torch.manual_seed(123)
    negative_pos = torch.randint(0, n, (n, num_negatives), device=target_ids.device)
    negative_ids = target_ids[negative_pos]
    candidate_ids = torch.cat([target_ids[:, None], negative_ids], dim=1)
    candidate_embeddings = dummy_encoder.item_embeddings(candidate_ids)
    logits = (queries[:, None, :] * candidate_embeddings).sum(dim=-1)

    neg_logq = _logq_shift_from_q_counts(model.q_counts, negative_ids, float(model.eps))
    logits[:, 1:] = logits[:, 1:] - neg_logq

    labels = torch.zeros(n, dtype=torch.long)
    expected = F.cross_entropy(logits, labels)
    assert torch.allclose(loss, expected, atol=1e-6, rtol=1e-5), (
        "compute_loss does not match sampled-softmax in-batch logq reference"
    )

    # Targets must affect loss.
    torch.manual_seed(123)
    shuffled_loss = model(
        {
            "history": inputs["history"],
            "length": inputs["length"],
            "targets": inputs["targets"].flip(0),
        }
    )
    assert not torch.allclose(loss, shuffled_loss), (
        "Changing targets should change the loss value"
    )

    # q_counts must affect correction.
    model_uniform_q = model_cls(
        num_items=num_items,
        embedding_dim=emb_dim,
        num_negatives=num_negatives,
        q_counts=torch.ones(num_items, dtype=torch.float32),
        max_seq_len=5,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
    )
    model_uniform_q.encoder = dummy_encoder
    torch.manual_seed(123)
    loss_uniform_q = model_uniform_q(inputs)
    assert not torch.allclose(loss, loss_uniform_q), (
        "Changing q_counts should change loss due to logq correction"
    )

    print("All good! :)")


def test_train_loop(target_train_loss):
    assert 16.0 <= target_train_loss <= 16.9, (
        f"Average train loss is not in the range: got {target_train_loss:.6f}. Should be in [16.0, 16.9]"
    )
    print("All good! :)")


def test_logq_coefficients(build_q_from_train_targets):
    # --- 1) Plain 1D: multiplicities per id, tail ids zero-padded by minlength ---
    train = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.long)
    catalog_size = 5
    q = build_q_from_train_targets(train, catalog_size)
    assert isinstance(q, torch.Tensor)
    assert q.ndim == 1 and q.shape[0] == catalog_size
    assert torch.allclose(
        q,
        torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0]),
    ), "counts should be bincount(train_targets, minlength=catalog_size)"

    # --- 2) 2D targets are flattened (same order as row-major flatten) ---
    train_2d = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    q2 = build_q_from_train_targets(train_2d, catalog_size=3)
    assert torch.allclose(q2, torch.tensor([2.0, 1.0, 1.0])), (
        "train_targets should be reshaped to 1D before counting"
    )

    # --- 3) Sparse high catalog_size: zeros for ids that never appear as targets ---
    train3 = torch.tensor([0, 3, 3], dtype=torch.long)
    q3 = build_q_from_train_targets(train3, catalog_size=6)
    assert q3.shape == (6,)
    assert float(q3[0].item()) == 1.0 and float(q3[3].item()) == 2.0
    assert float(q3.sum().item()) == 3.0

    # --- 4) Empty list of targets must fail (no q to define) ---
    try:
        build_q_from_train_targets(torch.tensor([], dtype=torch.long), catalog_size=4)
        raise AssertionError("expected ValueError for empty train_targets")
    except ValueError:
        pass

    # --- 5) Ids outside [0, catalog_size) ---
    for bad, n in [([5], 5), ([-1, 0], 3)]:
        try:
            build_q_from_train_targets(
                torch.tensor(bad, dtype=torch.long), catalog_size=n
            )
            raise AssertionError(f"expected ValueError for invalid ids {bad}, n={n}")
        except ValueError:
            pass

    assert torch.isfinite(q).all(), "counts should be finite"
    assert (q >= 0).all(), "counts must be non-negative"

    print("All good! :)")


def test_eval_nip_model(model_cls):
    torch.manual_seed(42)
    num_items = 7
    emb_dim = 4

    model = model_cls(
        num_items=num_items,
        embedding_dim=emb_dim,
        max_seq_len=5,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
    )
    model.eval()

    class DummyEncoder(nn.Module):
        def __init__(self, n_items: int, d_model: int):
            super().__init__()
            self.num_items = n_items
            self.item_embeddings = nn.Embedding(n_items, d_model)
            self.last_inputs = None

        def forward(self, inputs):
            self.last_inputs = {k: v.detach().clone() for k, v in inputs.items()}
            return self.item_embeddings(inputs["history"])

    dummy = DummyEncoder(num_items, emb_dim)
    with torch.no_grad():
        weight = torch.arange(num_items * emb_dim, dtype=torch.float32).view(
            num_items, emb_dim
        )
        dummy.item_embeddings.weight.copy_(weight / 10.0)
    model.encoder = dummy

    # Two users with lengths [3, 2] in flattened history.
    inputs = {
        "uid": torch.tensor([10, 11], dtype=torch.long),
        "length": torch.tensor([3, 2], dtype=torch.long),
        "history": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
    }

    scores = model(inputs)
    assert scores.shape == (2, num_items), "EvalNIPModel must return [B, num_items]"

    # Ensure inputs are passed to encoder in expected format.
    assert dummy.last_inputs is not None
    assert torch.equal(dummy.last_inputs["history"], inputs["history"])
    assert torch.equal(dummy.last_inputs["length"], inputs["length"])

    # Manual score check: take last position per user and dot with item matrix.
    encoder_output = dummy.item_embeddings(inputs["history"])  # [N, D]
    last_idx = inputs["length"].cumsum(dim=0) - 1
    user_last = encoder_output[last_idx]
    expected = user_last @ dummy.item_embeddings.weight.T
    assert torch.allclose(scores, expected, atol=1e-6, rtol=1e-5), (
        "EvalNIPModel scores mismatch manual reference"
    )

    print("All good! :)")


def test_eval(eval_fn):
    class DummyModel(nn.Module):
        def __init__(self, scores_by_uid: dict[int, torch.Tensor]) -> None:
            super().__init__()
            self.scores_by_uid = scores_by_uid
            self.grad_enabled_flags: list[bool] = []

        def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            self.grad_enabled_flags.append(torch.is_grad_enabled())
            rows = [self.scores_by_uid[int(uid.item())] for uid in inputs["uid"]]
            return torch.stack(rows, dim=0).to(inputs["uid"].device)

    scores_by_uid = {
        1: torch.tensor([0.1, 0.2, 0.9, 0.3, 0.0]),
        2: torch.tensor([0.8, 0.1, 0.2, 0.0, 0.3]),
        3: torch.tensor([0.0, 0.4, 0.5, 0.2, 0.1]),
    }
    model = DummyModel(scores_by_uid)
    model.train()  # eval_fn must switch it to eval mode

    dataloader = [
        {
            "uid": torch.tensor([1, 2], dtype=torch.long),
            "length": torch.tensor([2, 2], dtype=torch.long),
            "history": torch.tensor([10, 11, 12, 13], dtype=torch.long),
        },
        {
            "uid": torch.tensor([3], dtype=torch.long),
            "length": torch.tensor([1], dtype=torch.long),
            "history": torch.tensor([14], dtype=torch.long),
        },
    ]

    fake_targets = {1: [2], 2: [4], 3: [1]}
    expected_candidates = {
        1: [2, 3],
        2: [0, 4],
        3: [2, 1],
    }
    expected_metrics = {
        "hitrate": 0.5,
        "recall": 0.25,
        "ndcg": 0.2,
        "coverage": 0.4,
    }

    observed = {}

    def fake_evaluate(*, targets, candidates, topk, catalog_size):
        observed["targets"] = targets
        observed["candidates"] = candidates
        observed["topk"] = topk
        observed["catalog_size"] = catalog_size
        return expected_metrics

    g = eval_fn.__globals__
    old_evaluate = g.get("evaluate")
    old_targets = g.get("targets")
    g["evaluate"] = fake_evaluate
    g["targets"] = fake_targets
    try:
        metrics = eval_fn(
            dataloader=dataloader,
            model=model,
            catalog_size=5,
            topk=2,
            device="cpu",
        )
    finally:
        if old_evaluate is None:
            g.pop("evaluate", None)
        else:
            g["evaluate"] = old_evaluate
        if old_targets is None:
            g.pop("targets", None)
        else:
            g["targets"] = old_targets

    assert metrics == expected_metrics
    assert observed["targets"] == fake_targets
    assert observed["candidates"] == expected_candidates
    assert observed["topk"] == 2
    assert observed["catalog_size"] == 5
    assert model.training is False, "eval_fn must call model.eval()"
    assert model.grad_enabled_flags and all(
        not flag for flag in model.grad_enabled_flags
    ), "eval_fn must run model forward without gradients"

    print("All good! :)")


def check_all_metrics_geq(metrics, hitrate, recall, ndcg, coverage):
    assert metrics["recall"] >= recall, "Too low recall value"
    assert metrics["hitrate"] >= hitrate, "Too low hitrate value"
    assert metrics["ndcg"] >= ndcg, "Too low ndcg value"
    assert metrics["coverage"] >= coverage, "Too low coverage value"


def check_nip_recs(metrics):
    TARGET_HITRATE = 0.35
    TARGET_RECALL = 0.11
    TARGET_NDCG = 0.045
    TARGET_COVERAGE = 0.35

    if ... in (TARGET_HITRATE, TARGET_RECALL, TARGET_NDCG, TARGET_COVERAGE):
        raise AssertionError(
            "Set target thresholds in check_nip_recs before running metric assertions"
        )

    check_all_metrics_geq(
        metrics,
        hitrate=TARGET_HITRATE,
        recall=TARGET_RECALL,
        ndcg=TARGET_NDCG,
        coverage=TARGET_COVERAGE,
    )
    print("All good! :)")
