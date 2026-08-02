"""Tests for the private-skip Gaussian mechanism.

The interesting properties are the two that are invisible in a training curve: the clip
actually bounds one sample's contribution to C across all layers, and the noise is scaled to
that C rather than to the per-layer C/sqrt(L).

    python -m pytest dpgrape/tests/test_private_subspace.py
    python dpgrape/tests/test_private_subspace.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpgrape.private_subspace import PerLayerGaussianMechanism  # noqa: E402

SHAPES = [(8, 4), (4, 8), (6, 6)]
CLIP = 3.0
SIGMA = 1.5
BATCH = 5


def make_params():
    return [torch.zeros(*shape, requires_grad=True) for shape in SHAPES]


def set_grads(params, scale):
    """Give every param a gradient of a known, identical Frobenius norm."""
    for p in params:
        g = torch.randn(*p.shape)
        p.grad = g * (scale / torch.linalg.vector_norm(g))


def make_mech(params, noise_multiplier=SIGMA, batch=BATCH):
    return PerLayerGaussianMechanism(
        params=params,
        clip_threshold=CLIP,
        noise_multiplier=noise_multiplier,
        expected_batch_size=batch,
    )


def test_noise_is_calibrated_to_the_joint_sensitivity():
    """std must be sigma*C, not sigma*C/sqrt(L). See the module header of private_subspace."""
    params = make_params()
    mech = make_mech(params)

    assert mech.per_layer_clip == CLIP / math.sqrt(len(SHAPES))
    assert mech.noise_std == SIGMA * CLIP
    assert mech.noise_std != SIGMA * mech.per_layer_clip
    assert mech.released_noise_std == SIGMA * CLIP / BATCH


def test_one_sample_contribution_is_bounded_by_C():
    """The whole point: removing one sample moves the released sum by at most C."""
    params = make_params()
    mech = make_mech(params, noise_multiplier=0.0)

    # A sample whose gradient is enormous in every layer.
    mech.reset()
    set_grads(params, scale=1e6)
    mech.accumulate()

    contribution_sq = sum(
        torch.linalg.vector_norm(mech._sums[p]).item() ** 2 for p in params
    )
    assert math.sqrt(contribution_sq) <= CLIP + 1e-4
    # and every layer individually sits at its own budget
    for p in params:
        assert torch.linalg.vector_norm(mech._sums[p]).item() <= mech.per_layer_clip + 1e-5


def test_small_gradients_pass_through_unclipped():
    params = make_params()
    mech = make_mech(params, noise_multiplier=0.0)
    mech.reset()

    tiny = mech.per_layer_clip / 10
    expected = {p: torch.zeros_like(p) for p in params}
    for _ in range(BATCH):
        set_grads(params, scale=tiny)
        for p in params:
            expected[p] += p.grad
        mech.accumulate()

    assert mech.num_clipped == 0
    released = mech.release()
    for p in params:
        assert torch.allclose(released[p], expected[p] / BATCH, atol=1e-6)


def test_release_is_the_noised_mean():
    """With noise off, release() is exactly the mean of the clipped per-sample gradients."""
    params = make_params()
    mech = make_mech(params, noise_multiplier=0.0)
    mech.reset()
    for _ in range(BATCH):
        set_grads(params, scale=1e6)
        mech.accumulate()
    released = mech.release()

    # every sample was clipped to the per-layer budget, so the mean sits there too
    for p in params:
        assert torch.linalg.vector_norm(released[p]).item() <= mech.per_layer_clip + 1e-4
    assert mech.num_clipped == BATCH * len(SHAPES)


def test_empirical_noise_matches_released_noise_std():
    torch.manual_seed(0)
    params = [torch.zeros(64, 64, requires_grad=True) for _ in range(2)]
    mech = PerLayerGaussianMechanism(
        params=params, clip_threshold=CLIP, noise_multiplier=SIGMA, expected_batch_size=BATCH
    )
    mech.reset()
    for _ in range(BATCH):
        for p in params:
            p.grad = torch.zeros_like(p)   # zero signal, so the release is pure noise
        mech.accumulate()
    released = mech.release()

    for p in params:
        empirical = released[p].std().item()
        assert abs(empirical - mech.released_noise_std) / mech.released_noise_std < 0.05


def test_zero_gradient_layer_does_not_produce_nan():
    params = make_params()
    mech = make_mech(params, noise_multiplier=0.0)
    mech.reset()
    for _ in range(BATCH):
        for p in params:
            p.grad = torch.zeros_like(p)
        mech.accumulate()
    released = mech.release()
    for p in params:
        assert torch.isfinite(released[p]).all()
        assert torch.count_nonzero(released[p]) == 0


def test_short_batch_is_rejected():
    """A short final batch would make the accountant's sampling rate wrong."""
    params = make_params()
    mech = make_mech(params, noise_multiplier=0.0)
    mech.reset()
    set_grads(params, scale=0.1)
    mech.accumulate()
    try:
        mech.release()
    except RuntimeError as exc:
        assert "drop_last" in str(exc)
    else:
        raise AssertionError("release() accepted a batch of the wrong size")


def test_diagnostics_report_the_pre_clip_scale():
    params = make_params()
    mech = make_mech(params, noise_multiplier=0.0)
    mech.reset()

    per_layer_norm = 2 * mech.per_layer_clip     # every layer over budget by 2x
    for _ in range(BATCH):
        set_grads(params, scale=per_layer_norm)
        mech.accumulate()
    diag = mech.diagnostics()

    assert diag["clipped_fraction"] == 1.0
    expected_total = per_layer_norm * math.sqrt(len(SHAPES))
    assert abs(diag["mean_sample_norm_pre_clip"] - expected_total) < 1e-4
    assert diag["num_layers"] == len(SHAPES)


def test_microbatch_loop_recovers_per_sample_gradients():
    """The trainer's microbatch-of-one loop is only correct if a 1-sample backward gives
    that sample's gradient. It does, because the loss is a mean over the batch. With the
    clip never binding, summing those and dividing by B must reproduce the batch gradient
    exactly -- that equality is what makes the clipped version a bounded-sensitivity
    estimator of the same quantity."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
    params = [m.weight for m in model if isinstance(m, torch.nn.Linear)]
    x = torch.randn(BATCH, 6)
    y = torch.randint(0, 3, (BATCH,))
    loss_fn = torch.nn.CrossEntropyLoss()

    model.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()
    batch_grads = {p: p.grad.clone() for p in params}

    # Huge clip threshold, no noise: the mechanism must be a no-op on top of averaging.
    mech = PerLayerGaussianMechanism(
        params=params, clip_threshold=1e6, noise_multiplier=0.0, expected_batch_size=BATCH
    )
    mech.reset()
    for i in range(BATCH):
        model.zero_grad(set_to_none=True)
        loss_fn(model(x[i:i + 1]), y[i:i + 1]).backward()
        mech.accumulate()
    released = mech.release()

    assert mech.num_clipped == 0
    for p in params:
        assert torch.allclose(released[p], batch_grads[p], atol=1e-6)


def test_drives_the_projector():
    """The privatized gradient is a drop-in for the bare one LowRankProjectorDP expects."""
    from dpgrape.low_rank_projector_dp import LowRankProjectorDP

    params = make_params()
    mech = make_mech(params)
    mech.reset()
    for _ in range(BATCH):
        set_grads(params, scale=1.0)
        mech.accumulate()
    released = mech.release()

    for method in ("galore", "subtrack"):
        for p in params:
            proj = LowRankProjectorDP(rank=2, method=method, subspace_update_interval=2)
            proj.update_subspace(released[p], iter=0)
            assert proj.ortho_matrix is not None
            assert proj.last_ortho_err < 1e-4


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:   # noqa: BLE001 - a standalone runner wants the message
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
