"""Smoke test for LowRankProjectorDP.

Checks that the SubTrack++ / GaLore projector drops into DP-GRAPE's pipeline: same shapes
as RandProjectorDP, correct project/project_back round trip, and a full
per-sample-project -> flat-clip -> noise -> DPAdamW -> project-back step that leaves the
weight update inside the tracked subspace.

Needs only torch + transformers (no opacus, no GPU):

    python -m pytest dpgrape/tests/test_low_rank_projector_dp.py
    python dpgrape/tests/test_low_rank_projector_dp.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# low_rank_projector.track_the_subspace hardcodes .to('cuda'); redirect it on CPU-only boxes.
if not torch.cuda.is_available():
    _orig_to = torch.Tensor.to

    def _to(self, *args, **kwargs):
        args = tuple("cpu" if a == "cuda" else a for a in args)
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return _orig_to(self, *args, **kwargs)

    torch.Tensor.to = _to

from dpgrape.dpadamw import DPAdamW                       # noqa: E402
from dpgrape.low_rank_projector_dp import LowRankProjectorDP  # noqa: E402
from dpgrape.rand_projector_dp import RandProjectorDP     # noqa: E402

SHAPES = [(8, 4), (4, 8), (6, 6)]   # m > n, m < n, square
RANK = 2
BATCH = 5


def make_projector(shape, method, seed=0):
    torch.manual_seed(seed)
    proj = LowRankProjectorDP(rank=RANK, method=method, subspace_update_interval=2)
    proj.update_subspace(torch.randn(*shape), iter=0)
    return proj


def test_shapes_match_rand_projector():
    """Projected per-sample gradients must have the shape DP-GRAPE's optimizer expects."""
    for shape in SHAPES:
        gs = torch.randn(BATCH, *shape)

        ref = RandProjectorDP(rank=RANK, rand_type="gaussian")
        ref.update_seed()
        ref.generate(shape, torch.float32, torch.device("cpu"))
        expected = ref.project(gs, batch=True).shape

        for method in ("galore", "subtrack"):
            proj = make_projector(shape, method)
            proj.generate(shape, torch.float32, torch.device("cpu"))
            assert proj.project(gs, batch=True).shape == expected, (shape, method)


def test_project_back_round_trip():
    """project_back(project(G)) must equal the orthogonal projection of G onto the basis."""
    for shape in SHAPES:
        for method in ("galore", "subtrack"):
            proj = make_projector(shape, method)
            ortho = proj.ortho_matrix
            grad = torch.randn(*shape)

            low = proj.project(grad, batch=False)
            back = proj.project_back(low)
            assert back.shape == shape

            if proj.side == "right":       # basis has orthonormal rows, acts on columns
                expected = grad @ ortho.t() @ ortho
            else:                          # basis has orthonormal columns, acts on rows
                expected = ortho @ ortho.t() @ grad
            assert torch.allclose(back, expected, atol=1e-5), (shape, method)


def test_generate_and_clear_are_no_ops():
    """The hooks call generate/clear around every projection; neither may touch the basis."""
    proj = make_projector((8, 4), "subtrack")
    before = proj.ortho_matrix.clone()
    proj.generate((8, 4), torch.float32, torch.device("cpu"))
    proj.clear_projection_matrix()
    proj.update_seed()
    assert proj.ortho_matrix is not None
    assert torch.equal(proj.ortho_matrix, before)


def test_basis_is_orthonormal_and_rotates():
    """GaLore re-SVDs, SubTrack takes a geodesic step; both keep an orthonormal basis."""
    for method in ("galore", "subtrack"):
        proj = make_projector((8, 4), method, seed=1)
        torch.manual_seed(2)
        proj.update_subspace(torch.randn(8, 4), iter=2)

        basis = proj.ortho_matrix.t() if proj.side == "right" else proj.ortho_matrix
        eye = torch.eye(basis.shape[1])
        assert torch.allclose(basis.t() @ basis, eye, atol=1e-4), method
        assert proj.last_ortho_err < 1e-4, method
        assert proj.last_rotation_deg is not None and proj.last_rotation_deg > 0.0, method


def test_full_dp_step_stays_in_subspace():
    """End-to-end: the weight update must lie in the subspace the projector is tracking.

    This is the property the whole design rests on -- it is why colspace(W_T - W_0) leaks
    the projectors, and why the DP noise (added inside the subspace) cannot move the
    weights out of it.
    """
    torch.manual_seed(0)
    clip, noise_multiplier = 1.0, 1.1

    for shape in SHAPES:
        for method in ("galore", "subtrack"):
            weight = torch.nn.Parameter(torch.randn(*shape))
            weight.proj_grad = None
            proj = make_projector(shape, method)

            opt = DPAdamW(
                [{"params": [weight], "rank": RANK, "scale": 1.0, "proj_type": "std"}],
                lr=1e-2, no_deprecation_warning=True,
            )
            opt.state[weight]["projector"] = proj

            # --- what the backward hook does: project each per-sample gradient ---
            per_sample = proj.project(torch.randn(BATCH, *shape), batch=True)

            # --- what DPOptimizerRandomProj does: flat clip, sum, noise, divide by B ---
            norms = per_sample.flatten(1).norm(dim=1)
            factors = (clip / norms).clamp(max=1.0)
            summed = (per_sample * factors.view(-1, 1, 1)).sum(0)
            noise = torch.normal(0.0, noise_multiplier * clip, size=summed.shape)
            weight.proj_grad = (summed + noise) / BATCH
            # DPAdamW skips params whose .grad is None; autograd still fills it in a real
            # run, opacus only redirects the *projected* gradient to .proj_grad.
            weight.grad = torch.zeros_like(weight)

            before = weight.detach().clone()
            opt.step()
            delta = weight.detach() - before

            assert delta.shape == shape
            assert torch.isfinite(delta).all()
            assert delta.abs().max() > 0

            # Orthonormal basis for the subspace, then residual outside it.
            ortho = proj.ortho_matrix
            if proj.side == "right":
                basis = torch.linalg.qr(ortho.t())[0]           # (n, rank)
                residual = delta - delta @ basis @ basis.t()
            else:
                basis = torch.linalg.qr(ortho)[0]               # (m, rank)
                residual = delta - basis @ basis.t() @ delta
            rel = residual.norm() / delta.norm()
            assert rel < 1e-5, (shape, method, rel.item())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
