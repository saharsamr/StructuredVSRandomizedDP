# Adapter that lets the SubTrack++ / GaLore projector drive DP-GRAPE's machinery.
#
# DP-GRAPE's pipeline (hooks -> DPOptimizerRandomProj -> DPAdamW) talks to a projector
# object through a fixed interface:
#
#     generate(shape, dtype, device)   # (re)build the projection matrix
#     project(grad, batch=True)        # per-sample projection, inside the backward hook
#     project_back(low_rank_grad)      # inside DPAdamW.step
#     clear_projection_matrix()        # free the matrix
#     update_seed()                    # redraw the random projector
#
# RandProjectorDP regenerates its matrix from a seed on every call, so generate/clear are
# real work and nothing is stored. A data-driven subspace cannot be regenerated, so
# LowRankProjectorDP *stores* the basis and makes generate/clear/update_seed no-ops. The
# basis itself is produced only by ``update_subspace``, which the trainer calls at subspace
# update steps.
#
# !!! PRIVACY !!!  What ``update_subspace`` is fed decides the privacy of the whole run, and
# this class does not and cannot check which it got:
#
#   --oracle_batch_mode shared | skip    a BARE (unclipped, unnoised) batch gradient. The
#       basis is then a noiseless function of private data, the weight *updates* are DP but
#       the subspace is not, and the method is eps = infinity as a whole (DP-GaLore /
#       DPTrack-Oracle). See DPTRACK_DESIGN.md section 4.
#
#   --oracle_batch_mode private-skip     the output of a Gaussian mechanism over the
#       held-out dev split (dpgrape/private_subspace.py). Everything this class then does is
#       post-processing, so the run has a finite eps.

import torch

from .low_rank_torch import LowRankProjector


class LowRankProjectorDP:
    """Stores a data-driven subspace and applies it the way RandProjectorDP does.

    All subspace math is delegated to the vendored ``LowRankProjector`` (SubTrack++ /
    GaLore, unmodified). This class only handles the interface and the shape bookkeeping
    that DP-GRAPE's optimizer expects.

    Args:
        rank: subspace rank r.
        scale: multiplied into project_back (DP-GRAPE uses 1.0).
        proj_type: only 'std' is supported, matching RandProjectorDP's default.
        method: 'galore' (SVD every update) or 'subtrack' (rank-1 geodesic step).
        subspace_update_interval: F, must equal the trainer's subspace_T.
        st_step_size: SubTrack geodesic step size (ignored by GaLore).
        st_step_size_scheduler: 'iterative_decrease' or None.
    """

    def __init__(
        self,
        rank,
        scale=1.0,
        proj_type="std",
        method="subtrack",
        subspace_update_interval=100,
        st_step_size=1.0,
        st_step_size_scheduler=None,
    ):
        if proj_type != "std":
            raise ValueError(f"only proj_type='std' is supported, got {proj_type}")
        if method not in ("galore", "subtrack"):
            raise ValueError(f"method must be 'galore' or 'subtrack', got {method}")

        self.rank = rank
        self.scale = scale
        self.proj_type = proj_type
        self.method = method

        # 'right' -> basis is (rank, n), acts on the second axis of an (m, n) gradient.
        # 'left'  -> basis is (m, rank), acts on the first axis.
        # Same rule as RandProjectorDP/LowRankProjector: reduce the smaller of the two axes.
        self.side = None

        # Diagnostics filled in by update_subspace (see trainer logging).
        self.last_rotation_deg = None
        self.last_ortho_err = None

        # update_rank=1 only: the k>1 geodesic path in low_rank_projector.py is shape-buggy.
        # Everything else is left at its default, which disables randomization, hybrid
        # GrassJump/GrassWalk, tangent-energy gating and step-size tuning.
        self._lrp = LowRankProjector(
            rank=rank,
            scale=scale,
            proj_type=proj_type,
            st_init_step_size=st_step_size,
            subspace_update_method=method,
            st_step_size_scheduler=st_step_size_scheduler,
            subspace_update_interval=subspace_update_interval,
            update_rank=1,
        )

    # ------------------------------------------------------------------ subspace

    @property
    def ortho_matrix(self):
        return self._lrp.ortho_matrix

    @torch.no_grad()
    def update_subspace(self, full_rank_grad, iter):
        """Advance the subspace using a batch gradient (bare, or privatized -- see header).

        ``LowRankProjector.project`` is what actually mutates ``ortho_matrix``: at
        ``iter == 0`` it takes the top-r singular vectors, and at ``iter % F == 0`` it
        either redoes the SVD (galore) or takes one rank-1 geodesic step (subtrack). The
        low-rank output is discarded -- the weight update is computed later from the
        privatized per-sample gradients.
        """
        if self.side is None:
            self.side = "right" if full_rank_grad.shape[0] >= full_rank_grad.shape[1] else "left"

        prev = None if self.ortho_matrix is None else self.ortho_matrix.detach().clone()
        self._lrp.project(full_rank_grad, iter)
        self._record_diagnostics(prev)

    def _basis(self, ortho):
        """Return the basis as a (d, rank) column-orthonormal matrix."""
        return ortho.t() if self.side == "right" else ortho

    @torch.no_grad()
    def _record_diagnostics(self, prev):
        """Largest principal angle vs the previous subspace, and orthonormality drift."""
        cur = self._basis(self.ortho_matrix.float())
        eye = torch.eye(cur.shape[1], device=cur.device, dtype=cur.dtype)
        self.last_ortho_err = torch.linalg.matrix_norm(cur.t() @ cur - eye).item()

        if prev is None:
            self.last_rotation_deg = None
            return
        sv = torch.linalg.svdvals(self._basis(prev.float()).t() @ cur)
        self.last_rotation_deg = torch.rad2deg(torch.arccos(sv.min().clamp(-1.0, 1.0))).item()

    # ------------------------------------------- RandProjectorDP-compatible interface

    def generate(self, full_rank_grad_shape, grad_dtype, grad_device):
        """No-op: the basis is stored, not regenerated. Records the projection side."""
        if self.side is None:
            self.side = "right" if full_rank_grad_shape[0] >= full_rank_grad_shape[1] else "left"

    def clear_projection_matrix(self):
        """No-op: clearing the stored basis would destroy the subspace."""

    def update_seed(self):
        """No-op: kept so DPOptimizerRandomProj.update_projectors stays callable."""

    def _oriented(self, reference):
        ortho = self.ortho_matrix
        if ortho is None:
            raise RuntimeError(
                "projector has no subspace yet -- update_subspace() must run before the "
                "first optimizer step (the trainer does this at global_step 0)"
            )
        return ortho.to(device=reference.device, dtype=reference.dtype)

    def project(self, full_rank_grad, batch=True):
        """Project (B, m, n) per-sample gradients, or a single (m, n) gradient."""
        ortho = self._oriented(full_rank_grad)
        if self.side == "right":
            return torch.matmul(full_rank_grad, ortho.t())     # (..., m, rank)
        return torch.matmul(ortho.t(), full_rank_grad)         # (..., rank, n)

    def project_back(self, low_rank_grad):
        """Map an (m, rank) / (rank, n) update back to (m, n).

        DPAdamW transposes the low-rank gradient so the larger dimension comes first, so
        the rank axis can arrive either way round; normalize it to be last.
        """
        ortho = self._oriented(low_rank_grad)
        low = low_rank_grad if low_rank_grad.shape[-1] == self.rank else low_rank_grad.t()
        if self.side == "right":
            full_rank_grad = torch.matmul(low, ortho)          # (m, rank) @ (rank, n)
        else:
            full_rank_grad = torch.matmul(ortho, low.t())      # (m, rank) @ (rank, n)
        return full_rank_grad * self.scale


def attach_low_rank_projectors(
    optimizer,
    model,
    method,
    subspace_update_interval,
    st_step_size=1.0,
    st_step_size_scheduler=None,
):
    """Install LowRankProjectorDP objects in place of DP-GRAPE's random projectors.

    Call once, right after ``privacy_engine.make_private``. The hooks bind the projector
    objects by reference at ``add_hooks()`` time, so later subspace updates -- which mutate
    the objects in place -- are picked up without re-registering anything.

    Args:
        optimizer: the DPOptimizerRandomProj returned by make_private.
        model: the ProjectedGradSampleModule returned by make_private.
        method: 'galore' or 'subtrack'.
        subspace_update_interval: F, must equal the trainer's subspace_T.
    """
    for group in optimizer.original_optimizer.param_groups:
        if "rank" not in group:
            continue
        for p in group["params"]:
            optimizer.original_optimizer.state[p]["projector"] = LowRankProjectorDP(
                rank=group["rank"],
                scale=group.get("scale", 1.0),
                proj_type=group.get("proj_type", "std"),
                method=method,
                subspace_update_interval=subspace_update_interval,
                st_step_size=st_step_size,
                st_step_size_scheduler=st_step_size_scheduler,
            )

    model.update_projectors(optimizer)
    model.remove_hooks(keep_ddp_hooks=True)
    model.add_hooks()


def iter_low_rank_projectors(optimizer):
    """Yield (param, projector) for every projected parameter."""
    for group in optimizer.original_optimizer.param_groups:
        if "rank" not in group:
            continue
        for p in group["params"]:
            state = optimizer.original_optimizer.state[p]
            if "projector" in state:
                yield p, state["projector"]
