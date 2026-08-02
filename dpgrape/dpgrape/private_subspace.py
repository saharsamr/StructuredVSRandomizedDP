# The DP mechanism behind --oracle_batch_mode private-skip.
#
# DP-GaLore / DPTrack-Oracle estimate their subspace from a BARE batch gradient, which makes
# the run eps = infinity (DPTRACK_DESIGN.md section 4.1). private-skip replaces that bare
# gradient with a proper Gaussian-mechanism release, so the subspace becomes post-processing
# of a DP output and the run is finite-eps again.
#
# The release, per subspace update, is the concatenation over the L projected weight matrices
#
#     v = ( sum_i clip_{C_l}(g_i^1) , ... , sum_i clip_{C_l}(g_i^L) ),    C_l = C / sqrt(L)
#
# plus spherical Gaussian noise, divided by the (public, fixed) batch size.
#
# !!! NOISE SCALE -- the one thing that is easy to get wrong !!!
# Adding or removing one sample moves v by (clip(g^1), ..., clip(g^L)), whose L2 norm is at
# most sqrt(L * C_l^2) = C. So the noise that buys a noise multiplier sigma is
#
#     std = sigma * C          per coordinate, the same in every layer
#
# NOT ``sigma * C_l``. Opacus' per-layer optimizer uses the ``sigma * C_l`` convention, and
# for a *joint* release that is under-noised by sqrt(L): the privacy loss of a Gaussian with
# block-diagonal covariance is governed by ||Sigma^{-1/2} delta||, which for per-layer noise
# sigma*C_l and per-layer sensitivity C_l is sqrt(L)/sigma -- an effective noise multiplier of
# sigma/sqrt(L), i.e. 12x too small at L = 144 for RoBERTa-Large. Do not "simplify" the std
# below to track ``per_layer_clip``.
#
# Per-layer clipping (rather than one flat clip across all L matrices) is what the caller
# asked for. It costs some bias -- a high-energy layer cannot borrow budget from a quiet one
# -- and buys nothing in noise, since the total sensitivity is C either way.

import math

import torch


class PerLayerGaussianMechanism:
    """Per-layer-clipped, noised sum of per-sample gradients over a batch.

    Usage per subspace update::

        mech.reset()
        for each sample:
            <backward on that one sample>      # p.grad is then that sample's gradient
            mech.accumulate()
        grads = mech.release()                 # {param: privatized mean gradient}

    ``accumulate`` reads ``p.grad``, so the caller must arrange for it to hold a single
    sample's gradient -- in practice a microbatch-of-one backward with opacus' hooks
    disabled. Anything larger would clip a microbatch *mean*, which bounds nothing.

    Args:
        params: the projected weight matrices, i.e. exactly the tensors whose privatized
            gradients get released. L is ``len(params)``. Gradients of any other parameter
            are never read and never released, so they cost no privacy.
        clip_threshold: C, the L2 bound on one sample's contribution across all L matrices.
        noise_multiplier: sigma, from the accountant for *this* mechanism's step count and
            sampling rate -- not the training loop's.
        expected_batch_size: the divisor. Public and fixed (the loader uses drop_last), so
            it leaks nothing; using the observed count instead would.
        generator: optional torch.Generator for reproducible noise.
    """

    def __init__(
        self,
        params,
        clip_threshold,
        noise_multiplier,
        expected_batch_size,
        generator=None,
    ):
        self.params = list(params)
        if not self.params:
            raise ValueError("no projected parameters to privatize")
        if clip_threshold <= 0:
            raise ValueError(f"clip_threshold must be > 0, got {clip_threshold}")
        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier must be >= 0, got {noise_multiplier}")

        self.num_layers = len(self.params)
        self.clip_threshold = clip_threshold
        self.per_layer_clip = clip_threshold / math.sqrt(self.num_layers)
        self.noise_multiplier = noise_multiplier
        self.expected_batch_size = expected_batch_size
        self.generator = generator

        self._sums = None
        self.num_samples = 0
        # Diagnostics: these are what tell you whether C is set anywhere near right.
        self.sum_sample_norm = 0.0     # sum over samples of the total pre-clip norm
        self.max_sample_norm = 0.0
        self.num_clipped = 0           # (sample, layer) pairs where the clip bound

    @property
    def noise_std(self):
        """Per-coordinate std of the noise added to the *sum*. See the module header."""
        return self.noise_multiplier * self.clip_threshold

    @property
    def released_noise_std(self):
        """Per-coordinate std after dividing by the batch size -- 's' in DPTRACK_DESIGN section 5."""
        return self.noise_std / self.expected_batch_size

    def reset(self):
        self._sums = {p: torch.zeros_like(p, dtype=torch.float32) for p in self.params}
        self.num_samples = 0
        self.sum_sample_norm = 0.0
        self.max_sample_norm = 0.0
        self.num_clipped = 0

    @torch.no_grad()
    def accumulate(self):
        """Clip this sample's gradient per layer and add it to the running sums."""
        if self._sums is None:
            raise RuntimeError("call reset() before accumulate()")

        sq_total = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            norm = torch.linalg.vector_norm(g).item()
            sq_total += norm * norm
            # clamp, not min(1, C/norm): at norm == 0 the ratio is inf, and a zero-gradient
            # layer must contribute zero rather than NaN.
            factor = min(1.0, self.per_layer_clip / norm) if norm > 0 else 0.0
            if factor < 1.0:
                self.num_clipped += 1
            self._sums[p].add_(g, alpha=factor)

        sample_norm = math.sqrt(sq_total)
        self.sum_sample_norm += sample_norm
        self.max_sample_norm = max(self.max_sample_norm, sample_norm)
        self.num_samples += 1

    @torch.no_grad()
    def release(self):
        """Add noise and divide by the batch size. Returns {param: (m, n) float32 tensor}."""
        if self._sums is None:
            raise RuntimeError("call reset() before release()")
        if self.num_samples != self.expected_batch_size:
            raise RuntimeError(
                f"accumulated {self.num_samples} samples but the accountant was calibrated "
                f"for a fixed batch of {self.expected_batch_size}; a short batch would make "
                "the sampling rate wrong. Use drop_last=True on the subspace loader."
            )

        out = {}
        for p, summed in self._sums.items():
            noise = torch.normal(
                mean=0.0,
                std=self.noise_std,
                size=summed.shape,
                device=summed.device,
                dtype=summed.dtype,
                generator=self.generator,
            )
            out[p] = (summed + noise) / self.expected_batch_size
        self._sums = None
        return out

    def diagnostics(self):
        """Numbers worth logging every update. Safe to call after release()."""
        n = max(self.num_samples, 1)
        return {
            "num_layers": self.num_layers,
            "clip_threshold": self.clip_threshold,
            "per_layer_clip": self.per_layer_clip,
            "noise_multiplier": self.noise_multiplier,
            "released_noise_std": self.released_noise_std,
            "mean_sample_norm_pre_clip": self.sum_sample_norm / n,
            "max_sample_norm_pre_clip": self.max_sample_norm,
            "clipped_fraction": self.num_clipped / (n * self.num_layers),
        }
