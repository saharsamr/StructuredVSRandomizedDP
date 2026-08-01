# Verbatim copy of low_rank_projector.py from the SubTrack++ / GaLore repo.
# Only the projector is vendored here -- the SubTrack++ AdamW optimizers are not used;
# weight updates go through dpgrape.dpadamw.DPAdamW so that DP-GaLore / DPTrack-Oracle
# are byte-for-byte identical to DP-GRAPE everywhere except the projector.

from .low_rank_projector import LowRankProjector

__all__ = ["LowRankProjector"]
