"""Music-information-retrieval rewards and qualification utilities."""

from .composite_reward import (
    ComponentStats,
    FrozenComponentNormalizer,
    combine_components,
    online_group_zscore,
)
from .reward_safety import (
    PairAudit,
    audit_pairs,
    nuisance_pair_snr,
    provisional_safe_radius,
    risk_coverage,
)

__all__ = [
    "ComponentStats",
    "FrozenComponentNormalizer",
    "PairAudit",
    "audit_pairs",
    "combine_components",
    "nuisance_pair_snr",
    "online_group_zscore",
    "provisional_safe_radius",
    "risk_coverage",
]
