"""Stage 1 plugin implementations (one paper per file).

``STAGE1_PLUGIN_MANIFEST`` is the ordered plugin tuple the orchestrator
(``stage1/orchestrator.py``) feeds to
:class:`~moe_compress.pipeline.registry.PluginRegistry`. The orchestrator
invokes plugins by explicit, sequential ``run()`` calls — it does *not*
iterate this manifest to execute them — so the manifest order is kept
aligned with the orchestrator's call sequence by convention. Adding a new
plugin means dropping a file here, inserting a line in this tuple, AND
adding the explicit ``run()`` call in the orchestrator (see
``plugins/README.md``).
"""
from .ablation_filter import AblationFilterPlugin
from .aimer import AimerDetectorPlugin
from .cka_distance import CKADistancePlugin
from .damage_curve_dp import DamageCurveDpPlugin
from .grape_merge import GrapeMergePlugin
from .ma_detection import MADetectionPlugin
from .magnitude_topk import MagnitudeTopkPlugin
from .sink_token import SinkTokenDetectorPlugin
from .three_way_and import ThreeWayAndPlugin

# Ordered execution sequence — Phase A → Phase C (4 detectors) → D → E → F.
# DamageCurveDpPlugin (S1_DP) sits between Phase E (CKA) and Phase F
# (GRAPE) — it consumes D_matrices and populates merge_cost_prior into
# the in-ctx config so GRAPE's existing inert hook activates.
STAGE1_PLUGIN_MANIFEST = (
    MADetectionPlugin(),       # Phase A   — MA-formation detection
    ThreeWayAndPlugin(),       # Phase C₁  — three-way AND (mandatory)
    AimerDetectorPlugin(),     # Phase C₂  — AIMER bottom-pct
    SinkTokenDetectorPlugin(), # Phase C₃  — sink-token routing
    MagnitudeTopkPlugin(),     # Phase C₄  — magnitude top-K
    AblationFilterPlugin(),    # Phase D   — causal ΔNLL filter
    CKADistancePlugin(),       # Phase E   — CKA distance matrices
    DamageCurveDpPlugin(),     # Phase E.5 — S1_DP damage-curve + DP knapsack (optional, default OFF)
    GrapeMergePlugin(),        # Phase F   — GRAPE greedy merge
)

__all__ = [
    "STAGE1_PLUGIN_MANIFEST",
    "MADetectionPlugin", "ThreeWayAndPlugin", "AimerDetectorPlugin",
    "SinkTokenDetectorPlugin", "MagnitudeTopkPlugin",
    "AblationFilterPlugin", "CKADistancePlugin",
    "DamageCurveDpPlugin", "GrapeMergePlugin",
]
