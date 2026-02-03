"""
CMAAM (Cross-Modal Alignment-Aware Merging)

A novel approach for merging Multimodal Large Language Models (MLLMs) that considers
modality-specific characteristics for adaptive merging.

Key Components:
- ModalityComponent: Enum for classifying parameters by modality
- ComponentClassifier: Automatic parameter classification
- AlignmentScorer: Cross-modal alignment quality measurement
- SensitivityAnalyzer: Layer-wise modality sensitivity analysis
- CMAMerger: Adaptive merging with component/layer-specific alphas
"""

from .component_classifier import (
    ModalityComponent,
    ComponentClassifier,
    classify_parameter,
    get_component_groups,
)

from .alignment_scorer import (
    AlignmentScorer,
    compute_pda,
    compute_gfs,
    compute_cmbc,
)

from .sensitivity_analyzer import (
    SensitivityAnalyzer,
    compute_layer_sensitivity,
    rank_layers_by_importance,
)

from .adaptive_merger import (
    MergeStrategy,
    MergeConfig,
    CMAMerger,
    create_merger,
)

__all__ = [
    # Component Classifier
    "ModalityComponent",
    "ComponentClassifier",
    "classify_parameter",
    "get_component_groups",
    # Alignment Scorer
    "AlignmentScorer",
    "compute_pda",
    "compute_gfs",
    "compute_cmbc",
    # Sensitivity Analyzer
    "SensitivityAnalyzer",
    "compute_layer_sensitivity",
    "rank_layers_by_importance",
    # Adaptive Merger
    "MergeStrategy",
    "MergeConfig",
    "CMAMerger",
    "create_merger",
]

__version__ = "0.1.0"
