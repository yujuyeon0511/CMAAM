"""
Cross-Modal Alignment Scorer for CMAAM

Measures alignment quality between two models without requiring data.
Uses three key metrics:
- PDA (Parameter Distribution Alignment): Distribution similarity of parameters
- GFS (Gradient Flow Similarity): Information flow pattern similarity
- CMBC (Cross-Modal Bridge Coherence): Bridge layer alignment quality
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

try:
    from scipy.stats import wasserstein_distance
    from scipy.spatial.distance import cosine as cosine_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using torch-based distance metrics")

from .component_classifier import (
    ModalityComponent,
    ComponentClassifier,
    get_component_groups,
)


def _tensor_to_numpy(tensor: torch.Tensor, max_samples: int = 100000) -> np.ndarray:
    """
    Convert tensor to numpy array, with optional downsampling for large tensors.

    Args:
        tensor: Input tensor
        max_samples: Maximum number of samples to use

    Returns:
        Numpy array
    """
    flat = tensor.flatten().float().cpu()
    if len(flat) > max_samples:
        # Random sampling for large tensors
        indices = torch.randperm(len(flat))[:max_samples]
        flat = flat[indices]
    return flat.numpy()


def _torch_wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute 1D Wasserstein distance using PyTorch (fallback when scipy unavailable).

    Args:
        a, b: 1D tensors

    Returns:
        Wasserstein distance
    """
    a_sorted = torch.sort(a.flatten())[0]
    b_sorted = torch.sort(b.flatten())[0]

    # Interpolate to same length
    n = min(len(a_sorted), len(b_sorted), 10000)
    a_interp = torch.nn.functional.interpolate(
        a_sorted.unsqueeze(0).unsqueeze(0),
        size=n,
        mode='linear',
        align_corners=True
    ).squeeze()
    b_interp = torch.nn.functional.interpolate(
        b_sorted.unsqueeze(0).unsqueeze(0),
        size=n,
        mode='linear',
        align_corners=True
    ).squeeze()

    return torch.abs(a_interp - b_interp).mean().item()


def compute_pda(
    weights_a: Dict[str, torch.Tensor],
    weights_b: Dict[str, torch.Tensor],
    component: Optional[ModalityComponent] = None,
    model_type: str = "generic",
    max_samples_per_param: int = 50000,
) -> Dict[str, float]:
    """
    Compute Parameter Distribution Alignment (PDA) score.

    Measures how similar the parameter distributions are between two models.
    Higher scores indicate better alignment.

    Args:
        weights_a: State dict of model A
        weights_b: State dict of model B
        component: Optional specific component to analyze
        model_type: Model type for parameter classification
        max_samples_per_param: Max samples per parameter for distance calculation

    Returns:
        Dictionary with PDA scores per component and overall
    """
    classifier = ComponentClassifier(model_type)
    common_keys = set(weights_a.keys()) & set(weights_b.keys())

    if not common_keys:
        raise ValueError("No common parameters found between models")

    # Group parameters by component
    component_distances = defaultdict(list)

    for key in common_keys:
        param_component = classifier.classify(key)

        # Skip if filtering by specific component
        if component is not None and param_component != component:
            continue

        tensor_a = weights_a[key]
        tensor_b = weights_b[key]

        # Skip if shapes don't match
        if tensor_a.shape != tensor_b.shape:
            continue

        # Compute Wasserstein distance
        if SCIPY_AVAILABLE:
            arr_a = _tensor_to_numpy(tensor_a, max_samples_per_param)
            arr_b = _tensor_to_numpy(tensor_b, max_samples_per_param)
            try:
                dist = wasserstein_distance(arr_a, arr_b)
            except Exception:
                dist = _torch_wasserstein_1d(tensor_a, tensor_b)
        else:
            dist = _torch_wasserstein_1d(tensor_a, tensor_b)

        component_distances[param_component].append(dist)

    # Compute alignment scores (inverse of distance, normalized)
    pda_scores = {}

    for comp, distances in component_distances.items():
        if distances:
            mean_dist = np.mean(distances)
            # Convert distance to alignment score (0-1 range)
            # Using exponential decay: score = exp(-distance)
            pda_scores[comp.value] = float(np.exp(-mean_dist))

    # Compute overall score (weighted average by parameter count)
    total_params = sum(len(d) for d in component_distances.values())
    if total_params > 0:
        overall = sum(
            len(component_distances[comp]) * pda_scores.get(comp.value, 0)
            for comp in component_distances.keys()
        ) / total_params
        pda_scores["overall"] = float(overall)

    return pda_scores


def compute_gfs(
    weights_a: Dict[str, torch.Tensor],
    weights_b: Dict[str, torch.Tensor],
    model_type: str = "generic",
) -> Dict[str, float]:
    """
    Compute Gradient Flow Similarity (GFS) score.

    Approximates similarity in information flow patterns using parameter
    magnitudes and variances as proxies for gradient flow characteristics.

    Args:
        weights_a: State dict of model A
        weights_b: State dict of model B
        model_type: Model type for parameter classification

    Returns:
        Dictionary with GFS scores per component and overall
    """
    classifier = ComponentClassifier(model_type)
    common_keys = set(weights_a.keys()) & set(weights_b.keys())

    if not common_keys:
        raise ValueError("No common parameters found between models")

    # Compute layer-wise magnitude profiles
    component_magnitudes_a = defaultdict(list)
    component_magnitudes_b = defaultdict(list)
    layer_magnitudes_a = defaultdict(list)
    layer_magnitudes_b = defaultdict(list)

    for key in sorted(common_keys):
        param_component = classifier.classify(key)
        layer_num = classifier.extract_layer_number(key)

        tensor_a = weights_a[key]
        tensor_b = weights_b[key]

        if tensor_a.shape != tensor_b.shape:
            continue

        # Compute magnitude statistics
        mag_a = tensor_a.abs().mean().item()
        mag_b = tensor_b.abs().mean().item()
        var_a = tensor_a.var().item()
        var_b = tensor_b.var().item()

        # Combined flow indicator: magnitude * sqrt(variance)
        flow_a = mag_a * np.sqrt(abs(var_a) + 1e-10)
        flow_b = mag_b * np.sqrt(abs(var_b) + 1e-10)

        component_magnitudes_a[param_component].append(flow_a)
        component_magnitudes_b[param_component].append(flow_b)

        if layer_num is not None:
            layer_magnitudes_a[layer_num].append(flow_a)
            layer_magnitudes_b[layer_num].append(flow_b)

    # Compute cosine similarity for flow patterns
    gfs_scores = {}

    for comp in component_magnitudes_a.keys():
        vec_a = np.array(component_magnitudes_a[comp])
        vec_b = np.array(component_magnitudes_b[comp])

        if len(vec_a) > 0 and len(vec_b) > 0:
            # Normalize and compute cosine similarity
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)

            if norm_a > 1e-10 and norm_b > 1e-10:
                similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
                gfs_scores[comp.value] = float((similarity + 1) / 2)  # Map to 0-1
            else:
                gfs_scores[comp.value] = 0.5  # Neutral score for zero norms

    # Layer-wise similarity
    layer_similarities = []
    for layer_num in sorted(set(layer_magnitudes_a.keys()) & set(layer_magnitudes_b.keys())):
        vec_a = np.array(layer_magnitudes_a[layer_num])
        vec_b = np.array(layer_magnitudes_b[layer_num])

        if len(vec_a) > 0 and len(vec_b) > 0:
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a > 1e-10 and norm_b > 1e-10:
                sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
                layer_similarities.append((sim + 1) / 2)

    if layer_similarities:
        gfs_scores["layer_mean"] = float(np.mean(layer_similarities))
        gfs_scores["layer_std"] = float(np.std(layer_similarities))

    # Overall score
    if gfs_scores:
        gfs_scores["overall"] = float(np.mean([
            v for k, v in gfs_scores.items()
            if k not in ["layer_std"]
        ]))

    return gfs_scores


def compute_cmbc(
    weights_a: Dict[str, torch.Tensor],
    weights_b: Dict[str, torch.Tensor],
    model_type: str = "generic",
) -> Dict[str, float]:
    """
    Compute Cross-Modal Bridge Coherence (CMBC) score.

    Specifically analyzes the alignment quality of cross-modal bridge layers
    (mm_projector, adapter, etc.) which are critical for multimodal integration.

    Args:
        weights_a: State dict of model A
        weights_b: State dict of model B
        model_type: Model type for parameter classification

    Returns:
        Dictionary with CMBC scores and metrics
    """
    classifier = ComponentClassifier(model_type)
    groups_a = get_component_groups(weights_a, model_type)
    groups_b = get_component_groups(weights_b, model_type)

    bridge_keys_a = set(groups_a.get(ModalityComponent.CROSS_MODAL_BRIDGE, []))
    bridge_keys_b = set(groups_b.get(ModalityComponent.CROSS_MODAL_BRIDGE, []))

    common_bridge_keys = bridge_keys_a & bridge_keys_b

    if not common_bridge_keys:
        # No common bridge parameters found
        return {
            "coherence": 0.5,  # Neutral score
            "coverage": 0.0,
            "n_params": 0,
            "warning": "No common bridge parameters found"
        }

    # Analyze bridge layer alignment
    coherence_scores = []
    spectral_similarities = []

    for key in common_bridge_keys:
        tensor_a = weights_a[key]
        tensor_b = weights_b[key]

        if tensor_a.shape != tensor_b.shape:
            continue

        # 1. Direct parameter similarity (cosine)
        flat_a = tensor_a.flatten().float()
        flat_b = tensor_b.flatten().float()

        norm_a = torch.norm(flat_a)
        norm_b = torch.norm(flat_b)

        if norm_a > 1e-10 and norm_b > 1e-10:
            cos_sim = torch.dot(flat_a, flat_b) / (norm_a * norm_b)
            coherence_scores.append((cos_sim.item() + 1) / 2)

        # 2. Spectral similarity for weight matrices (2D+ tensors)
        if tensor_a.dim() >= 2:
            # Reshape to 2D if needed
            if tensor_a.dim() > 2:
                shape_2d = (tensor_a.shape[0], -1)
                mat_a = tensor_a.reshape(shape_2d).float()
                mat_b = tensor_b.reshape(shape_2d).float()
            else:
                mat_a = tensor_a.float()
                mat_b = tensor_b.float()

            try:
                # Compare singular value distributions
                # Use only top-k singular values for efficiency
                k = min(min(mat_a.shape), 50)

                # SVD with error handling
                try:
                    _, s_a, _ = torch.svd_lowrank(mat_a, q=k)
                    _, s_b, _ = torch.svd_lowrank(mat_b, q=k)
                except Exception:
                    # Fallback to eigenvalue comparison
                    s_a = torch.linalg.eigvalsh(mat_a @ mat_a.T)[-k:].abs().sqrt()
                    s_b = torch.linalg.eigvalsh(mat_b @ mat_b.T)[-k:].abs().sqrt()

                # Normalize and compare
                s_a = s_a / (s_a.sum() + 1e-10)
                s_b = s_b / (s_b.sum() + 1e-10)

                # KL divergence-based similarity
                kl_div = torch.nn.functional.kl_div(
                    (s_a + 1e-10).log(),
                    s_b + 1e-10,
                    reduction='sum'
                ).item()
                spectral_sim = np.exp(-abs(kl_div))
                spectral_similarities.append(spectral_sim)

            except Exception:
                # Skip spectral analysis on error
                pass

    # Compute final scores
    cmbc_scores = {}

    if coherence_scores:
        cmbc_scores["coherence"] = float(np.mean(coherence_scores))
        cmbc_scores["coherence_std"] = float(np.std(coherence_scores))

    if spectral_similarities:
        cmbc_scores["spectral"] = float(np.mean(spectral_similarities))

    # Coverage: how many bridge params are shared
    total_bridge = len(bridge_keys_a | bridge_keys_b)
    cmbc_scores["coverage"] = len(common_bridge_keys) / total_bridge if total_bridge > 0 else 0
    cmbc_scores["n_params"] = len(common_bridge_keys)

    # Overall CMBC score
    if coherence_scores:
        overall = cmbc_scores["coherence"]
        if spectral_similarities:
            overall = 0.7 * overall + 0.3 * cmbc_scores["spectral"]
        cmbc_scores["overall"] = float(overall)
    else:
        cmbc_scores["overall"] = 0.5

    return cmbc_scores


class AlignmentScorer:
    """
    Main class for computing cross-modal alignment scores between models.

    Combines PDA, GFS, and CMBC metrics into comprehensive alignment analysis.
    """

    def __init__(
        self,
        model_type: str = "generic",
        weights_pda: float = 0.4,
        weights_gfs: float = 0.3,
        weights_cmbc: float = 0.3,
    ):
        """
        Initialize the AlignmentScorer.

        Args:
            model_type: Model type for parameter classification
            weights_pda: Weight for PDA score in overall computation
            weights_gfs: Weight for GFS score in overall computation
            weights_cmbc: Weight for CMBC score in overall computation
        """
        self.model_type = model_type
        self.weights = {
            "pda": weights_pda,
            "gfs": weights_gfs,
            "cmbc": weights_cmbc,
        }
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        self.classifier = ComponentClassifier(model_type)

    def compute_all_scores(
        self,
        weights_a: Dict[str, torch.Tensor],
        weights_b: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute all alignment scores between two models.

        Args:
            weights_a: State dict of model A (source)
            weights_b: State dict of model B (target)
            verbose: Whether to print detailed results

        Returns:
            Dictionary containing all metrics
        """
        results = {}

        # Compute individual metrics
        if verbose:
            print("Computing Parameter Distribution Alignment (PDA)...")
        results["pda"] = compute_pda(weights_a, weights_b, model_type=self.model_type)

        if verbose:
            print("Computing Gradient Flow Similarity (GFS)...")
        results["gfs"] = compute_gfs(weights_a, weights_b, model_type=self.model_type)

        if verbose:
            print("Computing Cross-Modal Bridge Coherence (CMBC)...")
        results["cmbc"] = compute_cmbc(weights_a, weights_b, model_type=self.model_type)

        # Compute weighted overall score
        overall = (
            self.weights["pda"] * results["pda"].get("overall", 0.5) +
            self.weights["gfs"] * results["gfs"].get("overall", 0.5) +
            self.weights["cmbc"] * results["cmbc"].get("overall", 0.5)
        )
        results["overall"] = {"score": float(overall)}

        # Component-wise combined scores
        component_scores = {}
        for comp in ModalityComponent:
            comp_score = (
                self.weights["pda"] * results["pda"].get(comp.value, 0.5) +
                self.weights["gfs"] * results["gfs"].get(comp.value, 0.5)
            )
            if comp == ModalityComponent.CROSS_MODAL_BRIDGE:
                comp_score = (
                    0.5 * comp_score +
                    0.5 * results["cmbc"].get("overall", 0.5)
                )
            component_scores[comp.value] = float(comp_score)

        results["component_scores"] = component_scores

        if verbose:
            self._print_results(results)

        return results

    def _print_results(self, results: Dict) -> None:
        """Print formatted alignment results."""
        print(f"\n{'='*60}")
        print("Cross-Modal Alignment Analysis Results")
        print(f"{'='*60}")

        print(f"\nOverall Alignment Score: {results['overall']['score']:.4f}")

        print(f"\n{'Component Scores':^60}")
        print("-" * 60)
        for comp, score in results["component_scores"].items():
            print(f"  {comp:20s}: {score:.4f}")

        print(f"\n{'Detailed Metrics':^60}")
        print("-" * 60)

        print("\nPDA (Parameter Distribution Alignment):")
        for key, value in results["pda"].items():
            print(f"  {key:20s}: {value:.4f}")

        print("\nGFS (Gradient Flow Similarity):")
        for key, value in results["gfs"].items():
            print(f"  {key:20s}: {value:.4f}")

        print("\nCMBC (Cross-Modal Bridge Coherence):")
        for key, value in results["cmbc"].items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
            else:
                print(f"  {key:20s}: {value}")

        print(f"\n{'='*60}\n")

    def get_merge_recommendations(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Generate recommended merge weights based on alignment scores.

        Args:
            results: Alignment score results from compute_all_scores

        Returns:
            Recommended alpha values per component
        """
        recommendations = {}

        # Base alpha is proportional to overall alignment
        base_alpha = results["overall"]["score"]

        for comp in ModalityComponent:
            comp_score = results["component_scores"].get(comp.value, 0.5)

            # Higher alignment -> higher merge ratio
            # Scale around base_alpha
            rec_alpha = base_alpha * comp_score / 0.5  # Normalize around 0.5

            # Clamp to valid range
            rec_alpha = max(0.1, min(0.9, rec_alpha))

            recommendations[comp.value] = rec_alpha

        # Special handling for bridge layers
        # Be more conservative if CMBC is low
        cmbc_overall = results["cmbc"].get("overall", 0.5)
        if cmbc_overall < 0.4:
            recommendations["bridge"] = min(recommendations.get("bridge", 0.5), 0.3)

        recommendations["base"] = base_alpha

        return recommendations


if __name__ == "__main__":
    # Demo/test code
    import torch

    print("Testing AlignmentScorer...")

    # Create mock weights for testing
    torch.manual_seed(42)

    def create_mock_weights(seed: int = 0):
        torch.manual_seed(seed)
        return {
            # Vision encoder
            "visual.patch_embed.proj.weight": torch.randn(768, 3, 14, 14),
            "visual.blocks.0.attn.qkv.weight": torch.randn(2304, 768),
            # Bridge
            "visual.merger.mlp.0.weight": torch.randn(4096, 768),
            "visual.merger.mlp.0.bias": torch.randn(4096),
            # Shared
            "model.embed_tokens.weight": torch.randn(151936, 4096),
            "model.norm.weight": torch.randn(4096),
            "lm_head.weight": torch.randn(151936, 4096),
            # Language model
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(1024, 4096),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(1024, 4096),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(11008, 4096),
            "model.layers.0.mlp.up_proj.weight": torch.randn(11008, 4096),
            "model.layers.0.mlp.down_proj.weight": torch.randn(4096, 11008),
        }

    # Create two models with related but different weights
    weights_a = create_mock_weights(42)
    weights_b = {k: v + 0.1 * torch.randn_like(v) for k, v in weights_a.items()}

    # Test scorer
    scorer = AlignmentScorer(model_type="qwen2vl")
    results = scorer.compute_all_scores(weights_a, weights_b, verbose=True)

    # Get recommendations
    recommendations = scorer.get_merge_recommendations(results)
    print("Merge Recommendations:")
    for comp, alpha in recommendations.items():
        print(f"  {comp}: alpha = {alpha:.3f}")
