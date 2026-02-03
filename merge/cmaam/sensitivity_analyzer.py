"""
Modality Sensitivity Analyzer for CMAAM

Analyzes how sensitive each layer/component is to vision vs language modalities.
This information is used to determine appropriate merge weights:
- High vision sensitivity: Be careful merging vision-related weights
- High language sensitivity: Be careful merging language-related weights
- Balanced: Can merge more aggressively
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from dataclasses import dataclass

from .component_classifier import (
    ModalityComponent,
    ComponentClassifier,
)


@dataclass
class LayerSensitivity:
    """Dataclass to hold layer sensitivity information."""
    layer_name: str
    layer_num: Optional[int]
    component: ModalityComponent
    vision_sensitivity: float
    language_sensitivity: float
    importance_score: float
    magnitude: float
    variance: float

    @property
    def modality_balance(self) -> float:
        """Returns balance ratio: 1.0 = balanced, <1.0 = vision heavy, >1.0 = language heavy"""
        if self.vision_sensitivity < 1e-10:
            return float('inf')
        return self.language_sensitivity / self.vision_sensitivity

    @property
    def dominant_modality(self) -> str:
        """Returns the dominant modality for this layer."""
        if abs(self.vision_sensitivity - self.language_sensitivity) < 0.1:
            return "balanced"
        return "vision" if self.vision_sensitivity > self.language_sensitivity else "language"


def compute_magnitude_statistics(
    tensor: torch.Tensor
) -> Dict[str, float]:
    """
    Compute magnitude-based statistics for a tensor.

    Args:
        tensor: Input tensor

    Returns:
        Dictionary with magnitude statistics
    """
    flat = tensor.flatten().float()

    return {
        "mean": flat.abs().mean().item(),
        "std": flat.std().item(),
        "max": flat.abs().max().item(),
        "l1_norm": flat.abs().sum().item(),
        "l2_norm": flat.norm(2).item(),
        "sparsity": (flat.abs() < 1e-6).float().mean().item(),
    }


def compute_layer_sensitivity(
    layer_weights: Dict[str, torch.Tensor],
    reference_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Compute modality sensitivity for a single layer.

    Sensitivity is estimated based on:
    1. Weight magnitude patterns (attention vs MLP)
    2. Variance distribution (higher variance = more specialized)
    3. Comparison with reference weights if provided

    Args:
        layer_weights: Dictionary of weights for a single layer
        reference_weights: Optional reference weights for comparison

    Returns:
        Dictionary with sensitivity scores
    """
    # Analyze attention components (often more vision-related in MLLMs)
    attn_stats = []
    mlp_stats = []
    other_stats = []

    for name, tensor in layer_weights.items():
        stats = compute_magnitude_statistics(tensor)
        stats["name"] = name

        if any(k in name.lower() for k in ["q_proj", "k_proj", "v_proj", "o_proj", "attn"]):
            attn_stats.append(stats)
        elif any(k in name.lower() for k in ["mlp", "gate", "up_proj", "down_proj", "fc"]):
            mlp_stats.append(stats)
        else:
            other_stats.append(stats)

    # Compute sensitivity based on relative magnitudes
    attn_magnitude = np.mean([s["l2_norm"] for s in attn_stats]) if attn_stats else 0
    mlp_magnitude = np.mean([s["l2_norm"] for s in mlp_stats]) if mlp_stats else 0
    total_magnitude = attn_magnitude + mlp_magnitude + 1e-10

    # Attention layers tend to be more important for visual processing
    # MLP layers tend to be more important for language processing
    # This is a heuristic based on typical MLLM architectures

    # Normalize to [0, 1]
    vision_sensitivity = attn_magnitude / total_magnitude
    language_sensitivity = mlp_magnitude / total_magnitude

    # Compute variance-based sensitivity (higher variance = more specialized)
    attn_variance = np.mean([s["std"] for s in attn_stats]) if attn_stats else 0
    mlp_variance = np.mean([s["std"] for s in mlp_stats]) if mlp_stats else 0

    # Adjust sensitivity based on variance
    variance_factor = attn_variance / (mlp_variance + 1e-10)
    if variance_factor > 1:
        vision_sensitivity *= min(1.5, variance_factor)
    elif variance_factor > 1e-10:  # Avoid division by zero
        language_sensitivity *= min(1.5, 1/variance_factor)

    # Renormalize
    total = vision_sensitivity + language_sensitivity + 1e-10
    vision_sensitivity /= total
    language_sensitivity /= total

    # Compute delta if reference provided
    delta_sensitivity = 0.0
    if reference_weights is not None:
        common_keys = set(layer_weights.keys()) & set(reference_weights.keys())
        deltas = []
        for key in common_keys:
            if layer_weights[key].shape == reference_weights[key].shape:
                delta = (layer_weights[key] - reference_weights[key]).abs().mean().item()
                deltas.append(delta)
        if deltas:
            delta_sensitivity = np.mean(deltas)

    return {
        "vision_sensitivity": float(vision_sensitivity),
        "language_sensitivity": float(language_sensitivity),
        "attn_magnitude": float(attn_magnitude),
        "mlp_magnitude": float(mlp_magnitude),
        "attn_variance": float(attn_variance),
        "mlp_variance": float(mlp_variance),
        "delta_sensitivity": float(delta_sensitivity),
    }


def compute_fisher_approximation(
    weights: Dict[str, torch.Tensor],
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    Approximate Fisher information using parameter statistics.

    Fisher information indicates parameter importance - higher Fisher
    means the parameter is more important for the model's output.

    This is an approximation that doesn't require actual gradients:
    Fisher ≈ E[g²] ≈ parameter_magnitude × inverse_variance

    Args:
        weights: Model weights
        epsilon: Small value for numerical stability

    Returns:
        Dictionary mapping parameter names to Fisher approximations
    """
    fisher_scores = {}

    for name, tensor in weights.items():
        flat = tensor.flatten().float()

        # Approximation: magnitude * (1 / variance)
        # High magnitude with low variance = potentially important
        magnitude = flat.abs().mean().item()
        variance = flat.var().item() + epsilon

        # Alternative: use L2 norm based approximation
        l2_norm = flat.norm(2).item()
        n_params = len(flat)

        # Combined score
        fisher_approx = l2_norm / np.sqrt(n_params) / np.sqrt(variance)
        fisher_scores[name] = float(fisher_approx)

    return fisher_scores


def rank_layers_by_importance(
    model_weights: Dict[str, torch.Tensor],
    model_type: str = "generic",
    method: str = "fisher",
) -> List[Tuple[str, float]]:
    """
    Rank layers by their importance for merging decisions.

    More important layers should be merged more conservatively.

    Args:
        model_weights: Full model state dict
        model_type: Model type for parameter classification
        method: Ranking method ("fisher", "magnitude", "variance")

    Returns:
        List of (layer_key, importance_score) tuples, sorted by importance
    """
    classifier = ComponentClassifier(model_type)
    layer_groups = classifier.group_by_layer(list(model_weights.keys()))

    layer_importance = []

    for layer_num, param_names in layer_groups.items():
        if layer_num is None:
            continue

        layer_weights = {k: model_weights[k] for k in param_names}

        if method == "fisher":
            fisher_scores = compute_fisher_approximation(layer_weights)
            importance = np.mean(list(fisher_scores.values()))
        elif method == "magnitude":
            magnitudes = [w.abs().mean().item() for w in layer_weights.values()]
            importance = np.mean(magnitudes)
        elif method == "variance":
            variances = [w.var().item() for w in layer_weights.values()]
            importance = np.mean(variances)
        else:
            raise ValueError(f"Unknown method: {method}")

        layer_importance.append((f"layer_{layer_num}", importance))

    # Sort by importance (descending)
    layer_importance.sort(key=lambda x: x[1], reverse=True)

    return layer_importance


class SensitivityAnalyzer:
    """
    Main class for analyzing modality sensitivity across a model.

    Provides comprehensive analysis of how each component and layer
    responds to visual vs language modalities.
    """

    def __init__(self, model_type: str = "generic"):
        """
        Initialize the SensitivityAnalyzer.

        Args:
            model_type: Model type for parameter classification
        """
        self.model_type = model_type
        self.classifier = ComponentClassifier(model_type)

    def analyze_model(
        self,
        weights: Dict[str, torch.Tensor],
        reference_weights: Optional[Dict[str, torch.Tensor]] = None,
        verbose: bool = False,
    ) -> Dict[str, LayerSensitivity]:
        """
        Perform comprehensive sensitivity analysis on a model.

        Args:
            weights: Model state dict to analyze
            reference_weights: Optional reference weights for comparison
            verbose: Whether to print detailed results

        Returns:
            Dictionary mapping layer keys to LayerSensitivity objects
        """
        layer_groups = self.classifier.group_by_layer(list(weights.keys()))
        results = {}

        for layer_num, param_names in layer_groups.items():
            layer_key = f"layer_{layer_num}" if layer_num is not None else "non_layer"
            layer_weights = {k: weights[k] for k in param_names}

            # Get reference weights for this layer if available
            ref_layer_weights = None
            if reference_weights is not None:
                ref_layer_weights = {
                    k: reference_weights[k]
                    for k in param_names
                    if k in reference_weights
                }

            # Compute sensitivity
            sensitivity = compute_layer_sensitivity(layer_weights, ref_layer_weights)

            # Compute importance
            fisher_scores = compute_fisher_approximation(layer_weights)
            importance = np.mean(list(fisher_scores.values()))

            # Get component (use first param as representative)
            component = self.classifier.classify(param_names[0])

            # Compute overall magnitude and variance
            all_tensors = torch.cat([w.flatten() for w in layer_weights.values()])
            magnitude = all_tensors.abs().mean().item()
            variance = all_tensors.var().item()

            results[layer_key] = LayerSensitivity(
                layer_name=layer_key,
                layer_num=layer_num,
                component=component,
                vision_sensitivity=sensitivity["vision_sensitivity"],
                language_sensitivity=sensitivity["language_sensitivity"],
                importance_score=importance,
                magnitude=magnitude,
                variance=variance,
            )

        if verbose:
            self._print_analysis(results)

        return results

    def analyze_component_sensitivity(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> Dict[ModalityComponent, Dict[str, float]]:
        """
        Analyze sensitivity grouped by modality component.

        Args:
            weights: Model state dict

        Returns:
            Dictionary mapping components to their sensitivity metrics
        """
        component_groups = self.classifier.group_by_component(list(weights.keys()))
        results = {}

        for component, param_names in component_groups.items():
            component_weights = {k: weights[k] for k in param_names}

            # Aggregate statistics
            magnitudes = []
            variances = []
            fisher_scores = compute_fisher_approximation(component_weights)

            for tensor in component_weights.values():
                magnitudes.append(tensor.abs().mean().item())
                variances.append(tensor.var().item())

            results[component] = {
                "mean_magnitude": np.mean(magnitudes),
                "std_magnitude": np.std(magnitudes),
                "mean_variance": np.mean(variances),
                "mean_fisher": np.mean(list(fisher_scores.values())),
                "n_params": len(param_names),
            }

        return results

    def compute_merge_weights(
        self,
        source_weights: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor],
        base_alpha: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute recommended merge weights based on sensitivity analysis.

        Args:
            source_weights: Source model weights
            target_weights: Target model weights
            base_alpha: Base merge coefficient

        Returns:
            Dictionary mapping parameter names to recommended alphas
        """
        # Analyze both models
        source_analysis = self.analyze_model(source_weights, verbose=False)
        target_analysis = self.analyze_model(target_weights, verbose=False)

        merge_weights = {}
        layer_groups = self.classifier.group_by_layer(list(source_weights.keys()))

        for layer_num, param_names in layer_groups.items():
            layer_key = f"layer_{layer_num}" if layer_num is not None else "non_layer"

            # Get sensitivity info
            source_sens = source_analysis.get(layer_key)
            target_sens = target_analysis.get(layer_key)

            if source_sens is None or target_sens is None:
                # Default to base alpha
                for name in param_names:
                    merge_weights[name] = base_alpha
                continue

            # Compute adaptive alpha based on:
            # 1. Importance: More important layers get lower alpha (more conservative)
            # 2. Sensitivity variance: High variance in sensitivity gets lower alpha
            # 3. Modality balance: Balanced layers can be merged more aggressively

            importance_factor = 1 - min(1, source_sens.importance_score / 10)  # Normalize
            balance_factor = 1 - abs(source_sens.vision_sensitivity - 0.5)  # Peak at balanced

            # Sensitivity variance between source and target
            sens_diff = abs(
                source_sens.vision_sensitivity - target_sens.vision_sensitivity
            )
            alignment_factor = 1 - sens_diff

            # Combine factors
            layer_alpha = base_alpha * (
                0.4 * importance_factor +
                0.3 * balance_factor +
                0.3 * alignment_factor
            )

            # Clamp to reasonable range
            layer_alpha = max(0.1, min(0.9, layer_alpha))

            for name in param_names:
                merge_weights[name] = layer_alpha

        return merge_weights

    def get_layer_ranking(
        self,
        weights: Dict[str, torch.Tensor],
        method: str = "fisher",
    ) -> List[Tuple[str, float]]:
        """
        Get layer importance ranking.

        Args:
            weights: Model weights
            method: Ranking method

        Returns:
            Sorted list of (layer_key, importance) tuples
        """
        return rank_layers_by_importance(weights, self.model_type, method)

    def _print_analysis(self, results: Dict[str, LayerSensitivity]) -> None:
        """Print formatted sensitivity analysis results."""
        print(f"\n{'='*80}")
        print("Modality Sensitivity Analysis")
        print(f"{'='*80}")

        # Sort by layer number
        sorted_results = sorted(
            results.items(),
            key=lambda x: (x[1].layer_num if x[1].layer_num is not None else -1)
        )

        print(f"\n{'Layer':<15} {'Component':<12} {'Vision':<10} {'Language':<10} "
              f"{'Balance':<12} {'Importance':<12}")
        print("-" * 80)

        for layer_key, sens in sorted_results:
            balance_str = sens.dominant_modality
            print(f"{layer_key:<15} {sens.component.value:<12} "
                  f"{sens.vision_sensitivity:<10.4f} {sens.language_sensitivity:<10.4f} "
                  f"{balance_str:<12} {sens.importance_score:<12.4f}")

        # Summary statistics
        vision_sensitivities = [s.vision_sensitivity for s in results.values()]
        language_sensitivities = [s.language_sensitivity for s in results.values()]
        importances = [s.importance_score for s in results.values()]

        print(f"\n{'Summary Statistics':^80}")
        print("-" * 80)
        print(f"Vision Sensitivity:   mean={np.mean(vision_sensitivities):.4f}, "
              f"std={np.std(vision_sensitivities):.4f}")
        print(f"Language Sensitivity: mean={np.mean(language_sensitivities):.4f}, "
              f"std={np.std(language_sensitivities):.4f}")
        print(f"Importance Scores:    mean={np.mean(importances):.4f}, "
              f"std={np.std(importances):.4f}")

        # Count dominant modalities
        dominant_counts = defaultdict(int)
        for sens in results.values():
            dominant_counts[sens.dominant_modality] += 1

        print(f"\nDominant Modality Distribution:")
        for mod, count in dominant_counts.items():
            print(f"  {mod}: {count} layers ({count/len(results)*100:.1f}%)")

        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Demo/test code
    import torch

    print("Testing SensitivityAnalyzer...")

    # Create mock weights
    torch.manual_seed(42)

    def create_mock_layer_weights(layer_num: int, vision_bias: float = 0.5):
        """Create mock weights for a single layer with configurable vision bias."""
        base_size = 4096

        # Attention weights (vision-related)
        attn_scale = 1.0 + (vision_bias - 0.5) * 0.5

        # MLP weights (language-related)
        mlp_scale = 1.0 + (0.5 - vision_bias) * 0.5

        return {
            f"model.layers.{layer_num}.self_attn.q_proj.weight":
                torch.randn(base_size, base_size) * attn_scale,
            f"model.layers.{layer_num}.self_attn.k_proj.weight":
                torch.randn(base_size // 4, base_size) * attn_scale,
            f"model.layers.{layer_num}.self_attn.v_proj.weight":
                torch.randn(base_size // 4, base_size) * attn_scale,
            f"model.layers.{layer_num}.self_attn.o_proj.weight":
                torch.randn(base_size, base_size) * attn_scale,
            f"model.layers.{layer_num}.mlp.gate_proj.weight":
                torch.randn(base_size * 2, base_size) * mlp_scale,
            f"model.layers.{layer_num}.mlp.up_proj.weight":
                torch.randn(base_size * 2, base_size) * mlp_scale,
            f"model.layers.{layer_num}.mlp.down_proj.weight":
                torch.randn(base_size, base_size * 2) * mlp_scale,
        }

    # Create a model with varying vision/language bias across layers
    mock_weights = {}
    for i in range(4):
        # Early layers more vision-biased, later layers more language-biased
        vision_bias = 0.7 - i * 0.1
        mock_weights.update(create_mock_layer_weights(i, vision_bias))

    # Add shared embedding weights
    mock_weights["model.embed_tokens.weight"] = torch.randn(32000, 4096)
    mock_weights["lm_head.weight"] = torch.randn(32000, 4096)
    mock_weights["model.norm.weight"] = torch.randn(4096)

    # Test analyzer
    analyzer = SensitivityAnalyzer(model_type="qwen2vl")

    # Analyze model
    results = analyzer.analyze_model(mock_weights, verbose=True)

    # Get layer ranking
    print("\nLayer Importance Ranking:")
    ranking = analyzer.get_layer_ranking(mock_weights)
    for layer_key, importance in ranking[:5]:
        print(f"  {layer_key}: {importance:.4f}")

    # Component sensitivity
    print("\nComponent Sensitivity Analysis:")
    comp_sens = analyzer.analyze_component_sensitivity(mock_weights)
    for comp, metrics in comp_sens.items():
        print(f"\n{comp.value}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
