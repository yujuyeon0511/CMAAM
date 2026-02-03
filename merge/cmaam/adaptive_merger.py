"""
Adaptive Merger for CMAAM

Implements the core merging algorithm that adapts merge coefficients (alphas)
based on:
- Cross-modal alignment scores
- Modality sensitivity analysis
- Component/layer-specific characteristics

Supports multiple merge strategies:
- BASIC: Simple adaptive alpha per component
- LAYERWISE: Different alpha for each layer
- COMPONENT: Different alpha for each modality component
- FULL: Layer + component combination
"""

import torch
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from dataclasses import dataclass
import json
import os

from .component_classifier import (
    ModalityComponent,
    ComponentClassifier,
    get_component_groups,
)
from .alignment_scorer import AlignmentScorer
from .sensitivity_analyzer import SensitivityAnalyzer


class MergeStrategy(Enum):
    """Merge strategy options for CMAAM."""
    BASIC = "basic"              # Basic adaptive merging
    LAYERWISE = "layerwise"      # Layer-specific alphas
    COMPONENT = "component"      # Component-specific alphas (Vision/Bridge/Language)
    FULL = "full"                # Full: layer + component combination


@dataclass
class MergeConfig:
    """Configuration for the merge operation."""
    base_alpha: float = 0.5
    strategy: MergeStrategy = MergeStrategy.FULL
    model_type: str = "generic"
    # Weights for different factors in alpha computation
    alignment_weight: float = 0.4
    sensitivity_weight: float = 0.3
    importance_weight: float = 0.3
    # Constraints
    min_alpha: float = 0.05
    max_alpha: float = 0.95
    # Component-specific base alphas (optional overrides)
    component_base_alphas: Optional[Dict[str, float]] = None


class CMAMerger:
    """
    Cross-Modal Alignment-Aware Merger.

    Main class for performing adaptive model merging that considers
    modality characteristics and alignment quality.
    """

    def __init__(self, config: Optional[MergeConfig] = None):
        """
        Initialize the CMAMerger.

        Args:
            config: Merge configuration (uses defaults if None)
        """
        self.config = config or MergeConfig()
        self.classifier = ComponentClassifier(self.config.model_type)
        self.alignment_scorer = AlignmentScorer(self.config.model_type)
        self.sensitivity_analyzer = SensitivityAnalyzer(self.config.model_type)

        # Cached analysis results
        self._alignment_scores = None
        self._sensitivity_analysis = None
        self._computed_alphas = None

    def analyze(
        self,
        source_weights: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict:
        """
        Perform comprehensive analysis of source and target models.

        Args:
            source_weights: Source model state dict
            target_weights: Target model state dict
            verbose: Whether to print detailed results

        Returns:
            Dictionary containing all analysis results
        """
        results = {}

        if verbose:
            print("=" * 70)
            print("CMAAM Analysis")
            print("=" * 70)

        # 1. Alignment scoring
        if verbose:
            print("\n[1/3] Computing alignment scores...")
        self._alignment_scores = self.alignment_scorer.compute_all_scores(
            source_weights, target_weights, verbose=verbose
        )
        results["alignment"] = self._alignment_scores

        # 2. Sensitivity analysis for source
        if verbose:
            print("\n[2/3] Analyzing source model sensitivity...")
        source_sensitivity = self.sensitivity_analyzer.analyze_model(
            source_weights, verbose=verbose
        )
        results["source_sensitivity"] = {
            k: {
                "vision": v.vision_sensitivity,
                "language": v.language_sensitivity,
                "importance": v.importance_score,
            }
            for k, v in source_sensitivity.items()
        }

        # 3. Sensitivity analysis for target
        if verbose:
            print("\n[3/3] Analyzing target model sensitivity...")
        target_sensitivity = self.sensitivity_analyzer.analyze_model(
            target_weights, verbose=verbose
        )
        results["target_sensitivity"] = {
            k: {
                "vision": v.vision_sensitivity,
                "language": v.language_sensitivity,
                "importance": v.importance_score,
            }
            for k, v in target_sensitivity.items()
        }

        self._sensitivity_analysis = {
            "source": source_sensitivity,
            "target": target_sensitivity,
        }

        return results

    def compute_adaptive_alphas(
        self,
        source_weights: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Compute adaptive alpha values for each parameter.

        The alpha formula is:
        α_param = base_α × alignment_factor × sensitivity_factor × importance_factor

        Where:
        - alignment_factor: Based on cross-modal alignment scores
        - sensitivity_factor: Based on modality sensitivity variance
        - importance_factor: Based on Fisher information approximation

        Args:
            source_weights: Source model state dict
            target_weights: Target model state dict
            verbose: Whether to print detailed results

        Returns:
            Dictionary mapping parameter names to alpha values
        """
        # Run analysis if not already done
        if self._alignment_scores is None or self._sensitivity_analysis is None:
            self.analyze(source_weights, target_weights, verbose=verbose)

        common_keys = set(source_weights.keys()) & set(target_weights.keys())
        alphas = {}

        # Get component and layer groupings
        layer_groups = self.classifier.group_by_layer(list(common_keys))
        component_groups = self.classifier.group_by_component(list(common_keys))

        # Compute component-level alphas
        component_alphas = self._compute_component_alphas()

        # Compute layer-level alphas
        layer_alphas = self._compute_layer_alphas(source_weights, target_weights)

        # Compute final alphas based on strategy
        for param_name in common_keys:
            component = self.classifier.classify(param_name)
            layer_num = self.classifier.extract_layer_number(param_name)
            layer_key = f"layer_{layer_num}" if layer_num is not None else "non_layer"

            if self.config.strategy == MergeStrategy.BASIC:
                # Simple: use overall alignment-based alpha
                alpha = self._compute_basic_alpha()

            elif self.config.strategy == MergeStrategy.COMPONENT:
                # Component-specific alpha
                alpha = component_alphas.get(component.value, self.config.base_alpha)

            elif self.config.strategy == MergeStrategy.LAYERWISE:
                # Layer-specific alpha
                alpha = layer_alphas.get(layer_key, self.config.base_alpha)

            elif self.config.strategy == MergeStrategy.FULL:
                # Combination of component and layer
                comp_alpha = component_alphas.get(component.value, self.config.base_alpha)
                layer_alpha = layer_alphas.get(layer_key, self.config.base_alpha)

                # Weighted combination
                alpha = 0.5 * comp_alpha + 0.5 * layer_alpha

            else:
                alpha = self.config.base_alpha

            # Apply constraints
            alpha = max(self.config.min_alpha, min(self.config.max_alpha, alpha))
            alphas[param_name] = alpha

        self._computed_alphas = alphas

        if verbose:
            self._print_alpha_summary(alphas)

        return alphas

    def _compute_basic_alpha(self) -> float:
        """Compute basic alpha from overall alignment score."""
        if self._alignment_scores is None:
            return self.config.base_alpha

        overall_alignment = self._alignment_scores.get("overall", {}).get("score", 0.5)

        # Higher alignment = can merge more aggressively
        # Scale around base_alpha
        return self.config.base_alpha * (1 + (overall_alignment - 0.5))

    def _compute_component_alphas(self) -> Dict[str, float]:
        """Compute alphas for each modality component."""
        if self._alignment_scores is None:
            return {comp.value: self.config.base_alpha for comp in ModalityComponent}

        component_scores = self._alignment_scores.get("component_scores", {})
        cmbc_score = self._alignment_scores.get("cmbc", {}).get("overall", 0.5)

        alphas = {}

        for comp in ModalityComponent:
            comp_alignment = component_scores.get(comp.value, 0.5)

            # Base alpha possibly overridden by config
            if (self.config.component_base_alphas and
                comp.value in self.config.component_base_alphas):
                base = self.config.component_base_alphas[comp.value]
            else:
                base = self.config.base_alpha

            # Adjust based on alignment
            # Higher alignment = higher alpha (merge more from source)
            alpha = base * (0.5 + comp_alignment)

            # Special handling for bridge component
            if comp == ModalityComponent.CROSS_MODAL_BRIDGE:
                # Bridge layers need extra care based on CMBC
                if cmbc_score < 0.4:
                    alpha *= 0.7  # Be more conservative
                elif cmbc_score > 0.7:
                    alpha *= 1.2  # Can merge more aggressively

            alphas[comp.value] = alpha

        return alphas

    def _compute_layer_alphas(
        self,
        source_weights: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute alphas for each layer."""
        if self._sensitivity_analysis is None:
            layer_groups = self.classifier.group_by_layer(list(source_weights.keys()))
            return {f"layer_{k}": self.config.base_alpha
                    for k in layer_groups.keys() if k is not None}

        source_sens = self._sensitivity_analysis["source"]
        target_sens = self._sensitivity_analysis["target"]

        alphas = {}

        for layer_key in source_sens.keys():
            if layer_key not in target_sens:
                alphas[layer_key] = self.config.base_alpha
                continue

            src = source_sens[layer_key]
            tgt = target_sens[layer_key]

            # Factor 1: Sensitivity alignment
            # If both models have similar sensitivity patterns, merge more
            sens_diff = abs(src.vision_sensitivity - tgt.vision_sensitivity)
            sens_factor = 1 - sens_diff  # Higher when more aligned

            # Factor 2: Importance
            # More important layers should be merged more conservatively
            avg_importance = (src.importance_score + tgt.importance_score) / 2
            # Normalize importance (assuming typical range 0-10)
            importance_factor = 1 - min(1, avg_importance / 10)

            # Factor 3: Modality balance
            # Balanced layers (not strongly vision or language) can be merged more
            avg_vision = (src.vision_sensitivity + tgt.vision_sensitivity) / 2
            balance_factor = 1 - abs(avg_vision - 0.5) * 2

            # Combine factors using configured weights
            combined = (
                self.config.alignment_weight * sens_factor +
                self.config.sensitivity_weight * balance_factor +
                self.config.importance_weight * importance_factor
            )

            # Scale around base alpha
            alpha = self.config.base_alpha * (0.5 + combined)
            alphas[layer_key] = alpha

        return alphas

    def merge(
        self,
        source_weights: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor],
        alphas: Optional[Dict[str, float]] = None,
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform the actual merging of weights.

        merged[key] = (1 - α_key) × target[key] + α_key × source[key]

        Args:
            source_weights: Source model state dict
            target_weights: Target model state dict (base)
            alphas: Optional pre-computed alphas (computes if None)
            verbose: Whether to print progress

        Returns:
            Merged state dict
        """
        if alphas is None:
            alphas = self.compute_adaptive_alphas(
                source_weights, target_weights, verbose=verbose
            )

        merged = {}
        common_keys = set(source_weights.keys()) & set(target_weights.keys())
        source_only = set(source_weights.keys()) - set(target_weights.keys())
        target_only = set(target_weights.keys()) - set(source_weights.keys())

        if verbose:
            print(f"\nMerging {len(common_keys)} common parameters...")
            if source_only:
                print(f"  {len(source_only)} source-only parameters (kept from source)")
            if target_only:
                print(f"  {len(target_only)} target-only parameters (kept from target)")

        # Merge common parameters
        for key in common_keys:
            alpha = alphas.get(key, self.config.base_alpha)

            source_tensor = source_weights[key]
            target_tensor = target_weights[key]

            # Check shape compatibility
            if source_tensor.shape != target_tensor.shape:
                if verbose:
                    print(f"  Warning: Shape mismatch for {key}, using target")
                merged[key] = target_tensor
                continue

            # Merge: merged = (1 - α) × target + α × source
            merged[key] = (1 - alpha) * target_tensor + alpha * source_tensor

        # Keep source-only parameters
        for key in source_only:
            merged[key] = source_weights[key]

        # Keep target-only parameters
        for key in target_only:
            merged[key] = target_weights[key]

        if verbose:
            print(f"Merge complete. Total parameters: {len(merged)}")

        return merged

    def merge_with_analysis(
        self,
        source_weights: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor],
        verbose: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Perform full analysis and merging in one call.

        Args:
            source_weights: Source model state dict
            target_weights: Target model state dict
            verbose: Whether to print detailed results

        Returns:
            Tuple of (merged_weights, analysis_report)
        """
        # Analyze
        analysis = self.analyze(source_weights, target_weights, verbose=verbose)

        # Compute alphas
        alphas = self.compute_adaptive_alphas(
            source_weights, target_weights, verbose=verbose
        )
        analysis["computed_alphas"] = {
            "mean": float(np.mean(list(alphas.values()))),
            "std": float(np.std(list(alphas.values()))),
            "min": float(min(alphas.values())),
            "max": float(max(alphas.values())),
        }

        # Merge
        merged = self.merge(source_weights, target_weights, alphas, verbose=verbose)

        return merged, analysis

    def save_alphas(self, path: str) -> None:
        """Save computed alphas to JSON file."""
        if self._computed_alphas is None:
            raise ValueError("No alphas computed yet. Run compute_adaptive_alphas first.")

        # Convert to serializable format
        save_data = {
            "config": {
                "base_alpha": self.config.base_alpha,
                "strategy": self.config.strategy.value,
                "model_type": self.config.model_type,
            },
            "alphas": self._computed_alphas,
            "statistics": {
                "mean": float(np.mean(list(self._computed_alphas.values()))),
                "std": float(np.std(list(self._computed_alphas.values()))),
                "min": float(min(self._computed_alphas.values())),
                "max": float(max(self._computed_alphas.values())),
            }
        }

        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"Alphas saved to {path}")

    def load_alphas(self, path: str) -> Dict[str, float]:
        """Load alphas from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        self._computed_alphas = data["alphas"]
        return self._computed_alphas

    def _print_alpha_summary(self, alphas: Dict[str, float]) -> None:
        """Print summary of computed alphas."""
        print(f"\n{'='*70}")
        print("Computed Alpha Summary")
        print(f"{'='*70}")

        # Group alphas by component
        component_alphas = defaultdict(list)
        layer_alphas = defaultdict(list)

        for param_name, alpha in alphas.items():
            component = self.classifier.classify(param_name)
            layer_num = self.classifier.extract_layer_number(param_name)

            component_alphas[component.value].append(alpha)
            if layer_num is not None:
                layer_alphas[layer_num].append(alpha)

        # Print component-wise statistics
        print("\nBy Component:")
        print("-" * 50)
        for comp in ModalityComponent:
            comp_vals = component_alphas.get(comp.value, [])
            if comp_vals:
                print(f"  {comp.value:12s}: mean={np.mean(comp_vals):.4f}, "
                      f"std={np.std(comp_vals):.4f}, "
                      f"range=[{min(comp_vals):.4f}, {max(comp_vals):.4f}]")

        # Print layer-wise statistics (summary)
        if layer_alphas:
            print("\nBy Layer (summary):")
            print("-" * 50)
            layer_means = [(k, np.mean(v)) for k, v in sorted(layer_alphas.items())]

            # Show first few and last few layers
            n_show = 3
            if len(layer_means) <= 2 * n_show:
                for layer_num, mean_alpha in layer_means:
                    print(f"  Layer {layer_num:3d}: mean={mean_alpha:.4f}")
            else:
                for layer_num, mean_alpha in layer_means[:n_show]:
                    print(f"  Layer {layer_num:3d}: mean={mean_alpha:.4f}")
                print("  ...")
                for layer_num, mean_alpha in layer_means[-n_show:]:
                    print(f"  Layer {layer_num:3d}: mean={mean_alpha:.4f}")

        # Overall statistics
        all_alphas = list(alphas.values())
        print(f"\nOverall Statistics:")
        print("-" * 50)
        print(f"  Total parameters: {len(alphas)}")
        print(f"  Mean alpha: {np.mean(all_alphas):.4f}")
        print(f"  Std alpha: {np.std(all_alphas):.4f}")
        print(f"  Min alpha: {min(all_alphas):.4f}")
        print(f"  Max alpha: {max(all_alphas):.4f}")
        print(f"{'='*70}\n")


def create_merger(
    base_alpha: float = 0.5,
    strategy: str = "full",
    model_type: str = "generic",
    **kwargs,
) -> CMAMerger:
    """
    Factory function to create a CMAMerger with common configurations.

    Args:
        base_alpha: Base merge coefficient
        strategy: Merge strategy name
        model_type: Model type for parameter classification
        **kwargs: Additional config options

    Returns:
        Configured CMAMerger instance
    """
    strategy_enum = MergeStrategy(strategy.lower())

    config = MergeConfig(
        base_alpha=base_alpha,
        strategy=strategy_enum,
        model_type=model_type,
        **kwargs,
    )

    return CMAMerger(config)


if __name__ == "__main__":
    # Demo/test code
    import torch

    print("Testing CMAMerger...")

    # Create mock weights
    torch.manual_seed(42)

    def create_mock_model_weights(seed: int = 42):
        torch.manual_seed(seed)
        weights = {}

        # Vision encoder
        weights["visual.patch_embed.proj.weight"] = torch.randn(768, 3, 14, 14)
        weights["visual.blocks.0.attn.qkv.weight"] = torch.randn(2304, 768)

        # Bridge
        weights["visual.merger.mlp.0.weight"] = torch.randn(4096, 768)
        weights["visual.merger.mlp.0.bias"] = torch.randn(4096)

        # Shared
        weights["model.embed_tokens.weight"] = torch.randn(32000, 4096)
        weights["model.norm.weight"] = torch.randn(4096)
        weights["lm_head.weight"] = torch.randn(32000, 4096)

        # Language model layers
        for i in range(4):
            weights[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(4096, 4096)
            weights[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(1024, 4096)
            weights[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(1024, 4096)
            weights[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(4096, 4096)
            weights[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(11008, 4096)
            weights[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(11008, 4096)
            weights[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(4096, 11008)
            weights[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(4096)
            weights[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(4096)

        return weights

    # Create source and target models
    source = create_mock_model_weights(42)
    target = {k: v + 0.1 * torch.randn_like(v) for k, v in create_mock_model_weights(43).items()}

    # Test different strategies
    for strategy in ["basic", "component", "layerwise", "full"]:
        print(f"\n{'='*70}")
        print(f"Testing strategy: {strategy}")
        print(f"{'='*70}")

        merger = create_merger(
            base_alpha=0.5,
            strategy=strategy,
            model_type="qwen2vl",
        )

        # Perform merge
        merged, analysis = merger.merge_with_analysis(source, target, verbose=True)

        print(f"\nMerged model has {len(merged)} parameters")
