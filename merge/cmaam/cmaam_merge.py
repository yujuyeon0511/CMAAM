#!/usr/bin/env python3
"""
CMAAM Merge Script

Main CLI script for Cross-Modal Alignment-Aware Merging of Multimodal LLMs.

Usage:
    python merge/cmaam/cmaam_merge.py \
        --source /path/to/source_model \
        --target /path/to/target_model \
        --output /path/to/output \
        --base-alpha 0.5 \
        --strategy full \
        --model-type qwen2vl \
        --analyze
"""

import os
import sys
import json
import argparse
import torch
import safetensors.torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from merge.cmaam.component_classifier import (
    ModalityComponent,
    ComponentClassifier,
    print_classification_summary,
)
from merge.cmaam.alignment_scorer import AlignmentScorer
from merge.cmaam.sensitivity_analyzer import SensitivityAnalyzer
from merge.cmaam.adaptive_merger import (
    MergeStrategy,
    MergeConfig,
    CMAMerger,
    create_merger,
)


# ============================================================================
# Model Loading Utilities
# ============================================================================

def get_model_files(model_path: str) -> List[str]:
    """Get list of weight files from a model directory."""
    if not os.path.isdir(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    # Check for index file
    index_files = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]

    for index_file in index_files:
        index_path = os.path.join(model_path, index_file)
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            files = list(set(weight_map.values()))
            return sorted(files)

    # Fallback: find all weight files
    safetensor_files = sorted([
        f for f in os.listdir(model_path)
        if f.endswith(".safetensors")
    ])
    if safetensor_files:
        return safetensor_files

    bin_files = sorted([
        f for f in os.listdir(model_path)
        if f.endswith(".bin") and "pytorch_model" in f
    ])
    return bin_files


def load_model_weights(model_path: str, dtype: torch.dtype = torch.bfloat16) -> Dict[str, torch.Tensor]:
    """
    Load model weights from a checkpoint directory.

    Args:
        model_path: Path to the model directory
        dtype: Data type for weights

    Returns:
        State dictionary
    """
    print(f"Loading weights from: {model_path}")

    files = get_model_files(model_path)
    if not files:
        raise ValueError(f"No weight files found in {model_path}")

    print(f"  Found {len(files)} weight file(s)")

    weights = {}
    for file in files:
        file_path = os.path.join(model_path, file)
        print(f"  Loading: {file}")

        if file.endswith(".safetensors"):
            file_weights = safetensors.torch.load_file(file_path)
        else:
            file_weights = torch.load(file_path, map_location="cpu")

        weights.update(file_weights)

    # Convert to specified dtype
    print(f"  Converting to {dtype}")
    for key in weights:
        if weights[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weights[key] = weights[key].to(dtype)

    print(f"  Loaded {len(weights)} parameters")
    return weights


def save_model_weights(
    weights: Dict[str, torch.Tensor],
    output_path: str,
    reference_model_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save merged model weights to output directory.

    Args:
        weights: Merged state dictionary
        output_path: Output directory path
        reference_model_path: Reference model path for config files
        metadata: Optional metadata to save
    """
    os.makedirs(output_path, exist_ok=True)

    # Get reference model's index to determine file splits
    index_path = os.path.join(reference_model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})

        # Split weights according to original mapping
        file_weights = {}
        for key, weight_file in weight_map.items():
            if key in weights:
                if weight_file not in file_weights:
                    file_weights[weight_file] = {}
                file_weights[weight_file][key] = weights[key]

        # Save each shard
        for file_name, shard_weights in file_weights.items():
            save_path = os.path.join(output_path, file_name)
            print(f"  Saving: {file_name} ({len(shard_weights)} params)")
            safetensors.torch.save_file(shard_weights, save_path, metadata={"format": "pt"})

        # Copy index file
        new_index_path = os.path.join(output_path, "model.safetensors.index.json")
        with open(new_index_path, "w") as f:
            json.dump(index_data, f, indent=2)

    else:
        # Save as single file
        save_path = os.path.join(output_path, "model.safetensors")
        print(f"  Saving: model.safetensors ({len(weights)} params)")
        safetensors.torch.save_file(weights, save_path, metadata={"format": "pt"})

    # Create symlinks for config files
    create_config_symlinks(reference_model_path, output_path)

    # Save merge metadata
    if metadata:
        metadata_path = os.path.join(output_path, "cmaam_merge_info.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved merge info to: cmaam_merge_info.json")

    print(f"Model saved to: {output_path}")


def create_config_symlinks(source_path: str, target_path: str) -> None:
    """Create symlinks for config files."""
    config_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
    ]

    for config_file in config_files:
        source_file = os.path.join(source_path, config_file)
        target_file = os.path.join(target_path, config_file)

        if os.path.exists(source_file) and not os.path.exists(target_file):
            try:
                os.symlink(source_file, target_file)
                print(f"  Linked: {config_file}")
            except OSError as e:
                print(f"  Warning: Could not create symlink for {config_file}: {e}")


# ============================================================================
# Main Merge Function
# ============================================================================

def run_cmaam_merge(
    source_path: str,
    target_path: str,
    output_path: str,
    base_alpha: float = 0.5,
    strategy: str = "full",
    model_type: str = "generic",
    analyze: bool = False,
    save_alphas: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Run CMAAM merge on two models.

    Args:
        source_path: Path to source model
        target_path: Path to target model
        output_path: Path for output merged model
        base_alpha: Base merge coefficient
        strategy: Merge strategy (basic/layerwise/component/full)
        model_type: Model type for parameter classification
        analyze: Whether to output detailed analysis report
        save_alphas: Whether to save computed alphas
        verbose: Whether to print progress

    Returns:
        Dictionary with merge report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "source": source_path,
        "target": target_path,
        "output": output_path,
        "config": {
            "base_alpha": base_alpha,
            "strategy": strategy,
            "model_type": model_type,
        },
    }

    print("=" * 70)
    print("CMAAM: Cross-Modal Alignment-Aware Merging")
    print("=" * 70)
    print(f"\nSource: {source_path}")
    print(f"Target: {target_path}")
    print(f"Output: {output_path}")
    print(f"\nConfig:")
    print(f"  Base Alpha: {base_alpha}")
    print(f"  Strategy: {strategy}")
    print(f"  Model Type: {model_type}")
    print()

    # Load models
    print("=" * 70)
    print("Loading Models")
    print("=" * 70)
    source_weights = load_model_weights(source_path)
    target_weights = load_model_weights(target_path)

    # Print classification summary
    if analyze:
        print("\n" + "=" * 70)
        print("Parameter Classification")
        print("=" * 70)
        print("\nSource Model:")
        print_classification_summary(source_weights, model_type)
        print("\nTarget Model:")
        print_classification_summary(target_weights, model_type)

    # Check compatibility
    common_keys = set(source_weights.keys()) & set(target_weights.keys())
    source_only = set(source_weights.keys()) - set(target_weights.keys())
    target_only = set(target_weights.keys()) - set(source_weights.keys())

    print("\n" + "=" * 70)
    print("Compatibility Check")
    print("=" * 70)
    print(f"  Common parameters: {len(common_keys)}")
    print(f"  Source-only parameters: {len(source_only)}")
    print(f"  Target-only parameters: {len(target_only)}")

    if source_only:
        print("\n  Sample source-only parameters:")
        for key in list(source_only)[:5]:
            print(f"    - {key}")

    if target_only:
        print("\n  Sample target-only parameters:")
        for key in list(target_only)[:5]:
            print(f"    - {key}")

    report["compatibility"] = {
        "common_params": len(common_keys),
        "source_only": len(source_only),
        "target_only": len(target_only),
    }

    # Create merger
    merger = create_merger(
        base_alpha=base_alpha,
        strategy=strategy,
        model_type=model_type,
    )

    # Run merge with analysis
    print("\n" + "=" * 70)
    print("Running CMAAM Merge")
    print("=" * 70)

    merged_weights, analysis = merger.merge_with_analysis(
        source_weights, target_weights, verbose=verbose
    )

    report["analysis"] = analysis

    # Save alphas if requested
    if save_alphas:
        alphas_path = os.path.join(output_path, "cmaam_alphas.json")
        os.makedirs(output_path, exist_ok=True)
        merger.save_alphas(alphas_path)
        report["alphas_path"] = alphas_path

    # Save merged model
    print("\n" + "=" * 70)
    print("Saving Merged Model")
    print("=" * 70)

    save_model_weights(
        merged_weights,
        output_path,
        target_path,  # Use target as reference for file structure
        metadata=report,
    )

    # Generate analysis report if requested
    if analyze:
        report_path = os.path.join(output_path, "cmaam_analysis_report.json")
        with open(report_path, "w") as f:
            # Convert any non-serializable objects
            serializable_report = json.loads(
                json.dumps(report, default=lambda x: str(x) if not isinstance(x, (int, float, str, list, dict, type(None))) else x)
            )
            json.dump(serializable_report, f, indent=2)
        print(f"\nAnalysis report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("CMAAM Merge Complete!")
    print("=" * 70)
    print(f"\nOutput saved to: {output_path}")

    return report


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CMAAM: Cross-Modal Alignment-Aware Merging for MLLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic merge with default settings
    python cmaam_merge.py --source /path/to/model_a --target /path/to/model_b --output /path/to/merged

    # Full analysis with component strategy
    python cmaam_merge.py --source /path/to/model_a --target /path/to/model_b \\
        --output /path/to/merged --strategy component --analyze

    # Merge Qwen2-VL with LLaVA-OneVision
    python cmaam_merge.py \\
        --source /path/to/llava-onevision-qwen2-7b-si \\
        --target /path/to/Qwen2-VL-7B-Instruct \\
        --output /path/to/cmaam_merged \\
        --model-type qwen2vl \\
        --strategy full \\
        --base-alpha 0.5 \\
        --analyze \\
        --save-alphas
        """
    )

    # Required arguments
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Path to source MLLM checkpoint"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        help="Path to target MLLM checkpoint"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for merged model"
    )

    # Merge configuration
    parser.add_argument(
        "--base-alpha",
        type=float,
        default=0.5,
        help="Base merge coefficient (default: 0.5)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="full",
        choices=["basic", "layerwise", "component", "full"],
        help="Merge strategy (default: full)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="generic",
        choices=["qwen2vl", "llava", "llava_onevision", "cogvlm", "mplugowl", "generic"],
        help="Model type for parameter classification (default: generic)"
    )

    # Output options
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Output detailed analysis report"
    )
    parser.add_argument(
        "--save-alphas",
        action="store_true",
        help="Save computed alpha values to JSON"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_cmaam_merge(
        source_path=args.source,
        target_path=args.target,
        output_path=args.output,
        base_alpha=args.base_alpha,
        strategy=args.strategy,
        model_type=args.model_type,
        analyze=args.analyze,
        save_alphas=args.save_alphas,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
