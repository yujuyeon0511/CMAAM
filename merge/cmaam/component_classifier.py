"""
Component Classifier for CMAAM

Automatically classifies MLLM parameters into modality-specific categories:
- VISION_ENCODER: Vision tower parameters (ViT, CLIP, etc.)
- CROSS_MODAL_BRIDGE: Cross-modal projection layers (mm_projector, adapter, etc.)
- LANGUAGE_MODEL: LLM backbone parameters
- SHARED_EMBEDDING: Shared embeddings (embed_tokens, lm_head, norm)
"""

import re
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class ModalityComponent(Enum):
    """Enum for classifying MLLM parameters by modality."""
    VISION_ENCODER = "vision"
    CROSS_MODAL_BRIDGE = "bridge"
    LANGUAGE_MODEL = "language"
    SHARED_EMBEDDING = "shared"


# Parameter name patterns for each model type
MODEL_PATTERNS = {
    "qwen2vl": {
        ModalityComponent.VISION_ENCODER: [
            r"^visual\..*",
            r"^model\.visual\..*",
            r"^vision_tower\..*",
            r"^image_encoder\..*",
        ],
        ModalityComponent.CROSS_MODAL_BRIDGE: [
            r"^visual\.merger\..*",
            r"^model\.visual\.merger\..*",
            r"^mm_projector\..*",
            r"^multi_modal_projector\..*",
            r"^adapter\..*",
            r"^connector\..*",
        ],
        ModalityComponent.SHARED_EMBEDDING: [
            r"^model\.embed_tokens\..*",
            r"^lm_head\..*",
            r"^model\.norm\..*",
            r"^embed_tokens\..*",
        ],
        ModalityComponent.LANGUAGE_MODEL: [
            r"^model\.layers\..*",
            r"^language_model\..*",
        ],
    },
    "llava": {
        ModalityComponent.VISION_ENCODER: [
            r"^vision_tower\..*",
            r"^model\.vision_tower\..*",
            r"^image_newline",
        ],
        ModalityComponent.CROSS_MODAL_BRIDGE: [
            r"^mm_projector\..*",
            r"^model\.mm_projector\..*",
            r"^multi_modal_projector\..*",
        ],
        ModalityComponent.SHARED_EMBEDDING: [
            r"^model\.embed_tokens\..*",
            r"^lm_head\..*",
            r"^model\.norm\..*",
            r"^embed_tokens\..*",
        ],
        ModalityComponent.LANGUAGE_MODEL: [
            r"^model\.layers\..*",
            r"^language_model\.model\.layers\..*",
        ],
    },
    "llava_onevision": {
        ModalityComponent.VISION_ENCODER: [
            r"^vision_tower\..*",
            r"^model\.vision_tower\..*",
            r"^image_newline",
        ],
        ModalityComponent.CROSS_MODAL_BRIDGE: [
            r"^mm_projector\..*",
            r"^model\.mm_projector\..*",
            r"^multi_modal_projector\..*",
        ],
        ModalityComponent.SHARED_EMBEDDING: [
            r"^model\.embed_tokens\..*",
            r"^lm_head\..*",
            r"^model\.norm\..*",
            r"^embed_tokens\..*",
        ],
        ModalityComponent.LANGUAGE_MODEL: [
            r"^model\.layers\..*",
        ],
    },
    "cogvlm": {
        ModalityComponent.VISION_ENCODER: [
            r"^model\.vision\..*",
            r"^vision_model\..*",
            r"^visual\..*",
        ],
        ModalityComponent.CROSS_MODAL_BRIDGE: [
            r"^model\.vision\.adapter\..*",
            r"^vision_proj\..*",
            r"^adapter\..*",
        ],
        ModalityComponent.SHARED_EMBEDDING: [
            r"^model\.embed_tokens\..*",
            r"^lm_head\..*",
            r"^model\.norm\..*",
            r"^transformer\.output_layer\..*",
        ],
        ModalityComponent.LANGUAGE_MODEL: [
            r"^model\.layers\..*",
            r"^transformer\.encoder\.layers\..*",
        ],
    },
    "mplugowl": {
        ModalityComponent.VISION_ENCODER: [
            r"^vision_model\..*",
            r"^visual_encoder\..*",
            r"^vit\..*",
        ],
        ModalityComponent.CROSS_MODAL_BRIDGE: [
            r"^visual_abstractor\..*",
            r"^vision_proj\..*",
            r"^query_tokens",
        ],
        ModalityComponent.SHARED_EMBEDDING: [
            r"^model\.embed_tokens\..*",
            r"^lm_head\..*",
            r"^model\.norm\..*",
            r"^language_model\.model\.embed_tokens\..*",
            r"^language_model\.lm_head\..*",
        ],
        ModalityComponent.LANGUAGE_MODEL: [
            r"^model\.layers\..*",
            r"^language_model\.model\.layers\..*",
        ],
    },
    # Generic patterns for unknown model types
    "generic": {
        ModalityComponent.VISION_ENCODER: [
            r".*vision.*",
            r".*visual.*",
            r".*vit.*",
            r".*image_encoder.*",
            r".*clip.*",
        ],
        ModalityComponent.CROSS_MODAL_BRIDGE: [
            r".*mm_projector.*",
            r".*multimodal_projector.*",
            r".*adapter.*",
            r".*connector.*",
            r".*merger.*",
            r".*abstractor.*",
        ],
        ModalityComponent.SHARED_EMBEDDING: [
            r".*embed_tokens.*",
            r".*lm_head.*",
            r".*\.norm\..*",
            r".*final_layer_norm.*",
        ],
        ModalityComponent.LANGUAGE_MODEL: [
            r".*\.layers\..*",
            r".*decoder.*",
            r".*transformer.*",
        ],
    },
}


class ComponentClassifier:
    """
    Classifier for MLLM parameters based on modality component.

    This class provides methods to classify parameters into different
    modality components and group them accordingly.
    """

    def __init__(self, model_type: str = "generic"):
        """
        Initialize the classifier with a specific model type.

        Args:
            model_type: Type of the model (qwen2vl, llava, llava_onevision,
                       cogvlm, mplugowl, generic)
        """
        self.model_type = model_type.lower()
        if self.model_type not in MODEL_PATTERNS:
            print(f"Warning: Unknown model type '{model_type}', using generic patterns")
            self.model_type = "generic"
        self.patterns = MODEL_PATTERNS[self.model_type]
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[ModalityComponent, List[re.Pattern]]:
        """Compile regex patterns for efficient matching."""
        compiled = {}
        for component, patterns in self.patterns.items():
            compiled[component] = [re.compile(p) for p in patterns]
        return compiled

    def classify(self, param_name: str) -> ModalityComponent:
        """
        Classify a single parameter by its name.

        Args:
            param_name: Name of the parameter

        Returns:
            ModalityComponent enum value
        """
        # Check patterns in order of specificity
        # Bridge patterns should be checked before vision/language
        # to catch specific adapter layers
        check_order = [
            ModalityComponent.CROSS_MODAL_BRIDGE,
            ModalityComponent.SHARED_EMBEDDING,
            ModalityComponent.VISION_ENCODER,
            ModalityComponent.LANGUAGE_MODEL,
        ]

        for component in check_order:
            for pattern in self._compiled_patterns[component]:
                if pattern.match(param_name):
                    return component

        # Default to language model if no pattern matches
        return ModalityComponent.LANGUAGE_MODEL

    def classify_all(self, param_names: List[str]) -> Dict[str, ModalityComponent]:
        """
        Classify all parameters.

        Args:
            param_names: List of parameter names

        Returns:
            Dictionary mapping parameter names to components
        """
        return {name: self.classify(name) for name in param_names}

    def group_by_component(
        self,
        param_names: List[str]
    ) -> Dict[ModalityComponent, List[str]]:
        """
        Group parameters by their modality component.

        Args:
            param_names: List of parameter names

        Returns:
            Dictionary mapping components to lists of parameter names
        """
        groups = defaultdict(list)
        for name in param_names:
            component = self.classify(name)
            groups[component].append(name)
        return dict(groups)

    def get_statistics(
        self,
        param_names: List[str]
    ) -> Dict[ModalityComponent, int]:
        """
        Get statistics about parameter distribution across components.

        Args:
            param_names: List of parameter names

        Returns:
            Dictionary mapping components to parameter counts
        """
        groups = self.group_by_component(param_names)
        return {comp: len(names) for comp, names in groups.items()}

    def extract_layer_number(self, param_name: str) -> Optional[int]:
        """
        Extract layer number from parameter name if present.

        Args:
            param_name: Parameter name

        Returns:
            Layer number or None if not found
        """
        # Common patterns for layer numbering
        patterns = [
            r"\.layers\.(\d+)\.",
            r"\.layer\.(\d+)\.",
            r"\.blocks\.(\d+)\.",
            r"\.block\.(\d+)\.",
            r"\.h\.(\d+)\.",
            r"_(\d+)\.",
        ]

        for pattern in patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))
        return None

    def group_by_layer(
        self,
        param_names: List[str]
    ) -> Dict[Optional[int], List[str]]:
        """
        Group parameters by their layer number.

        Args:
            param_names: List of parameter names

        Returns:
            Dictionary mapping layer numbers to parameter names
            (None key for parameters without layer number)
        """
        groups = defaultdict(list)
        for name in param_names:
            layer_num = self.extract_layer_number(name)
            groups[layer_num].append(name)
        return dict(groups)


def classify_parameter(name: str, model_type: str = "generic") -> ModalityComponent:
    """
    Convenience function to classify a single parameter.

    Args:
        name: Parameter name
        model_type: Type of the model

    Returns:
        ModalityComponent enum value
    """
    classifier = ComponentClassifier(model_type)
    return classifier.classify(name)


def get_component_groups(
    state_dict: Dict,
    model_type: str = "generic"
) -> Dict[ModalityComponent, List[str]]:
    """
    Group all parameters in a state dict by component.

    Args:
        state_dict: Model state dictionary
        model_type: Type of the model

    Returns:
        Dictionary mapping components to parameter names
    """
    classifier = ComponentClassifier(model_type)
    return classifier.group_by_component(list(state_dict.keys()))


def print_classification_summary(
    state_dict: Dict,
    model_type: str = "generic"
) -> None:
    """
    Print a summary of parameter classification.

    Args:
        state_dict: Model state dictionary
        model_type: Type of the model
    """
    classifier = ComponentClassifier(model_type)
    groups = classifier.group_by_component(list(state_dict.keys()))
    stats = classifier.get_statistics(list(state_dict.keys()))

    total = sum(stats.values())
    print(f"\n{'='*60}")
    print(f"Parameter Classification Summary (Model Type: {model_type})")
    print(f"{'='*60}")

    for component in ModalityComponent:
        count = stats.get(component, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{component.value:12s}: {count:6d} parameters ({percentage:5.1f}%)")

    print(f"{'='*60}")
    print(f"{'Total':12s}: {total:6d} parameters")
    print(f"{'='*60}\n")

    # Print sample parameters for each component
    print("Sample parameters per component:")
    print("-" * 60)
    for component in ModalityComponent:
        params = groups.get(component, [])
        print(f"\n{component.value}:")
        for param in params[:3]:  # Show first 3 parameters
            print(f"  - {param}")
        if len(params) > 3:
            print(f"  ... and {len(params) - 3} more")


if __name__ == "__main__":
    # Demo/test code
    import torch

    # Create sample parameter names for testing
    sample_params = {
        # Qwen2-VL style parameters
        "visual.patch_embed.proj.weight": torch.zeros(1),
        "visual.blocks.0.attn.qkv.weight": torch.zeros(1),
        "visual.merger.mlp.0.weight": torch.zeros(1),
        "model.embed_tokens.weight": torch.zeros(1),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(1),
        "model.norm.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
        # LLaVA style parameters
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "mm_projector.0.weight": torch.zeros(1),
    }

    print("Testing ComponentClassifier...")

    for model_type in ["qwen2vl", "llava", "generic"]:
        print(f"\n\nModel Type: {model_type}")
        print_classification_summary(sample_params, model_type)
