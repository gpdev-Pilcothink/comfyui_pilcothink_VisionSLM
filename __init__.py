# __init__.py
# ComfyUI 노드 매핑

from .nodes.model_selector import SLMVisionModelSelector
from .nodes.generator_node import SLMVisionGenerator

NODE_CLASS_MAPPINGS = {
    "SLMVisionModelSelector": SLMVisionModelSelector,
    "SLMVisionGenerator": SLMVisionGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SLMVisionModelSelector": "SLM‑Vision: Model Selector",
    "SLMVisionGenerator":   "SLM‑Vision: Generator (Vision + optional RAG)",
}
