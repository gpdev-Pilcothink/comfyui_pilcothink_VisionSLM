# __init__.py
# ComfyUI 노드 매핑

from .nodes.model_selector import SLMVisionModelSelector
from .nodes.generator_node import SLMVisionGenerator
from .nodes.rag_node import SLMVisionRAGNode


NODE_CLASS_MAPPINGS = {
    "SLMVisionModelSelector": SLMVisionModelSelector,
    "SLMVisionGenerator": SLMVisionGenerator,
    "Pilcothink-RAG" : SLMVisionRAGNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SLMVisionModelSelector": "SLM‑Vision: Model Selector",
    "SLMVisionGenerator":   "SLM‑Vision: Generator (Vision + optional RAG)",
    "Pilcothink-RAG": "SLM-Vision: Pilcothink-RAG",
}
