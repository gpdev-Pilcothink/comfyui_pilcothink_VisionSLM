# __init__.py
# ComfyUI 노드 매핑

from .nodes.model_selector import SLMVisionModelSelector
from .nodes.generator_node import SLMVisionGenerator
from .nodes.SLM_model_selector import SLMModelSelector
from .nodes.SLM_generator_node import SLMGenerator
from .nodes.rag_node import SLMVisionRAGNode

from .nodes.merge_texts_with_gap import MergeTextsWithGap, String_Text


NODE_CLASS_MAPPINGS = {
    "SLMVisionModelSelector": SLMVisionModelSelector,
    "SLMVisionGenerator": SLMVisionGenerator,

    "SLMModelSelector": SLMModelSelector,
    "SLMGenerator": SLMGenerator,


    "Pilcothink-RAG" : SLMVisionRAGNode,


    "MergeTextsWithGap": MergeTextsWithGap,
    "String_Text": String_Text,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SLMVisionModelSelector": "SLM‑Vision: Model Selector",
    "SLMVisionGenerator":   "SLM‑Vision: Generator (Vision + optional RAG)",


    "SLMModelSelector": "SLM: Model Selector",
    "SLMGenerator": "SLM: Generator (optional RAG)",


    "Pilcothink-RAG": "SLM-Vision: Pilcothink-RAG",


    "MergeTextsWithGap": "Merge Texts With Gap",
    "String_Text": "String Text",
}
