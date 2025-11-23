from .nodes.Vision_model_selector import SLMVisionModelSelector
from .nodes.Vision_generator_node import SLMVisionGenerator
from .nodes.SLM_model_selector import SLMModelSelector
from .nodes.SLM_generator_node import SLMGenerator
from .nodes.rag_node import SLMVisionRAGNode


from .nodes.vllm.vllm_connective import VLLMConnective
from .nodes.vllm.vllm_generator import VLLMGenerator
from .nodes.ollama.ollama_connective import OllamaConnective
from .nodes.ollama.ollama_generator import OllamaGenerator


from .nodes.merge_texts_with_gap import MergeTextsWithGap, String_Text, RemoveSpecificPatterns


NODE_CLASS_MAPPINGS = {
    "SLMVisionModelSelector": SLMVisionModelSelector,
    "SLMVisionGenerator": SLMVisionGenerator,

    "SLMModelSelector": SLMModelSelector,
    "SLMGenerator": SLMGenerator,


    "Pilcothink-RAG" : SLMVisionRAGNode,


    "MergeTextsWithGap": MergeTextsWithGap,
    "String_Text": String_Text,
    "RemoveSpecificPatterns": RemoveSpecificPatterns,


    "VLLMConnective": VLLMConnective,
    "VLLMGenerator": VLLMGenerator,


    "OllamaConnective": OllamaConnective,
    "OllamaGenerator": OllamaGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SLMVisionModelSelector": "Vision-SLM: Model Selector",
    "SLMVisionGenerator":   "Vision-SLM: Generator",


    "SLMModelSelector": "SLM: Model Selector",
    "SLMGenerator": "SLM: Generator",


    "Pilcothink-RAG": "Prompt+RAG",


    "MergeTextsWithGap": "Merge Texts With Gap",
    "String_Text": "String Text",
    "RemoveSpecificPatterns": "Remove Specific Patterns",


    "VLLMConnective": "vLLM Connective",
    "VLLMGenerator": "vLLM Generator",


    "OllamaConnective": "Ollama Connective",
    "OllamaGenerator": "Ollama Generator",
}
