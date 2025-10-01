import os
from typing import Optional

# RAG 엔진 관련 유틸리티
from ..utils.rag_engine import (
    list_rag_txt_files,
    build_or_load_index,
    query_with_rag,
)

class SLMVisionRAGNode:
    """
    - 입력: RAG 문서 선택, 프롬프트, 사용 여부 토글
    - 출력: STRING(완성된 프롬프트)
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 동적 파일 목록 로딩
        rag_choices = list_rag_txt_files()
        if not rag_choices:
            rag_choices = ["(no_txt_found)"]

        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "rag_txt": (rag_choices, {"default": rag_choices[0]}),
                "prompt": ("STRING", {"multiline": True, "default": "Analyze the image."}),
                "use_rag": ("BOOLEAN", {"default": False}),
                "k": ("INT", {"default": 4, "min": 1, "max": 20}),  # RAG k 값 조절
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_prompt",)
    FUNCTION = "generate"
    CATEGORY = "SLM‑Vision"

    def _compose_prompt(self, prompt: str, rag_context: Optional[str]) -> str:
        if rag_context:
            return (
                "You may use the following context to answer concisely.\n"
                f"[CONTEXT]\n{rag_context}\n[/CONTEXT]\n\n"
                f"{prompt}"
            )
        return prompt

    def generate(self, seed, rag_txt, prompt, use_rag, k):
        # RAG 사용하지 않거나 문서가 선택되지 않은 경우
        if not use_rag or rag_txt == "(no_txt_found)":
            return (prompt,)  # 원래 프롬프트 반환
        
        try:
            # 인덱스 빌드/로드
            index = build_or_load_index(rag_txt)
            # RAG 쿼리
            rag_context = query_with_rag(index, prompt, k=k)
            # 완성된 프롬프트 생성
            final_prompt = self._compose_prompt(prompt, rag_context)
            return (final_prompt,)
        except Exception as e:
            print(f"RAG processing error: {e}")
            return (prompt,)  # 오류 발생 시 원래 프롬프트 반환
