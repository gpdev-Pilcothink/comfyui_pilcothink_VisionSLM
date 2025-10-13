import os
from typing import Optional
from ..utils.rag_engine import (
    list_rag_txt_files,
    build_or_load_index,
    query_with_rag,
)
# 각 모델 로더
from ..utils.Generator.SLM.gen_qwen3 import Qwen3Generator


# 간단한 메모리 캐시(세션 내 재사용)
_MODEL_CACHE = {}

def _get_generator(loader_key, local_path, device, dtype):
    key = (loader_key, local_path, device, dtype)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if loader_key == "Qwen3":
        gen = Qwen3Generator.from_pretrained(local_path, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown loader: {loader_key}")

    _MODEL_CACHE[key] = gen
    return gen


class SLMGenerator:
    """
    - 입력: IMAGE, SLM_MODEL, 텍스트 프롬프트
    - 선택: RAG 사용 여부 + RAG_DOC 폴더의 txt 파일 선택
    - 출력: STRING(생성 텍스트)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "slm_model": ("SLM_MODEL",),
                "user_prompt": ("STRING", {"forceInput": True}),  # RAG 컨텍스트는 입력으로 받음
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 1000}),  # 추가
                "repetition_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1}),  # 추가
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SLM_TEXT",)
    FUNCTION = "generate"
    CATEGORY = "SLM‑Vision"

    def generate(self, seed, slm_model, user_prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
        # 모델 준비/호출
        gen = _get_generator(
            slm_model["loader"], slm_model["local_path"], slm_model["device"], slm_model["dtype"]
        )

        if top_k == 0 and repetition_penalty == 0.0:
            output_text = gen.generate(
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                # top_k=top_k,
                # repetition_penalty=repetition_penalty,
            )
        elif top_k == 0:
            output_text = gen.generate(
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                # top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        elif repetition_penalty == 0.0:
            output_text = gen.generate(
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                #repetition_penalty=repetition_penalty,
            )
        else :
            output_text = gen.generate(
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

        return (output_text,)
