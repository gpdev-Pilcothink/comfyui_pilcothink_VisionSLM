import os
from typing import Optional
from ..utils.rag_engine import (
    list_rag_txt_files,
    build_or_load_index,
    query_with_rag,
)
# 각 모델 로더
from ..utils.Generator.SLM.gen_qwen3 import Qwen3Generator

# ==============================
# 공용 캐시 & 유틸 (단일 상주 정책용)
# ==============================
_MODEL_CACHE = {}

def _to_cpu_safely(gen):
    """가능하면 모델을 CPU로 내린 뒤 참조 해제"""
    try:
        if hasattr(gen, "model") and hasattr(gen.model, "to"):
            gen.model.to("cpu")
    except Exception:
        pass

def _clear_all_models(except_key=None):
    """모든 모델 언로드. except_key만 남길 수 있음(단일 상주 보장)."""
    keys = list(_MODEL_CACHE.keys())
    for k in keys:
        if except_key is not None and k == except_key:
            continue
        try:
            _to_cpu_safely(_MODEL_CACHE[k])
        except Exception:
            pass
        try:
            del _MODEL_CACHE[k]
        except Exception:
            pass
    # CUDA 캐시 정리
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _get_generator(loader_key, local_path, device, dtype, use_cache=True):
    """
    단일 상주 정책:
      - use_cache=False  : 로드 전 전부 언로드, 로드 후에도 캐시에 저장하지 않음
      - use_cache=True   : 요청 키가 캐시에 없으면 전부 언로드 후 이 키만 저장
                           (요청 키가 있으면 그대로 재사용)
    """
    key = (loader_key, local_path, device, dtype)

    # 캐시 OFF → 로드 전 항상 전부 내리고 시작
    if not use_cache:
        _clear_all_models(except_key=None)

    # 캐시 ON + 동일 키 이미 있음 → 재사용
    if use_cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    # 캐시 ON + 새 키 → 다른 것 전부 내리고 이 키만 남길 준비
    if use_cache:
        _clear_all_models(except_key=None)

    # === 로드 ===
    if loader_key == "Qwen3":
        gen = Qwen3Generator.from_pretrained(local_path, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown loader: {loader_key}")

    # 캐시 ON이면 이 키만 보관(단일 상주)
    if use_cache:
        _MODEL_CACHE[key] = gen

    return gen


class SLMGenerator:
    """
    - 입력: SLM_MODEL, 텍스트 프롬프트
    - 출력: STRING(생성 텍스트)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "slm_model": ("SLM_MODEL",),
                "user_prompt": ("STRING", {"forceInput": True}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "repetition_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "use_model_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SLM_TEXT",)
    FUNCTION = "generate"
    CATEGORY = "Pilcothink/SLM"

    def generate(
        self,
        seed,
        slm_model,
        user_prompt,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        use_model_cache,
    ):
        # 로드(정책은 _get_generator 내부에서 처리)
        gen = _get_generator(
            slm_model["loader"],
            slm_model["local_path"],
            slm_model["device"],
            slm_model["dtype"],
            use_cache=use_model_cache,
        )

        # 생성
        if top_k == 0 and repetition_penalty == 0.0:
            output_text = gen.generate(
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elif top_k == 0:
            output_text = gen.generate(
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        elif repetition_penalty == 0.0:
            output_text = gen.generate(
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            output_text = gen.generate(
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

        # 캐시 OFF: 실행 후에도 깨끗이 내림(이중 안전)
        if not use_model_cache:
            _clear_all_models(except_key=None)

        return (output_text,)
