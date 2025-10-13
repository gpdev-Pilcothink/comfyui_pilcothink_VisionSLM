import os
from typing import Optional
from PIL import Image
# ComfyUI 텐서 → PIL 변환

def _tensor_to_pil(image):
    """
    ComfyUI IMAGE(torch.Tensor, [B,H,W,C] 또는 [H,W,C]) → PIL.Image
    그 외(list/tuple/dict/np.ndarray/경로/PIL)도 처리
    """
    # 0) 이미 PIL이면 그대로
    if isinstance(image, Image.Image):
        return image

    # 1) torchvision 사용 가능한 경우: kijai 방식
    try:
        import torch
        from torchvision.transforms import ToPILImage
        to_pil = ToPILImage()

        if isinstance(image, torch.Tensor):
            t = image
            # 배치면 첫 장
            if t.ndim == 4:  # [B,H,W,C] 또는 [B,C,H,W]
                # [B,H,W,C] → [B,C,H,W]
                if t.shape[-1] in (1, 3, 4):
                    t = t.permute(0, 3, 1, 2)
                t = t[0]
            elif t.ndim == 3:  # [H,W,C] 또는 [C,H,W]
                if t.shape[-1] in (1, 3, 4):  # HWC -> CHW
                    t = t.permute(2, 0, 1)
            # 값 범위 보정
            if t.dtype.is_floating_point:
                t = t.clamp(0, 1)
            return to_pil(t)
    except Exception:
        pass  # 아래 폴백으로

    # 2) dict 포맷(Comfy IMAGE 래퍼)
    if isinstance(image, dict) and "image" in image:
        return _tensor_to_pil(image["image"])

    # 3) list/tuple → 첫 항목
    if isinstance(image, (list, tuple)) and len(image) > 0:
        return _tensor_to_pil(image[0])

    # 4) 경로 문자열
    if isinstance(image, str) and os.path.exists(image):
        return Image.open(image).convert("RGB")

    # 5) numpy 폴백
    try:
        import numpy as np
        if isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255.0).round().astype(np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr, "L")
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                return Image.fromarray(arr, "RGB" if arr.shape[-1] == 3 else "RGBA")
    except Exception:
        pass

    # 6) torch 폴백(수동 변환)
    try:
        import torch
        if isinstance(image, torch.Tensor):
            t = image.detach()
            if t.ndim == 4:
                if t.shape[-1] in (1, 3, 4):
                    t = t.permute(0, 3, 1, 2)
                t = t[0]
            elif t.ndim == 3 and t.shape[-1] in (1, 3, 4):
                t = t.permute(2, 0, 1)
            if t.dtype.is_floating_point:
                t = (t.clamp(0, 1) * 255.0).round().to(torch.uint8)
            arr = t.permute(1, 2, 0).cpu().numpy()  # HWC
            return Image.fromarray(arr, "RGB" if arr.shape[-1] == 3 else "RGBA")
    except Exception:
        pass

    raise RuntimeError(f"Unsupported IMAGE type; got {type(image)}")
#############################################################################################

from ..utils.rag_engine import (
    list_rag_txt_files,
    build_or_load_index,
    query_with_rag,
)
# 각 모델 로더
from ..utils.Generator.VisionSLM.gen_deepseek_vl import DeepseekVL13BGenerator
from ..utils.Generator.VisionSLM.gen_qwen2 import QwenVL2Generator
from ..utils.Generator.VisionSLM.gen_qwen25 import QwenVL25Generator
from ..utils.Generator.VisionSLM.gen_gemma import Gemma3_4B_IT_Generator

# 간단한 메모리 캐시(세션 내 재사용)
_MODEL_CACHE = {}

def _get_generator(loader_key, local_path, device, dtype):
    key = (loader_key, local_path, device, dtype)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if loader_key == "deepseek_vl":
        gen = DeepseekVL13BGenerator.from_pretrained(local_path, device=device, dtype=dtype)
    elif loader_key == "qwen2":
        gen = QwenVL2Generator.from_pretrained(local_path, device=device, dtype=dtype)
    elif loader_key == "qwen25":
        gen = QwenVL25Generator.from_pretrained(local_path, device=device, dtype=dtype)
    elif loader_key == "gemma":
        gen = Gemma3_4B_IT_Generator.from_pretrained(local_path, device=device, dtype=dtype)
    elif loader_key == "moondream2":
        raise NotImplementedError("Moondream2 loader is not implemented yet.")
    else:
        raise ValueError(f"Unknown loader: {loader_key}")

    _MODEL_CACHE[key] = gen
    return gen


class SLMVisionGenerator:
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
                "image": ("IMAGE",),
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

    def generate(self, seed, image, slm_model, user_prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
        pil_image = _tensor_to_pil(image)

        # 모델 준비/호출
        gen = _get_generator(
            slm_model["loader"], slm_model["local_path"], slm_model["device"], slm_model["dtype"]
        )

        if top_k == 0 and repetition_penalty == 0.0:
            output_text = gen.generate(
                image=pil_image,
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                # top_k=top_k,
                # repetition_penalty=repetition_penalty,
            )
        elif top_k == 0:
            output_text = gen.generate(
                image=pil_image,
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                # top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        elif repetition_penalty == 0.0:
            output_text = gen.generate(
                image=pil_image,
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                #repetition_penalty=repetition_penalty,
            )
        else :
            output_text = gen.generate(
                image=pil_image,
                prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

        return (output_text,)
