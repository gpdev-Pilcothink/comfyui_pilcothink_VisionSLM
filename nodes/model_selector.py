import os
import json
import torch

from ..utils import backend_aliases  # ensure backend path 
backend_aliases.setup_backend_path()  

# 모델 다운로드 유틸
from ..utils.download import download_model  

def _plugin_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _models_dir():
    return os.path.join(_plugin_root(), "Models")

# 모델 식별자(ComfyUI UI 표시용 라벨 -> HF repo_id, 내부 로더 키)
MODEL_TABLE = {
    "deepseek-vl-1.3b-chat": {
        "repo_id": "deepseek-ai/deepseek-vl-1.3b-chat",
        "loader": "deepseek_vl",
    },
    "qwen2.5-vl-3b-instruct": {
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "loader": "qwen",
    },
    "gemma-3-4b-it": {
        "repo_id": "google/gemma-3-4b-it",
        "loader": "gemma",
    },
    # moondream2는 업데이트 예정 중입니다.(취소 될 수도 있음)
    "moondream2": {
        "repo_id": "vikhyatk/moondream2",  # placeholder
        "loader": "moondream2",
    },
}


class SLMVisionModelSelector:
    """
    - 모델을 선택하고, 없으면 Models/에 다운로드
    - 다음 노드가 사용할 수 있도록 구성 딕셔너리(SLM_MODEL)를 반환
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(MODEL_TABLE.keys()), {"default": "deepseek-vl-1.3b-chat"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                "download_if_missing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("SLM_MODEL", "STRING")
    RETURN_NAMES = ("slm_model", "model_path")
    FUNCTION = "select"
    CATEGORY = "SLM‑Vision"

    def _decide_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _decide_dtype(self, dtype, device):
        if dtype != "auto":
            return dtype
        if device == "cuda":
            # bfloat16 우선, 그다음 float16
            if torch.cuda.is_bf16_supported():
                return "bfloat16"
            return "float16"
        return "float32"

    def select(self, model, device, dtype, download_if_missing):
        entry = MODEL_TABLE[model]
        repo_id = entry["repo_id"]
        loader = entry["loader"]

        models_dir = _models_dir()
        os.makedirs(models_dir, exist_ok=True)

        # 로컬 디렉터리 경로(폴더명에 슬래시가 있어 안전하게 치환)
        safe_name = repo_id.replace("/", "__")
        local_dir = os.path.join(models_dir, safe_name)

        # 필요 시 다운로드
        need_download = not (os.path.exists(local_dir) and os.listdir(local_dir))
        if need_download and download_if_missing:
            download_model(repo_id, local_dir) 

        resolved_device = self._decide_device(device)
        resolved_dtype = self._decide_dtype(dtype, resolved_device)

        slm_model = {
            "name": model,
            "repo_id": repo_id,
            "local_path": local_dir,
            "loader": loader,
            "device": resolved_device,
            "dtype": resolved_dtype,
        }

        return (slm_model, local_dir)
