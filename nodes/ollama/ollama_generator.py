# nodes/ollama_generator.py

from __future__ import annotations

import base64
import io
import json
from typing import Tuple

import torch
from PIL import Image  # type: ignore

try:
    import requests  # type: ignore
except ImportError:
    requests = None  # type: ignore

from .ollama_connective import OllamaConnection, _build_ollama_endpoint


def _image_to_base64(image_tensor: torch.Tensor) -> str:
    """
    ComfyUI IMAGE (torch.Tensor [B,H,W,C], 0~1 float)을
    Ollama /api/generate 의 images 파라미터용 base64 문자열로 변환.
    """
    if image_tensor is None:
        raise ValueError("image_tensor가 None 입니다.")

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"IMAGE 타입이 torch.Tensor가 아닌 것 같습니다: {type(image_tensor)}")

    if image_tensor.dim() == 4:
        first = image_tensor[0]
    else:
        first = image_tensor

    first = first.clamp(0, 1)
    first = (first * 255.0).to(torch.uint8)

    arr = first.cpu().numpy()
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return b64


def _split_thinking_and_answer_from_text(text: str) -> Tuple[str, str]:
    """
    응답 텍스트 안에 <think> ... </think> 패턴이 들어 있을 때
    thinking/answer 를 분리하는 보조 유틸.
    """
    text = (text or "").strip()
    if not text:
        return "", ""

    close_tag = "</think>"
    open_tag = "<think>"

    idx_close = text.find(close_tag)
    if idx_close != -1:
        before = text[:idx_close]
        after = text[idx_close + len(close_tag):]

        idx_open = before.find(open_tag)
        if idx_open != -1:
            reasoning = before[idx_open + len(open_tag):]
        else:
            reasoning = before

        return reasoning.strip(), after.strip()

    idx_open = text.find(open_tag)
    if idx_open != -1:
        reasoning = text[idx_open + len(open_tag):]
        answer = text[:idx_open]
        return reasoning.strip(), answer.strip()

    return "", text


class OllamaGenerator:
    CATEGORY = "Pilcothink/Ollama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "connection": ("OLLAMA_CONNECTION", {"forceInput": True}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "Describe the image.", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "repeat_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "keep_alive": ("STRING", {"default": "", "multiline": False}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "enable_thinking": ("BOOLEAN", {"default": True}),   # 응답 분리용
                "is_thkinking": ("BOOLEAN", {"default": True}),          # think 파라미터 전송용
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("text", "thinking",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    def generate(
        self,
        seed,
        connection: OllamaConnection,
        system_prompt: str,
        prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        keep_alive: str,
        max_tokens: int,
        is_thkinking: bool,        # ← 새로 추가
        enable_thinking: bool,  # ← 위치 변경
        image=None,
    ):
        if requests is None:
            raise RuntimeError(
                "Python 'requests' 패키지가 없습니다. "
                "ComfyUI가 설치된 파이썬 환경에 'pip install requests'로 설치해 주세요."
            )

        if not isinstance(connection, OllamaConnection):
            raise RuntimeError(
                "OLLAMA_CONNECTION 타입이 아닌 값이 들어왔습니다. "
                "먼저 Ollama Connective 노드를 사용해서 연결을 만든 뒤 그 출력을 연결해 주세요."
            )

        base_url = connection.base_url
        api_key = connection.api_key
        model = (connection.model or "").strip()

        if not model:
            if connection.models:
                model = connection.models[0]
            else:
                raise RuntimeError("OllamaConnection 안에 선택된 모델 정보가 없습니다.")

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        images = []
        if image is not None:
            images.append(_image_to_base64(image))

        system_prompt = (system_prompt or "").strip()
        user_prompt = (prompt or "")

        if not user_prompt and not images:
            raise RuntimeError("프롬프트와 이미지가 모두 비어 있습니다.")

        
        options = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_tokens),
        }

        # ✅ top_k: 0이면 key 자체를 넣지 않음
        tk = int(top_k)
        if tk > 0:
            options["top_k"] = tk

        # ✅ repeat_penalty: 0이면 key 자체를 넣지 않음
        rp = float(repeat_penalty)
        if rp > 0:
            options["repeat_penalty"] = rp

        payload = {
            "model": model,
            "prompt": user_prompt,
            "stream": True,
            "options": options,
        }

        if is_thkinking:
            payload["think"] = enable_thinking

        if system_prompt:
            payload["system"] = system_prompt

        if images:
            payload["images"] = images

        ka = (keep_alive or "").strip()
        if ka != "":
            if ka.lstrip("-").isdigit():
                payload["keep_alive"] = int(ka)
            else:
                payload["keep_alive"] = ka

        url = _build_ollama_endpoint(base_url, "/generate")

        thinking_parts = []
        answer_parts = []

        try:
            with requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Ollama /generate 응답 오류 {resp.status_code}: {resp.text[:500]}"
                    )

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue

                    if "error" in item:
                        raise RuntimeError(str(item["error"]))

                    # 🔹 1) reasoning 토큰
                    t = item.get("thinking", "")
                    if t:
                        thinking_parts.append(t)

                    # 🔹 2) 실제 답변 토큰
                    r = item.get("response", "")
                    if r:
                        answer_parts.append(r)

                    if item.get("done"):
                        break

        except Exception as e:
            raise RuntimeError(f"Ollama /generate 요청 중 오류: {e}")

        thinking_text = "".join(thinking_parts).strip()
        answer_text = "".join(answer_parts).strip()

        # 만약 thinking 필드가 없는 모델인데 <think>...</think> 형식을 쓴다면 fallback
        if not thinking_text and "<think>" in answer_text:
            th2, ans2 = _split_thinking_and_answer_from_text(answer_text)
            if th2:
                thinking_text = th2
                answer_text = ans2

        print(f"[OllamaGenerator] model={model}, max_tokens={max_tokens}")
        return (answer_text, thinking_text,)
