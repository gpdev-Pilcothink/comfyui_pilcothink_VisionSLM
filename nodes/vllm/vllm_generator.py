# nodes/vllm_generator.py

from __future__ import annotations

import base64
import io
from typing import Any, Tuple

import torch
from PIL import Image  # type: ignore

try:
    import requests  # type: ignore
except ImportError:
    requests = None  # type: ignore

from .vllm_connective import VLLMConnection, _build_endpoint


def _image_to_data_url(image_tensor: torch.Tensor) -> str:
    """
    ComfyUI IMAGE (torch.Tensor [B,H,W,C], 0~1 float)을
    OpenAI Vision 스타일 image_url 용 data:image/png;base64,... 로 변환.
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
    return f"data:image/png;base64,{b64}"


def _extract_text_from_content(content: Any) -> str:
    """
    OpenAI ChatCompletion 스타일 message.content에서 텍스트만 뽑아낸다.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    texts.append(t)
        return "".join(texts).strip()

    return str(content)


def _split_thinking_and_answer_from_message(message: dict) -> Tuple[str, str]:
    """
    reasoning/Thinking 모델용 출력 분리 유틸.

    1) message.reasoning_content / message.reasoning 필드를 먼저 보고
    2) 없으면 content 안의 <think> ... </think> 태그를 파싱해서
       (thinking, answer) 튜플로 리턴한다.
    """
    raw_content = message.get("content", "")
    content_text = _extract_text_from_content(raw_content).strip()

    # 1) reasoning 필드
    reasoning_raw = message.get("reasoning_content") or message.get("reasoning")
    if reasoning_raw:
        reasoning_text = _extract_text_from_content(reasoning_raw).strip()
        return reasoning_text, content_text

    # 2) <think> ... </think> 패턴
    text = content_text
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

    # 3) <think>만 있고 </think>가 없는 경우
    idx_open = text.find(open_tag)
    if idx_open != -1:
        reasoning = text[idx_open + len(open_tag):]
        answer = text[:idx_open]
        return reasoning.strip(), answer.strip()

    # 4) Thinking 정보가 전혀 없으면 전체를 answer 로
    return "", content_text


class VLLMGenerator:
    """
    vLLM OpenAI 호환 서버에 /v1/chat/completions 요청을 보내는 노드.

    입력:
      - connection   (VLLM_CONNECTION): vLLMConnective 노드 출력 (여기서 모델도 이미 선택됨)
      - system_prompt: 시스템 프롬프트 (role=system)
      - prompt       : USER 프롬프트 (role=user)
      - image        : 선택, 비전 언어 모델일 때만 사용 (IMAGE 타입)
      - temperature, top_p
      - max_tokens   : 최대 생성 토큰 수

    출력:
      - text    : Thinking 을 제거한 최종 답변
      - thinking: Thinking (체인오브쏘트) 부분만 따로
    """

    CATEGORY = "Pilcothink/vLLM"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "connection": ("VLLM_CONNECTION", {"forceInput": True}),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Describe the image.",
                        "multiline": True,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                    },
                ),
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
        connection: VLLMConnection,
        system_prompt: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        image=None,
    ):
        if requests is None:
            raise RuntimeError(
                "Python 'requests' 패키지가 없습니다. "
                "ComfyUI가 설치된 파이썬 환경에 'pip install requests'로 설치해 주세요."
            )

        if not isinstance(connection, VLLMConnection):
            raise RuntimeError(
                "VLLM_CONNECTION 타입이 아닌 값이 들어왔습니다. "
                "먼저 vLLM Connective 노드를 사용해서 연결을 만든 뒤 그 출력을 연결해 주세요."
            )

        base_url = connection.base_url
        api_key = connection.api_key
        model = (connection.model or "").strip()

        if not model:
            if connection.models:
                model = connection.models[0]
            else:
                raise RuntimeError("VLLMConnection 안에 선택된 모델 정보가 없습니다.")

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        system_prompt = (system_prompt or "").strip()
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        # user 멀티모달
        content = []
        if prompt:
            content.append({"type": "text", "text": prompt})

        if image is not None:
            data_url = _image_to_data_url(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )

        if not content:
            raise RuntimeError("프롬프트와 이미지가 모두 비어 있습니다.")

        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }

        url = _build_endpoint(base_url, "/chat/completions")

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
        except Exception as e:
            raise RuntimeError(f"vLLM /chat/completions 요청 중 오류: {e}")

        if resp.status_code != 200:
            raise RuntimeError(
                f"vLLM /chat/completions 응답 오류 {resp.status_code}: {resp.text[:500]}"
            )

        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"/chat/completions 응답 JSON 파싱 실패: {e}")

        try:
            choices = data["choices"]
            if not choices:
                raise RuntimeError("vLLM 응답에 choices가 비어 있습니다.")
            message = choices[0].get("message", {})
            thinking_text, answer_text = _split_thinking_and_answer_from_message(message)
        except Exception as e:
            raise RuntimeError(f"vLLM 응답 구조 해석 실패: {e}\nRaw: {data}")

        print(f"[VLLMGenerator] model={model}, max_tokens={max_tokens}")
        return (answer_text, thinking_text,)
