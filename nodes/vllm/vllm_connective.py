# nodes/vllm_connective.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

try:
    import requests  # HTTP 통신용
except ImportError:
    requests = None


@dataclass
class VLLMConnection:
    """vLLM 연결 정보를 담는 데이터 클래스."""
    base_url: str           # 사용자가 입력한 서버 주소 (host:port 또는 host:port/v1)
    api_key: Optional[str]  # Authorization Bearer 토큰 (없으면 None)
    model: str              # 선택된 모델 이름
    models: List[str]       # /v1/models 에서 가져온 model id 목록


def _build_endpoint(base_url: str, path: str) -> str:
    """
    vLLM 서버의 base_url과 엔드포인트 path(/models, /chat/completions 등)를
    합쳐서 실제 호출 URL을 만들어 준다.

    - base_url 이 이미 .../v1 로 끝나면 그대로 사용
    - 아니라면 자동으로 /v1 을 붙인 후 path를 연결
    """
    if not base_url:
        raise ValueError("base_url이 비어 있습니다.")

    base = base_url.strip().rstrip("/")
    if not path.startswith("/"):
        path = "/" + path

    if base.endswith("/v1"):
        return base + path

    parsed = urlparse(base)
    if parsed.path.endswith("/v1"):
        return base + path

    return base + "/v1" + path


class VLLMConnective:
    """
    vLLM 서버 주소를 받아 /v1/models로 핑을 보내고
    사용 가능한 모델 리스트와 선택된 모델이 들어있는 연결 객체(VLLM_CONNECTION)를 반환하는 노드.

    - 입력:
        base_url: "http://192.168.x.x:8000" 또는 "http://192.168.x.x:8000/v1"
        api_key : (선택) vLLM 서버를 띄울 때 지정한 API 키 (없으면 기본값 EMPTY)
        model   : 사용할 모델 이름 (비우면 /models 결과의 첫 번째 모델 자동 선택)

    - 출력:
        connection : VLLM_CONNECTION (다음 노드에서 그대로 사용)
        models_text: STRING (선택된 모델 + 모델 리스트를 문자열로 반환)
    """

    CATEGORY = "Pilcothink/vLLM"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:8000/v1",
                        "multiline": False,
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "EMPTY",
                        "multiline": False,
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("VLLM_CONNECTION", "STRING",)
    RETURN_NAMES = ("connection", "models_text",)
    FUNCTION = "connect"
    OUTPUT_NODE = False

    def connect(self, base_url: str, api_key: str, model: str):
        if requests is None:
            raise RuntimeError(
                "Python 'requests' 패키지가 없습니다. "
                "ComfyUI가 설치된 파이썬 환경에 'pip install requests'로 설치해 주세요."
            )

        base_url = (base_url or "").strip()
        api_key = (api_key or "").strip()
        model = (model or "").strip()

        headers = {}
        if api_key and api_key.upper() != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        url = _build_endpoint(base_url, "/models")

        # Ping
        try:
            resp = requests.get(url, headers=headers, timeout=5)
        except Exception as e:
            raise RuntimeError(f"vLLM 서버({url})에 연결할 수 없습니다: {e}")

        if resp.status_code != 200:
            raise RuntimeError(
                f"vLLM 서버 /models 응답 오류 {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"/models 응답을 JSON으로 파싱하지 못했습니다: {e}")

        models: List[str] = []
        if isinstance(data, dict):
            items = data.get("data") or data.get("models") or []
            if isinstance(items, list):
                for m in items:
                    if isinstance(m, dict) and "id" in m:
                        models.append(str(m["id"]))

        if not models:
            raise RuntimeError("서버에 로드된 모델이 없습니다.")

        # 모델 선택
        if model:
            if model not in models:
                available = ", ".join(models)
                raise RuntimeError(
                    f"요청한 모델 '{model}' 이(가) vLLM 서버에서 발견되지 않았습니다.\n"
                    f"사용 가능한 모델: {available}"
                )
            selected_model = model
        else:
            selected_model = models[0]

        lines = [
            f"[선택된 모델] {selected_model}",
            "",
            "[서버에 로드된 모델]",
            *models,
        ]
        models_text = "\n".join(lines)

        print(
            f"[VLLMConnective] base_url={base_url}, "
            f"selected_model={selected_model}, models={models}"
        )
        conn = VLLMConnection(
            base_url=base_url,
            api_key=(api_key or None),
            model=selected_model,
            models=models,
        )
        return (conn, models_text,)
