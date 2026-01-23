# nodes/ollama_connective.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

try:
    import requests  # HTTP 통신용
except ImportError:
    requests = None


@dataclass
class OllamaConnection:
    """Ollama 연결 정보를 담는 데이터 클래스."""
    base_url: str             # Ollama 서버 주소
    api_key: Optional[str]    # Authorization Bearer 토큰 (없으면 None)
    model: str                # 선택된 모델 이름
    models: List[str]         # /api/tags 에서 가져온 모델 이름 목록


def _build_ollama_endpoint(base_url: str, path: str) -> str:
    """
    Ollama 서버의 base_url 과 엔드포인트 path(/tags, /generate 등)를
    합쳐서 실제 호출 URL을 만들어 준다.
    """
    if not base_url:
        raise ValueError("base_url이 비어 있습니다.")

    base = base_url.strip().rstrip("/")
    if not path.startswith("/"):
        path = "/" + path

    if base.endswith("/api"):
        return base + path

    parsed = urlparse(base)
    if parsed.path.endswith("/api"):
        return base + path

    return base + "/api" + path


class OllamaConnective:
    """
    Ollama 서버 주소를 받아 /api/tags 로 핑을 보내고
    사용 가능한 모델 리스트와 선택된 모델이 들어있는 연결 객체(OLLAMA_CONNECTION)를 반환하는 노드.
    """

    CATEGORY = "Pilcothink/Ollama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:11434",
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

    RETURN_TYPES = ("OLLAMA_CONNECTION", "STRING",)
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

        api_key_real = ""
        if api_key and api_key.upper() != "EMPTY":
            api_key_real = api_key

        headers = {}
        if api_key_real:
            headers["Authorization"] = f"Bearer {api_key_real}"

        url = _build_ollama_endpoint(base_url, "/tags")

        try:
            resp = requests.get(url, headers=headers, timeout=10)
        except Exception as e:
            raise RuntimeError(f"Ollama 서버({url})에 연결할 수 없습니다: {e}")

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama 서버 /tags 응답 오류 {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"/tags 응답을 JSON으로 파싱하지 못했습니다: {e}")

        models: List[str] = []
        if isinstance(data, dict):
            items = data.get("models") or []
            if isinstance(items, list):
                for m in items:
                    if isinstance(m, dict) and "name" in m:
                        models.append(str(m["name"]))

        if not models:
            raise RuntimeError(
                "Ollama 서버에서 사용할 수 있는 모델이 없습니다.\n"
                "먼저 `ollama pull <모델이름>` 으로 모델을 설치해 주세요."
            )

        if model:
            if model not in models:
                available = ", ".join(models)
                raise RuntimeError(
                    f"요청한 모델 '{model}' 이(가) Ollama 서버에서 발견되지 않았습니다.\n"
                    f"사용 가능한 모델: {available}"
                )
            selected_model = model
        else:
            selected_model = models[0]

        lines = [
            f"[선택된 모델] {selected_model}",
            "",
            "[사용 가능한 모델]",
            *models,
        ]
        models_text = "\n".join(lines)

        print(
            f"[OllamaConnective] base_url={base_url}, "
            f"selected_model={selected_model}, models={models}"
        )
        conn = OllamaConnection(
            base_url=base_url,
            api_key=(api_key_real or None),
            model=selected_model,
            models=models,
        )
        return (conn, models_text,)
