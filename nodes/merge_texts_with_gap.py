import re
from typing import Any, Dict

class MergeTextsWithGap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"multiline": True, "default": ""}),
                "text2": ("STRING", {"multiline": True, "default": ""}),
                "text3": ("STRING", {"multiline": True, "default": ""}),
                "text4": ("STRING", {"multiline": True, "default": ""}),
                "text5": ("STRING", {"multiline": True, "default": ""}),
                "text6": ("STRING", {"multiline": True, "default": ""}),
                "gap": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_texts"
    CATEGORY = "Pilcothink/Text"

    def merge_texts(self, text1, text2, text3, text4, text5, text6, gap):
        # 줄바꿈 문자 반복
        separator = "\n" * gap
        # 빈 문자열은 제외하고 합치기
        texts = [t for t in [text1, text2, text3, text4, text5, text6] if t.strip() != ""]
        result = separator.join(texts)
        return (result,)
    

class String_Text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "return_texts"
    CATEGORY = "Pilcothink/Text"

    def return_texts(self, text):
        return (text,)
    

class RemoveSpecificPatterns:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter text here..."
                }),
            },
            "optional": {
                "exclude_tags": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Tags to remove (comma or space separated)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "Pilcothink/Text"

    def remove_specific_patterns(self, text: str, exclude_tags: str = "") -> str:
        text = text.lower()

        if not exclude_tags:
            return text.strip()

        # 쉼표나 공백 기준으로 split → ['simple', 'black']
        patterns = [tag.strip().lower() for tag in re.split(r"[,\s]+", exclude_tags) if tag]

        # 정규식 생성
        regex = r"\b(" + "|".join(map(re.escape, patterns)) + r")\b"
        modified_text = re.sub(regex, "", text)

        # 공백 정리
        return re.sub(r"\s+", " ", modified_text).strip()

    def process(self, text: str, exclude_tags: str = ""):
        cleaned = self.remove_specific_patterns(text, exclude_tags)
        return (cleaned,)

# 노드 등록
NODE_CLASS_MAPPINGS = {
    "RemoveSpecificPatterns": RemoveSpecificPatterns
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveSpecificPatterns": "🧹 Remove Specific Patterns"
}
