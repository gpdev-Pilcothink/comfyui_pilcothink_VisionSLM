import os

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
    CATEGORY = "Custom/Text"

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
                "text1": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_texts"
    CATEGORY = "Custom/Text"

    def return_texts(self, text1):
        return (text1,)