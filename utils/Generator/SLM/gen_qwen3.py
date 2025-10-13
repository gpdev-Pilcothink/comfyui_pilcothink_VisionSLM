import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3Generator:
    def __init__(self, model, tokenizer, device, dtype):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, model_name, device="cuda", dtype="float16"):
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.float16 if device == "cuda" else torch.float32)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(model, tokenizer, device, dtype)

    def _messages(self, prompt, system=None):
        msgs = []
        if system:
            msgs.append({
                "role": "system",
                "content": [{"type": "text", "text": system}]
            })
        msgs.append({
            "role": "user",
            "content": prompt
        })
        return msgs

    def generate(self, prompt, system=None, max_new_tokens=512, temperature=0.7, top_p=0.9):
            messages = self._messages(prompt, system)

            # chat template (thinking 모드 ON 상태에서도 처리 가능)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # thinking 생성은 허용하되, 아래에서 제거
            )

            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
            )

            # ✅ 공식 예제처럼 thinking 제거 로직
            output_ids = gen_ids[0][len(inputs.input_ids[0]):].tolist()
            try:
                # </think> 토큰 ID (공식 기준)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            # reasoning 부분 (선택적, 필요 없으면 안 써도 됨)
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")

            # 최종 답변 부분만 남김
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            return content