import os
import torch
from transformers import AutoModelForCausalLM

from .. import backend_aliases
backend_aliases.setup_backend_path() 

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


class DeepseekVL13BGenerator:
    def __init__(self, model, processor, tokenizer, device, dtype):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, local_path, device="cuda", dtype="bfloat16"):
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16 if device == "cuda" else torch.float32)

        processor: VLChatProcessor = VLChatProcessor.from_pretrained(local_path)
        tokenizer = processor.tokenizer

        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            local_path, trust_remote_code=True
        )
        if device == "cuda":
            model = model.to(torch_dtype).cuda().eval()
        else:
            model = model.to(torch_dtype).cpu().eval()

        return cls(model, processor, tokenizer, device, dtype)

    def _make_conversation(self, img_path, prompt):
        return [
            {"role": "User", "content": f"<image_placeholder>{prompt}", "images": [img_path]},
            {"role": "Assistant", "content": ""},
        ]

    def generate(self, image, prompt, max_new_tokens=256, temperature=0.0, top_p=1.0):
        # 이미지 임시 파일로 저장(라이브러리 유틸이 경로 기반으로 동작)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name, format="PNG")
        tmp.flush()
        tmp.close()
        
        conv = self._make_conversation(tmp.name, prompt)

        pil_images = load_pil_images(conv)
        prepare_inputs = self.processor(
            conversations=conv,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            use_cache=True,
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # prepare_inputs['sft_format'][0] 접두어 제거
        prefix = prepare_inputs["sft_format"][0]
        if answer.startswith(prefix):
            answer = answer[len(prefix):].lstrip()
        return answer
