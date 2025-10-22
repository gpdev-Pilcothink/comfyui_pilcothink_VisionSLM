import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from ... import backend_aliases

backend_aliases.setup_backend_path()

from qwen_vl_utils.src.qwen_vl_utils.vision_process import process_vision_info


class QwenVL3Generator:
    def __init__(self, model, processor, device, dtype):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, local_path, device="auto", dtype="auto"):
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.float16 if device == "cuda" else torch.float32)

        if device == "cuda":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                local_path, torch_dtype=torch_dtype, device_map="auto"
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                local_path, torch_dtype=torch_dtype, device_map={"": "cpu"}
            )

        processor = AutoProcessor.from_pretrained(local_path)
        return cls(model, processor, device, dtype)

    def _messages(self, img_path, prompt, system=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
        msgs.append({
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt},
            ],
        })
        return msgs

    def generate(self, image, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name, format="PNG")
        tmp.flush()
        tmp.close()

        messages = self._messages(tmp.name, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            top_k=top_k if temperature > 0 else None,             # ← top_k 추가
            repetition_penalty=repetition_penalty if repetition_penalty > 0 else None,
        )

        gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
        text_out = self.processor.batch_decode(
            gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text_out