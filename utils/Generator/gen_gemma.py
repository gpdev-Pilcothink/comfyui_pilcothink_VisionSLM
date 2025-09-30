import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


class Gemma3_4B_IT_Generator:
    def __init__(self, model, processor, device, dtype):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, local_path, device="cuda", dtype="bfloat16"):
        # accelerate가 있으면 device_map="auto"가 가장 안정적
        model = Gemma3ForConditionalGeneration.from_pretrained(local_path, device_map="auto").eval()
        processor = AutoProcessor.from_pretrained(local_path)
        return cls(model, processor, device, dtype)

    def _messages(self, img_path, prompt, system="You are a helpful assistant."):
        return [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image", "image": img_path},
                                         {"type": "text", "text": prompt}]},
        ]

    def generate(self, image, prompt, max_new_tokens=256, temperature=0.0, top_p=1.0):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name, format="PNG")
        tmp.flush()
        tmp.close()
        
        messages = self._messages(tmp.name, prompt)

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p if temperature > 0.0 else None,
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
