import torch
from transformers import AutoProcessor
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

from .. import backend_aliases
backend_aliases.setup_backend_path()

from qwen_vl_utils.src.qwen_vl_utils.vision_process import process_vision_info


class QwenVLGenerator:
    def __init__(self, model, processor, device, dtype, version="2.5"):
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.version = version

    @classmethod
    def from_pretrained(cls, local_path, version="2.5", device="cuda", dtype="bfloat16"):
        model_cls = Qwen2_5_VLForConditionalGeneration if version == "2.5" else Qwen2VLForConditionalGeneration

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, "auto")

        if torch_dtype == "auto":
            model = model_cls.from_pretrained(local_path, torch_dtype="auto", device_map="auto")
        else:
            if device == "cuda":
                model = model_cls.from_pretrained(local_path, torch_dtype=torch_dtype, device_map="auto")
            else:
                model = model_cls.from_pretrained(local_path, torch_dtype=torch.float32, device_map={"": "cpu"})

        processor = AutoProcessor.from_pretrained(local_path)
        return cls(model, processor, device, dtype, version)

    def generate(self, image, prompt, max_new_tokens=256, temperature=0.0, top_p=1.0):
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name, format="PNG")
        tmp.flush()
        tmp.close()

        messages = self._messages(tmp.name, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if self.version == "2.5":
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt", **video_kwargs
            )
        else:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            )

        inputs = inputs.to(self.model.device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
        )
        gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
        text_out = self.processor.batch_decode(
            gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return text_out
