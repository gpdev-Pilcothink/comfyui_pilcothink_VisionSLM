# comfyUI_pilcothink_VisionSLM

Custom ComfyUI nodes to run SLM Vision models (DeepSeek-vl 1.3b chat, Qwen2.5-vl 3b, Gemma-3-4b-it) with optional RAG support.
<img width="2645" height="1288" alt="image" src="https://github.com/user-attachments/assets/739c01a6-ac6b-4067-97f7-4ace33c8536c" />



- Models are downloaded into `Models/SLM_Vision/` when selected in the node.


# LICENSE
-utils/backends/DeepSeek-vl 
https://github.com/deepseek-ai/DeepSeek-VL, MIT LICENSE

-utils/backends/qwen_vl_utils
https://github.com/QwenLM/Qwen3-VL, Apache-2.0 license


# Tips

If you choose the CPU option on the device, it will only work with bloat16. Other dtypes will result in errors.

You can enable RAG functionality by placing your data in the rag_doc folder in .txt format.

Please share any suggestions for improvements in the Issues section.

Moondream may be updated in the future. If you are aware of any other low-cost vision models, I would appreciate it if you could let me know.
