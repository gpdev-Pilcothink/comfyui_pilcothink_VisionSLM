# comfyUI_pilcothink_VisionSLM

Custom ComfyUI nodes to run SLM Vision models (DeepSeek-vl 1.3b chat, Qwen2-vl-2b-Instruct, Qwen2.5-vl 3b, Gemma-3-4b-it) with optional RAG support.
+
Support SLM Model (Qwen3-0.6b)
<img width="2645" height="1288" alt="image" src="https://github.com/user-attachments/assets/739c01a6-ac6b-4067-97f7-4ace33c8536c" />



- Models are downloaded into `Models/SLM_Vision/` when selected in the node.


# LICENSE
-utils/backends/DeepSeek-vl 
https://github.com/deepseek-ai/DeepSeek-VL, MIT LICENSE

-utils/backends/qwen_vl_utils
https://github.com/QwenLM/Qwen3-VL, Apache-2.0 license


# Tips

(1) If you choose the CPU option on the device, it will only work with bfloat16. Other dtypes will result in errors.

(2) You can enable RAG functionality by placing your data in the rag_doc folder in .txt format.

(3) Since gemma-3-4b-it cannot be accessed on Hugging Face without logging in, you will need to configure it separately, or alternatively, download it directly from the repository and place it in the Models folder.


#Thank you
Please share any suggestions for improvements in the Issues section.

Moondream2 may be updated in the future. If you are aware of any other low-cost vision models, I would appreciate it if you could let me know.
