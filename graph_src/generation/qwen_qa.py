import gc
import os
import torch
import warnings
warnings.filterwarnings("ignore")


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from typing import List
from PIL import Image


class TableQAGenerator:


    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        load_in_4bit: bool = False
    ):
        print(f"Loading VLM model from {model_path}...")


        if "2.5" in model_path:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen2VLForConditionalGeneration as ModelClass

        from transformers import AutoProcessor

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            self.model = ModelClass.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            self.model = ModelClass.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto", 
            )

        self.processor = AutoProcessor.from_pretrained(model_path)

        self.processor.image_processor.min_pixels = 256 * 28 * 28
        self.processor.image_processor.max_pixels = 1280 * 28 * 28
        self.device = device
        print(f"VLM loaded: {model_path} (device={device}, "
              f"max_pixels={self.processor.image_processor.max_pixels})")

    def answer_with_patches(
        self,
        query: str,
        patches: List[Image.Image],
        max_new_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:

        content = []
        for patch in patches:
            content.append({"type": "image", "image": patch})

        strict_prompt = (
            "You are a strict data extraction AI analyzing table images. "
            "Answer the question based ONLY on the visible data in the provided patches. "
            "Provide the shortest possible exact answer. "
            "Do NOT use full sentences. Do NOT explain. Include units, currency symbols ($, %), and formatting exactly as shown in the table.\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        content.append({
            "type": "text",
            "text": strict_prompt
        })

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=patches,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        input_len = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_len:]
        answer = self.processor.decode(new_tokens, skip_special_tokens=True).strip()


        del inputs, outputs, new_tokens
        gc.collect()
        torch.cuda.empty_cache()

        return answer

    def answer_with_context(
        self,
        query: str,
        full_image: Image.Image,
        patches: List[Image.Image],
        max_new_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:

        content = []
        all_images = []

        content.append({"type": "image", "image": full_image})
        all_images.append(full_image)

        for patch in patches:
            content.append({"type": "image", "image": patch})
            all_images.append(patch)

        n_patches = len(patches)
        strict_prompt = (
            "You are analyzing a table. The FIRST image is the full table for context. "
            f"The next {n_patches} image(s) are zoomed-in patches of the most relevant regions. "
            "Use the full table to understand headers and structure. "
            "Use the patches for detailed data extraction. "
            "Answer the question with the shortest possible exact answer. "
            "Do NOT use full sentences. Do NOT explain. "
            "Include units, currency symbols ($, %), and formatting exactly as shown in the table.\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        content.append({"type": "text", "text": strict_prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=all_images,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        input_len = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_len:]
        answer = self.processor.decode(new_tokens, skip_special_tokens=True).strip()


        del inputs, outputs, new_tokens
        gc.collect()
        torch.cuda.empty_cache()

        return answer

    def generate_answer(
        self,
        patches: List[Image.Image],
        question: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0
    ) -> str:
        return self.answer_with_patches(question, patches, max_new_tokens, temperature)



QwenQA = TableQAGenerator
