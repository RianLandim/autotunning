"""
Inference backends for the criteria matcher.

Both expose the same interface, chat(conversations) -> list[str], so the
pipeline in matcher.py runs unchanged locally and on AWS:

- LocalBackend: transformers on the local GPU (4-bit by default, to fit 8 GB).
- SageMakerBackend: a SageMaker real-time endpoint serving the model with
  vLLM (LMI or Hugging Face vLLM container), called with the OpenAI chat
  completions payload. See deploy_sagemaker.py.
"""

import json
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class LocalBackend:
    def __init__(self, model_id=DEFAULT_MODEL, load_in_4bit=True, batch_size=4, max_new_tokens=512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        self.name = f"local:{model_id}{' (4-bit)' if load_in_4bit else ''}"

    def chat(self, conversations, max_new_tokens=None):
        max_new_tokens = max_new_tokens or self.max_new_tokens
        prompts = [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations]
        outputs = []
        for i in range(0, len(prompts), self.batch_size):
            batch = self.tokenizer(prompts[i:i + self.batch_size], return_tensors="pt", padding=True,
                                   add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                generated = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                                pad_token_id=self.tokenizer.pad_token_id)
            new_tokens = generated[:, batch["input_ids"].shape[1]:]
            outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs


class SageMakerBackend:
    def __init__(self, endpoint_name, region=None, max_workers=8, max_new_tokens=512):
        import boto3

        self.client = boto3.client("sagemaker-runtime", region_name=region)
        self.endpoint_name = endpoint_name
        self.max_workers = max_workers
        self.max_new_tokens = max_new_tokens
        self.name = f"sagemaker:{endpoint_name}"

    def _invoke(self, messages, max_new_tokens):
        payload = {"messages": messages, "max_tokens": max_new_tokens, "temperature": 0}
        response = self.client.invoke_endpoint(EndpointName=self.endpoint_name, ContentType="application/json",
                                               Body=json.dumps(payload))
        body = json.loads(response["Body"].read())
        if "choices" in body:  # OpenAI chat completions schema
            return body["choices"][0]["message"]["content"]
        return body["generated_text"]

    def chat(self, conversations, max_new_tokens=None):
        max_new_tokens = max_new_tokens or self.max_new_tokens
        # the endpoint batches concurrent requests on the server side (continuous batching)
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(lambda c: self._invoke(c, max_new_tokens), conversations))
