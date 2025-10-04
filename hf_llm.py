# hf_llm.py
from transformers import pipeline

class HFLLM:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.3B", device=-1, max_new_tokens=256):
        """
        model_name: Hugging Face model name
        device: -1 for CPU, 0 for GPU
        max_new_tokens: max tokens to generate
        """
        self.generator = pipeline("text-generation", model=model_name, device=device)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        """
        Generate text from the LLM given a prompt.
        Returns a plain string.
        """
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,   # deterministic (can make True if you want creativity)
            temperature=0.0,   # keep it factual
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"][len(prompt):].strip()
