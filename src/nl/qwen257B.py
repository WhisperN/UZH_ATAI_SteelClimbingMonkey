# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

class Qwen257B:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    def ask(self, question):
        """
        Input question is translated into a sparql query
        """

        messages = [
            {
                "role": "user",
                "content": "Generate a SPARQL query from this question:"
                           "'''"
                           f"{question}"
                           "'''",
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=40)
        # Optional string parsing
        # print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
        return self.tokenizer.decode(outputs[0])