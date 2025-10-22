from transformers import AutoTokenizer, AutoModelForCausalLM
from rdflib import Graph, Literal, Namespace, URIRef
from sentence_transformers import SentenceTransformer, util
import torch, re


class Llama3B:
    def __init__(self):
        print("Initializing model and dataset...\n")

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.float16
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True
            )

        print("Model and dataset ready.\n")


    def ask(self, question, response, additional_info) -> str:
        """Generates query and returns both query and resolved results."""
        user_prompt= (
            f"please convert the following responses to one response in human like format, only provide the human like response without the question: the question was {question} and the response from sparql was {response}; "
            f"{'please note the additional infos: ' + str(additional_info) if additional_info else ''}"
        )


        messages = [
            {"role": "user", "content": user_prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=100,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs.shape[-1]:]
        query = self.tokenizer.decode(generated, skip_special_tokens=True)

        return query

    def askToPlaubalise(self, question, response, entities) -> str:
        """Generates query and returns both query and resolved results."""
        user_prompt = (
            f"The question was {question} and the response was {response}. The response was based on this entity: {entities}. "
        )

        messages = [
            {"role": "user", "content": user_prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=100,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs.shape[-1]:]
        query = self.tokenizer.decode(generated, skip_special_tokens=True)

        return query