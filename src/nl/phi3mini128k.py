from transformers import AutoTokenizer, AutoModelForCausalLM

class Phi3Mini128k:
	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
		self.model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)

	def ask(self, msg):
		"""
		Input question produces output that classifies the data according to:
		- Factual
		- Embedded
		"""
		messages = [
			{
				"role": "user",
			 	"content": "Classify a question according to these types: "
						   "Factual, Embedded"
						   "The question:"
						   "'''"
						   f"{msg}"
						   "'''"
						   "Please only respond with either Factual or Embedded"
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
		# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
		return self.tokenizer.decode(outputs[0])