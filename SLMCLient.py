from openai import OpenAI

class SLMClient:
	def __init__(self, api_base: str, api_key: str, model: str, temperature: float = 0.0) -> None:
		if not api_base or not api_key:
			raise ValueError("api_base and api_key are required for SLM calls.")
		self.api_base = api_base.rstrip("/")
		self.api_key = api_key
		self.model = model
		self.temperature = temperature
		self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

	def generate(self, system_prompt: str, user_prompt: str) -> str:
		messages = []
		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})
		messages.append({"role": "user", "content": user_prompt})
		response = self.client.chat.completions.create(
			model=self.model,
			messages=messages,
			temperature=self.temperature,
		)

		if not response.choices:
			raise RuntimeError(f"Unexpected SLM response: {response}")

		content = response.choices[0].message.content
		if content is None:
			raise RuntimeError(f"No content found in SLM response: {response}")
		return content