from enum import Enum
from dotenv import load_dotenv, dotenv_values
import asyncio
from openai import OpenAI

load_dotenv()

secrets = dotenv_values("secrets.env")
class Models(Enum):
    gpt_6_luna = "gpt_6_luna"

class ModelInterface:
    def __init__(self, model: Models, system_prompt="", temperature=0.8, max_output_tokens=100):
        self.model=model
        self.system_prompt=system_prompt
        self.temperature=temperature
        self.max_output_tokens=max_output_tokens
        self.conversation_history = []

        if self.model == Models.gpt_6_luna:
            self.client = OpenAI(api_key=secrets['openai_key'])

    def request(self, text: str):
        match self.model:
            case Models.gpt_6_luna:
                return self.gpt_6_luna_request(text)

    def gpt_6_luna_request(self, text):
        response = self.client.chat.completions.create(
            model="gpt-6-luna",
            messages=[{"role": "user", "content": text}],
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    client = ModelInterface(Models.gpt_6_luna)
    print(client.request("Say hello world!"))
