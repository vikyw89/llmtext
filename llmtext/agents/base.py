from openai import AsyncOpenAI


class BaseAgent:
    def __init__(self, name, age):
        self.name = name
        self.age = age
