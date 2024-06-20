from llmtext.agents.base import BaseAgent


class SimpleAgent(BaseAgent):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    async def astream(self) -> str: