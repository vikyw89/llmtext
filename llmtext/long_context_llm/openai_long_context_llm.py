from typing import Annotated
from pydantic import BaseModel, Field
from llmtext.llms.base import BaseLLM
from llmtext.llms.openai import OpenAILLM
from llmtext.long_context_llm.base import BaseLongContextLLM
from llmtext.text_splitter.base import BaseTextSplitter
from llmtext.chat_llms import BaseChatLLM, ChatOpenAI
import asyncio

class OpenaiLongContextLLM(BaseLongContextLLM):
    def __init__(self, llm: BaseLLM = OpenAILLM(), chat_llm: BaseChatLLM =ChatOpenAI(), text_splitter:BaseTextSplitter = BaseTextSplitter(max_token=12000), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.chat_llm = chat_llm
        self.text_splitter = text_splitter


    async def arun(self, text: str) -> str:
        summarized_text =  await self.summarize(text=text, max_token=int(self.text_splitter.max_token / 2))

        return await self.llm.arun(text=summarized_text)

    async def aembed(self, text: str) -> str:
        pass
    
    async def summarize(self, text: str, max_token: int = 12000) -> str:
        # base case
        token_count = self.text_splitter._token_counter(text=text)
        if token_count < self.text_splitter.max_token:
            return text
        
        # recursive case
        splitted_text = self.text_splitter.run(text=text)
        class Summary(BaseModel):
            """Summary of the text"""
            summary: Annotated[str, Field(description="Summary of the text")]

        gather_summaries = []
        for text in splitted_text:
            self.chat_llm.messages = []
            self.chat_llm.add_message(message={"role": "user", "content": text})
            gather_summaries.append(self.chat_llm.astructured_extraction(output_class=Summary))
    
        summaries : list[Summary] = await asyncio.gather(*gather_summaries)

        text_summaries = ""
        for summary in summaries:
            text_summaries += summary.summary + "\n"

        # bubble up
        return await self.summarize(text=text_summaries)

    