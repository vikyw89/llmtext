from typing import Any, Awaitable, Callable, get_type_hints
from pydantic import BaseModel
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition


class Tool:
    def __init__(self, afn: Callable[[Any], Awaitable[Any]]) -> None:
        self.afn = afn
        self.name = afn.__name__
        self.input: BaseModel = get_type_hints(afn)["input"]
        self.output = get_type_hints(afn)["return"]
        if not afn.__doc__:
            raise Exception("Tool needs description")
        self.description: str = afn.__doc__

    def to_openai_schema(self) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=self.input.model_json_schema(),
            ),
            type="function",
        )
