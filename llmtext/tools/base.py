import inspect
from typing import Any, Awaitable, Callable, get_type_hints
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import create_model


class Tool:
    def __init__(self, afn: Callable[..., Awaitable[Any]]) -> None:
        self.afn = afn
        self.name = afn.__name__

        # Get type hints for input and output
        type_hints = get_type_hints(afn)
        if "return" not in type_hints:
            raise Exception("Function must have a 'return' type hint")

        self.output = type_hints["return"]

        if not afn.__doc__:
            raise Exception("Tool needs description")
        self.description: str = afn.__doc__

        # Infer input type from the function's parameters
        self.sig = inspect.signature(afn)
        self.params = self.sig.parameters

    def to_openai_schema(self) -> ChatCompletionToolParam:
        fields = {}

        for name, param in self.params.items():
            # Determine the field type and default value
            field_type = param.annotation
            default_value = param.default if param.default is not param.empty else ...
            fields[name] = (field_type, default_value)

        # Create the Pydantic model dynamically
        DynamicModel = create_model("DynamicModel", **fields)

        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=DynamicModel.model_json_schema(),
            ),
            type="function",
        )
