from openai.types.chat.chat_completion_chunk import ChoiceDelta


def merge_deltas(original: ChoiceDelta, delta: ChoiceDelta) -> ChoiceDelta:
    original_dict = original.model_dump()
    for key, value in dict(delta).items():
        if value != None:
            if isinstance(value, str):
                if key in original_dict:
                    original_dict[key] = (original_dict[key] or "") + (value or "")
                else:
                    original_dict[key] = value
            else:
                value = dict(value)
                if key not in original_dict:
                    original_dict[key] = value
                else:
                    merge_deltas(original_dict[key], value)  # type: ignore

    return ChoiceDelta.model_validate(original_dict)
