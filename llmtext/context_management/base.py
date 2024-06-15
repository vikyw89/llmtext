from nltk import tokenize
import nltk
import tiktoken

nltk.download("punkt")


class BaseContextManagement:
    def __init__(self) -> None:
        pass

    def line_splitter(self, text: str) -> list[str]:
        return text.splitlines()

    def sentence_splitter(self, text: str, language: str = "english") -> list[str]:
        sentences = tokenize.sent_tokenize(text=text, language=language)
        return sentences

    def token_counter(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model)
        embeddings = encoding.encode(text=text)

        return len(embeddings)

    def auto_merge_chunks(self, lines: list[str], max_tokens: int) -> list[str]:
        final = []
        current_token_count = 0
        for line in lines:
            line_token_count = self._get_token_count(text=line)

            # if final is empty, merge it
            if len(final) == 0:
                final.append(line)
                current_token_count = line_token_count
                continue

            # if final is not empty, check if the combined tokens are less than max_tokens
            if line_token_count + current_token_count < max_tokens:
                final[-1] += f"\n{line}"
                current_token_count += line_token_count
                continue

            # if the combined tokens are greater than max_tokens, split the line
            final.append(line)
            current_token_count = line_token_count

        return final
