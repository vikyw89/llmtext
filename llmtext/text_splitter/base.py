from nltk import tokenize
import nltk
import tiktoken

nltk.download("punkt")


class BaseTextSplitter:
    def __init__(self, max_token: int) -> None:
        self.max_token = max_token

    def run(self, text: str) -> list[str]:
        # do a check for token length
        token_count = self._token_counter(text=text)

        # if within max token, return the text
        if token_count < self.max_token:
            return [text]

        # if over max token, split the text
        lines = self._line_splitter(text=text)

        # merge the lines
        merged_lines = self._auto_merge_chunks(lines=lines, max_tokens=self.max_token)

        # return the merged text
        return merged_lines

    def _line_splitter(self, text: str) -> list[str]:
        return text.splitlines()

    def _sentence_splitter(self, text: str, language: str = "english") -> list[str]:
        sentences = tokenize.sent_tokenize(text=text, language=language)
        return sentences

    def _token_counter(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model)
        embeddings = encoding.encode(text=text)

        return len(embeddings)

    def _auto_merge_chunks(self, lines: list[str], max_tokens: int) -> list[str]:
        final = []
        current_token_count = 0
        for line in lines:
            line_token_count = self._token_counter(text=line)

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
