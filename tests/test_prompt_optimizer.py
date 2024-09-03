import asyncio
import csv
from llmtext.messages_fns import agenerate


async def test_prompt_generator():
    from llmtext.prompt_optimizer import agenerate_prompt

    example_input = [
        """残高証明書を発行してください""",
        """プリペイドカードを作成してください""",
    ]
    example_output = [
        """# Source
japanese
# Target
korean
# Text
残高証明書を発行してください""",
        """# Source
japanese
# Target
korean
# Text
プリペイドカードを作成してください""",
    ]
    prompt = await agenerate_prompt(
        example_inputs=example_input,
        example_outputs=example_output,
    )

    print("prompt", prompt)


async def test_dataset_generator():
    inputs = [
        "残高証明書を発行してください",
        "プリペイドカードを作成してください",
        "口座を作ってください",
        "送金してください",
        "セキュリティカードの再発行をお願いします",
        "紛失したカードを報告したい",
        "口座残高を確認したい",
        "インターネットバンキングを登録したい",
        "外貨両替をしたい",
        "デビットカードを申し込みたい",
        "クレジットカードの限度額を上げたい",
    ]

    tasks = []

    for input in inputs:
        tasks.append(
            agenerate(
                messages=[
                    {
                        "role": "system",
                        "content": """Translate Japanese text to Korean while identifying the source and target languages.

Instructions: Given a Japanese sentence, provide the translation in Korean. Include a section for the source language, a section for the target language, and a section for the translated text. Use the format provided in the examples.

Example:
- Input: 残高証明書を発行してください
- Output:
  # Source
  japanese
  # Target
  korean
  # Text
  잔액 증명서를 발행해 주세요

- Input: プリペイドカードを作成してください
- Output:
  # Source
  japanese
  # Target
  korean
  # Text
  선불카드를 만들어 주세요""",
                    },
                    {"role": "user", "content": input},
                ]
            )
        )

    results = await asyncio.gather(*tasks)

    with open("dataset.csv", "a+") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        for result, input in zip(results, inputs):
            writer.writerow({"input": input, "output": result})


async def test_prompt_optimizer_run():
    from llmtext.prompt_optimizer import agenerate_prompt_and_optimize

    async def ascore_fn(
        input: list[str], output: list[str], reference_output: list[str]
    ) -> float:
        correct = 0
        wrong = 0

        for i, o, r in zip(input, output, reference_output):
            score = 1 if r == o else 0
            if score == 1:
                correct += 1
            else:
                wrong += 1
        return correct / (len(input) + 1)

    res = await agenerate_prompt_and_optimize(
        example_inputs=[
            "# Source\njapanese\n# Target\nkorean\n# Text\n잔액 증명서를 발행해 주세요",
            "# Source\njapanese\n# Target\nkorean\n# Text\n선불카드를 만들어 주세요",
            "# Source\njapanese\n# Target\nkorean\n# Text\n계좌를 개설해 주세요",
            "# Source\njapanese\n# Target\nkorean\n# Text\n송금해 주세요",
            "# Source\njapanese\n# Target\nkorean\n# Text\n보안카드 재발급을 부탁드립니다",
            "# Source\njapanese\n# Target\nkorean\n# Text\n분실한 카드를 신고하고 싶습니다",
            "# Source\njapanese\n# Target\nkorean\n# Text\n계좌 잔고를 확인하고 싶습니다",
            "# Source\njapanese\n# Target\nkorean\n# Text\n인터넷뱅킹을 등록하고 싶습니다",
            "# Source\njapanese\n# Target\nkorean\n# Text\n외화 환전을 하고 싶습니다",
            "# Source\njapanese\n# Target\nkorean\n# Text\n직불카드를 신청하고 싶습니다",
            "# Source\njapanese\n# Target\nkorean\n# Text\n신용카드의 한도를 올리고 싶습니다",
        ],
        example_outputs=[
            "잔액증명서를 발급해주세요",
            "선불카드 만들어주세요",
            "계좌 만들어주세요",
            "송금해주세요",
            "보안카드 재발급해주세요",
            "분실된 카드를 신고하고 싶어요.",
            "계좌 잔액을 확인하고 싶습니다",
            "인터넷 뱅킹을 등록하고 싶어요",
            "환전해 주세요",
            "체크카드를 신청하고 싶어요",
            "신용카드 한도를 올리고 싶습니다",
        ],
        scoring_fn=ascore_fn,
        parallel_count=10,
    )

    print("res", res)
