import regex as re


PATTERN = r"""'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def pretokenize(text:str) -> list[str]:
    """
    Pre processing of text before using regex before tokenization.
    This is to ensure BPE merges never happen across word boundaries, always within word chunks
    """

    return re.findall(PATTERN, text)





if __name__ == "__main__" :
    tests = [
        "Hello, world!",
        "I don't think so...",
        "def foo(x):\n    return x + 1",
        '{"key": "value", "num": 42}',
        "  multiple   spaces  ",
        "Hello\nworld\n",
        """
        for i in range(1, 101):
            if i % 3 == 0 and i % 5 == 0:
                print("FizzBuzz")
            elif i % 3 == 0:
                print("Fizz")
            elif i % 5 == 0:
                print("Buzz")
            else:
                print(i)
        """,
        "만나서 반가워요. 저는 OpenAI에서 개발한 대규모 언어 모델인 ChatGPT입니다. 궁금한 것이 있으시면 무엇이든 물어보세요."
    ]

    # for test in tests:
    #     chunks = pretokenize(test)
    #     print(f"Input: {repr(test)}")
    #     print(f"Chunks: {chunks}\n")


    with open("./training_data/raw/openwebtext.txt", "r", encoding="utf-8") as f:
        first_line = f.readline()

    chunks = pretokenize(first_line)
    for i, chunk in enumerate(chunks[:20]):
        print(f"{i} : {repr(chunk)}")
