# 토큰화 함수
def tokenize(text):
    return text.lower().split()


# 어휘 사전 구축 함수 - 이게 토크나이저?
def build_vocab(sentences):
    vocab = {}
    for sentence in sentences:
        tokens = tokenize(sentence)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1
    return vocab


# 문장을 숫자의 sequence로
def text_to_sequence(text, vocab):
    # dict.get(key, 기본값)...
    return [vocab.get(token, 0) for token in tokenize(text)]


sentences = [
    "Today is a sunny day",
    "Today is a rainy day",
]

vocab = build_vocab(sentences)
# 이게 시퀀스...그냥 단어 아이디 나열...
print(text_to_sequence(sentences[0], vocab))
print(text_to_sequence(sentences[1], vocab))

# OOV 토큰 사용하기...모르면 그냥 0?
test_data = ["Today is a snowy day", "Will it be rainy tomorrow?"]
print(text_to_sequence(test_data[0], vocab))
print(text_to_sequence(test_data[1], vocab))

# 패딩과 잘림
sentences += ["Is it sunny today?", "I really enjoyed walking in the snow today"]
print(sentences)
vocab = build_vocab(sentences)
for sentence in sentences:
    print(text_to_sequence(sentence, vocab))


def pad_sequence(sequences, max_len):
    return [
        seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
        for seq in sequences
    ]


for sentence in sentences:
    print(pad_sequence([text_to_sequence(sentence, vocab)], 10))
