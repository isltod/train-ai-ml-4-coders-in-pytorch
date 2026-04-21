import torch

# vscode 코드 추천에 BertTokenizerFast가 뜨질 않는다...동작은 문제 없는데...
from transformers import BertTokenizerFast

sentences = [
    "Today is a sunny day",
    "Today is a rainy day",
]


# 사용자 정의 토크나이저로 기본 골격 익히고...
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


# 어휘 사전 만들기
vocab = build_vocab(sentences)
print("어휘 사전:", vocab)

# BertTokenizerFast 이용 - bert-base-uncased가 모델 이름인 듯...
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# return_tensors="pt" 옵션이 파이토치 텐서로 반환 지정...
encoded_inputs = tokenizer(
    sentences, padding=True, truncation=True, return_tensors="pt"
)
# 이게 어떻게 생겼는지...
tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in encoded_inputs["input_ids"]]
print("토큰:", tokens)
# 입력되서 변환된 문장은 input_ids 키로
print("토큰 ID:", encoded_inputs["input_ids"])
# 이건 그냥 통으로 BertTokenizerFast의 단어사전 같고...30,522개
word_index = tokenizer.get_vocab()
print(len(word_index))
print("어휘 사전:", dict(list(word_index.items())[:10]))
