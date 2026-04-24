from helper import *
import json
import torch

with open("data/sarcasm.json", "r") as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])

    sentence = item["headline"].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    # 책 소스에서는 따로 bs4, stopword 적용해서 하는데, 나는 책 helper에서 만들어 둔 함수 이용...
    sentence = " ".join(tokenize(sentence))
    sentences.append(sentence)

# 훈련 테스트 데이터셋 만들기...
# max_length = 100
max_length = 85
training_size = 23000

# 훈련과 테스트 나누고
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

# 정답지 라벨은 바로 텐서로...
training_labels = labels[0:training_size]
training_labels = torch.tensor(training_labels, dtype=torch.float32)
testing_labels = labels[training_size:]
testing_labels = torch.tensor(testing_labels, dtype=torch.float32)

# 어휘 사전은 훈련에만 만들고 - helper의 함수를 수정해서 단어 수 제한해서 과대적합 줄이기...
vocab_size = 2000
word_index = build_vocab(training_sentences, vocab_size)
print(len(word_index))

test_sentences = [
    "granny starting to fear spiders in the garden might be real",
    "game of thrones season finale showing this sunday night",
    "PyTorch book will be a best seller",
]
print(texts_to_sequences(test_sentences, word_index))
