import json
import matplotlib.pyplot as plt
from helper import *


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

# 이건 책에 없고 소스에만 있는데...긴 문장 순서대로 단어 수 보기...
xs, ys = [], []
current_item = 1
for item in sentences:
    xs.append(current_item)
    ys.append(len(item))
    current_item += 1

# newys = sorted(ys)
# plt.plot(xs, newys)
# plt.axis([20000, 27000, 50, 250])
# plt.show()

# 훈련 테스트 데이터셋 만들기...
max_length = 100
training_size = 23000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

word_index = build_vocab(training_sentences)
training_sequences = texts_to_sequences(training_sentences, word_index)
training_padded = pad_sequences(training_sequences, max_length)

testing_sequences = texts_to_sequences(testing_sentences, word_index)
testing_padded = pad_sequences(testing_sequences, max_length)

# 많이 나오는 단어...근데 여기 트럼프가 있네...
word_frequency = word_frequency(training_sentences, word_index)
for i in range(10):
    print(f"{list(word_index.keys())[i]}: {word_frequency[list(word_index.keys())[i]]}")
