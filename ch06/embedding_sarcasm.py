import json
import matplotlib.pyplot as plt
from helper import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary


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
print(len(word_index))
training_sequences = texts_to_sequences(training_sentences, word_index)
training_padded = pad_sequences(training_sequences, max_length)

testing_sequences = texts_to_sequences(testing_sentences, word_index)
testing_padded = pad_sequences(testing_sequences, max_length)

# 많이 나오는 단어...근데 여기 트럼프가 있네...
word_frequency = word_frequency(training_sentences, word_index)
for i in range(10):
    print(f"{list(word_index.keys())[i]}: {word_frequency[list(word_index.keys())[i]]}")


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=24):
        super(TextClassificationModel, self).__init__()
        # 단어 ID를 embedding dim(속성 차원들)에 난수로 시작되는 벡터에 맞추기...
        # 앞에가 vocab_size인걸 보면 단어 ID를 원핫 벡터로 인식하는 듯...
        # 예를 들면 (배치, 어휘 수) x (어휘 수, 임베딩 차원) -> (배치, 임베딩 차원)으로...
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 임베딩 층을 밀집 층에 연결한다...임베딩은 밀집이 아니다?...입력의 차원을 크기가 1인 고정 벡터로 줄인다?
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 이건 (배치, 임베딩 차원) x (임베딩 차원, 24) -> (배치, 24)로...
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        # 이건 (배치, 24) x (24, 1) -> (배치, 1)로
        self.fc2 = nn.Linear(hidden_dim, 1)
        # 그 1이 조롱이냐 아니냐를 판별
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 입력 x는 (batch_size, max_length)
        embedded = self.embedding(x)
        # 거쳐 나오면 (batch_size, max_length, embedding_dim)
        # 뒤에 올 AdaptiveAvgPool1d가 (batch_size, embedding_dim, max_length)를 요구하니까...그에 맞춰 축 바꿈...
        embedded = embedded.transpose(1, 2)
        pooled = self.avg_pool(embedded).squeeze(2)
        # 거쳐 나오면 (batch_size, embedding_dim)
        x = self.fc1(pooled)
        # 거쳐 나오면 (batch_size, 24)
        x = self.relu(x)
        x = self.fc2(x)
        # 거쳐 나오면 (batch_size, 1)
        return self.sigmoid(x)


# 하이퍼파라미터 설정
vocab_size = len(word_index)
# 책 소스에서 16과 100이 있는데, 일단 순서대로 16부터...
embedding_dim = 16

model = TextClassificationModel(vocab_size, embedding_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
print(model)
batch_size = 32
summary(
    model,
    input_size=(batch_size, max_length),
    # 뭐지? 이거 때문에 되고 안되고가...임베딩 층에 타입을 지정한다는데...
    dtypes=[torch.long],
)
