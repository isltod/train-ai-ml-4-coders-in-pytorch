from bs4 import BeautifulSoup
from collections import Counter
import torch.nn as nn
import string
import sys

sys.path.append(".")
from utils import stopwords


def tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    tokens = cleaned_text.lower().split()
    # 이 부분은 책의 helper에 없고 sarcasm 예제에 따로 붙어있던 부분인데...
    # - , . 등을 단어에서 없애는 역할...maketrans(기존, 변경, 제외)
    table = str.maketrans("", "", string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    filtered_tokens = [word for word in tokens if word not in stopwords]
    return filtered_tokens


def build_vocab(texts, max_words=10000):
    counter = Counter()
    for sentence in texts:
        tokens = tokenize(sentence)
        counter.update(tokens)

    # 이게 아예 사라지고...
    # sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # 빈도수 높은 순서로 최대 max_words - 2(pad, unk) 까지만 선택해서 사전 만들기
    most_common_words = counter.most_common(max_words - 2)

    # 그래서 인덱스는 2부터 시작
    vocab = {word: i + 2 for i, (word, _) in enumerate(most_common_words)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab


def texts_to_sequences(texts, word_index):
    sequences = []
    for sentence in texts:
        sequence = []
        for word in sentence.split():
            if word in word_index:
                sequence.append(word_index[word])
        sequences.append(sequence)
    return sequences


def pad_sequences(sequences, max_length):
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) < max_length:
            padded_seq = sequence + [0] * (max_length - len(sequence))
        else:
            padded_seq = sequence[:max_length]
        padded_sequences.append(padded_seq)
    return padded_sequences


def word_frequency(texts, word_dict):
    frequency = {word: 0 for word in word_dict}
    for sentence in texts:
        words = sentence.lower().split()
        for word in words:
            if word in frequency:
                frequency[word] += 1
    return frequency


# 모델 클래스 정의...
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=24, dropout_rate=0.25):
        super(TextClassificationModel, self).__init__()
        # 단어 ID를 embedding dim(속성 차원들)에 난수로 시작되는 벡터에 맞추기...
        # 앞에가 vocab_size인걸 보면 단어 ID를 원핫 벡터로 인식하는 듯...
        # 예를 들면 (배치, 어휘 수) x (어휘 수, 임베딩 차원) -> (배치, 임베딩 차원)으로...
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 임베딩 층을 밀집 층에 연결한다...임베딩은 밀집이 아니다?...입력의 차원을 크기가 1인 고정 벡터로 줄인다?
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 이건 (배치, 임베딩 차원) x (임베딩 차원, 히든) -> (배치, 히든)으로...
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        # 과대적합을 피하기 위해 여기에 드롭아웃 층 추가...
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        # 이건 (배치, 히든) x (히든, 1) -> (배치, 1)로
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
        # 과대적합을 피하기 위해 여기에 드롭아웃 층 추가...
        x = self.dropout(x)
        x = self.fc2(x)
        # 거쳐 나오면 (batch_size, 1)
        return self.sigmoid(x)
