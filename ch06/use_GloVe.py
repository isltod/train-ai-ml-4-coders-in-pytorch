from bs4 import BeautifulSoup
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import string
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request
import zipfile

sys.path.append(".")
from utils import stopwords


def download_extract_glove():
    # GloVe 임베딩 다운로드
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    urllib.request.urlretrieve(url, "data/glove.6B.zip")

    # 압축 해제
    with zipfile.ZipFile("data/glove.6B.zip", "r") as zip_ref:
        zip_ref.extractall("data/glove/")


# 텍스트 처리 함수들...
def tokenize_glove_style(text):
    text = BeautifulSoup(text, "html.parser").get_text().lower()

    # 숫자는 0으로 바꾸는게 GloVe 스타일이라고...re.sub('패턴', '바꿀문자열', '대상문자열')
    text = re.sub(r"\d", "0", text)

    # 공백과 구둣점 나누기...GloVe는 구둣점도 토큰으로 관리...
    # .,!?() 찾아서 그 앞뒤로 공백...
    text = re.sub(r"([.,!?()])", r" \1 ", text)
    # space 2개 이상을 찾아서 공백 하나로...
    text = re.sub(r"\s{2,}", " ", text)

    return text.split()


def build_vocab_glove(sentences, max_vocab_size=10000):
    # 액셀 피봇처럼 토큰 별 빈도수 사전으로 만들고...
    counter = Counter()
    for text in sentences:
        counter.update(tokenize_glove_style(text))

    # 빈도 순으로 어휘 수 제한 - 2는 pad, unk
    most_common = counter.most_common(max_vocab_size - 2)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1

    return vocab


# 여러 (문장)들을 받아서 (단어 ID 리스트)의 리스트로 반환
def texts_to_sequences_glove(sentences, word_index):
    sequences = []
    for sentence in sentences:
        sequence = []
        for word in tokenize_glove_style(sentence):
            sequence.append(word_index.get(word, 1))
        sequences.append(sequence)
    return sequences


# 문장들에 패딩 넣기...
def pad_sequences_glove(sequences, max_len):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            padded_seq = seq[:max_len]
        else:
            padded_seq = seq + [0] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
    return padded_sequences


# 문장들 받아서 그 안에서 단어사전에 있는 단어들에 대한 단어별 빈도수 반환...
def word_frequency_glove(sentences, vocab=None):
    # 뭔가 이건 위에 build_vocab이랑 똑 같은 코든데...
    counter = Counter()
    for sentence in sentences:
        tokens = tokenize_glove_style(sentence)
        counter.update(tokens)

    # 어휘 사전이 주어지면, 그 안에 있는 단어들만 남긴다...
    if vocab:
        counter = Counter(
            {word: count for word, count in counter.items() if word in vocab}
        )

    # 빈도와 알파벳 순으로 정렬해서 반환 - x[1]이 사전의 빈도수고, -는 내림차순...
    sorted_words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return sorted_words


# maketrans(기존, 변경, 제거)? 근데 GloVe는 구둣점도 관리한다며...없애?
table = str.maketrans("", "", string.punctuation)

with open("data/sarcasm.json", "r") as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []
for item in datastore:
    sentence = item["headline"].lower()
    # 여기도 왜 구둣점을 다 지워?
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence, "html.parser")
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    sentences.append(filtered_sentence)
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])

vocab_size = 8000
max_length = 60
training_size = 20000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# GloVe 파일을 쓰지 않고, 원래 쓰던 sarcasm 파일로 사전을 만든다...근데 크기는 8K다...
word_index = build_vocab_glove(training_sentences, max_vocab_size=vocab_size)
training_sequences = texts_to_sequences_glove(training_sentences, word_index)
training_padded = pad_sequences_glove(training_sequences, max_len=max_length)
testing_sequences = texts_to_sequences_glove(testing_sentences, word_index)
testing_padded = pad_sequences_glove(testing_sequences, max_len=max_length)

# word_freq = word_frequency_glove(sentences, word_index)
# print(word_freq)


def load_pretrained_embeddings(vocab, embedding_dim=100):
    # GloVe 임베딩을 여기서 로드한다고...
    embeddings_dict = {}
    glove_file = f"data/glove/glove.6B.{embedding_dim}d.txt"
    print(f"Loading GloVe embeddings from {glove_file}...")
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            # 각 줄이 공백으로 구분해서 단어 숫자 숫자...형식이다...그래서 첫 번째는 단어 그 뒤는 벡터...
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_dict[word] = vector
    print("GloVe embeddings loaded.")

    # sarcasm 어휘 사전의 임베딩 행렬 초기화...
    embedding_matrix = np.random.uniform(-0.25, 0.25, size=(len(vocab), embedding_dim))
    # 0 행은 패딩...한 행에 넣으려면 열 차원만큼 0을 만든다...
    embedding_matrix[0] = np.zeros(embedding_dim)

    # sarcasm 임베딩 행렬을 미리 훈련된 임베딩 값으로 채운다...
    found_words = 0
    # sarcasm 파일에서 만든 어휘 사전에 있는 단어들을 돌면서
    for word, idx in vocab.items():
        # 그 단어가 GloVe 어휘 사전에 있으면...해당 ID에 GloVe 임베딩 벡터로 업데이트...
        if word in embeddings_dict:
            embedding_matrix[idx] = embeddings_dict[word]
            found_words += 1

    # 근데 원래 행을 sarcasm 어휘 사전 크기로 맞췄는데,
    # 그 단어들을 GloVe에서 다 찾지 못하면 못 찾은 단어들은 랜덤 임베딩 값으로 남는데...
    # 50d 경우 7939/8000를 찾았다고 하는데...
    print(f"Found embeddings for {found_words}/{len(vocab)} words")
    return torch.FloatTensor(embedding_matrix)


class TextClassificationModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=16,
        dropout_rate=0.25,
        pretrained_embeddings=None,
        freeze_embeddings=True,
    ):
        super(TextClassificationModel, self).__init__()

        # 임베딩 레이어 초기화하고, 미리 훈련된 임베딩 있으면 가져온다...
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            # in-place copy 이용해서 가중치 복사해 넣고...
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # 임베딩 고정 옵션이면 weight에 requires_grad = False로 잠금...
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        # 매개변수에는 dropout_rate가 있는데 드롭아웃 레이어는 없다?
        # self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(2)
        # 실제로 여기서도 적용하지 않는다?
        # x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def create_model(vocab, device="cuda", embedding_dim=100):
    # 위 함수 이용해서 미리 훈련된 임베딩 로딩하고 그걸 모델에 넣어 만든다...
    pretrained_embeddings = load_pretrained_embeddings(vocab, embedding_dim)
    model = TextClassificationModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=16,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=True,
    ).to(device)
    return model


model = create_model(
    vocab=word_index,
    device="cuda" if torch.cuda.is_available() else "cpu",
    embedding_dim=50,
)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), amsgrad=False)
print(model)

# 데이터 로더 준비 - 텐서 만들기 -> 데이터 셋 -> 데이터 로더
training_padded = torch.tensor(training_padded, dtype=torch.long)
testing_padded = torch.tensor(testing_padded, dtype=torch.long)
training_labels = torch.tensor(training_labels, dtype=torch.float32)
testing_labels = torch.tensor(testing_labels, dtype=torch.float32)
batch_size = 32
train_dataset = TensorDataset(training_padded, training_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(testing_padded, testing_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 훈련
num_epochs = 300
# 여기서 gpu 설정 하려면 위에서 create_model에서 뭐하러 to를 넣나?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_total += targets.size(0)
        train_correct += ((outputs.squeeze() > 0.5) == targets).sum().item()

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)

            val_loss += loss.item()
            val_total += targets.size(0)
            val_correct += ((outputs.squeeze() > 0.5) == targets).sum().item()

    print(f"Epoch {epoch+1}/{num_epochs}:")
    train_loss_history.append(train_loss / len(train_loader))
    train_acc_history.append(train_correct / train_total)
    val_loss_history.append(val_loss / len(test_loader))
    val_acc_history.append(val_correct / val_total)
    print(
        f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_correct/train_total:.4f}"
    )
    print(
        f"Val Loss: {val_loss/len(test_loader):.4f}, Val Acc: {val_correct/val_total:.4f}"
    )

# 모델 매개변수 저장
torch.save(model.state_dict(), "data/text_classification_model_glove.pth")


# 결과를 그려보자...
def plot_training_metrics(train_loss, train_acc, val_loss, val_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_loss) + 1)
    ax1.plot(epochs, train_loss, "b-", label="Training Loss")
    ax1.plot(epochs, val_loss, "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, train_acc, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_acc, "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    # 이건 뭐야??
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    plt.tight_layout()
    return fig


plot_training_metrics(
    train_loss_history, train_acc_history, val_loss_history, val_acc_history
)
plt.show()
