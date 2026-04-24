# embedding_sarcasm.py의 overfit을 수정...
# 1. lr을 1/10으로 줄이기...
# 2. 단어 사전 어휘 수 줄이기
import json
import matplotlib.pyplot as plt
from helper import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict


with open("data/sarcasm.json", "r") as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

table = str.maketrans("", "", string.punctuation)
for item in datastore:
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])

    sentence = item["headline"].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    # 책 소스에서는 따로 bs4, stopword 적용해서 하는데, 나는 책 helper에서 만들어 둔 함수 이용...
    soup = BeautifulSoup(sentence, "html.parser")
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence += word + " "
    sentences.append(sentence)

# 문장 길이(문장 내 단어 수) 테스트 - 이걸로 max_length 줄여서 과대적합 감소 시도...
# xs, ys = [], []
# curr_item = 1
# for sentence in sentences:
#     xs.append(curr_item)
#     curr_item += 1
#     ys.append(len(sentence))
# newys = sorted(ys)
# plt.plot(xs, newys)
# plt.show()

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
# print(len(word_index))

# 이 위와 아래를 반복해서 적당한 vocab_size를 정해 과대적합을 줄인다...
# word_frequency = word_frequency(training_sentences, word_index)
# for i in range(10):
#     print(f"{list(word_index.keys())[i]}: {word_frequency[list(word_index.keys())[i]]}")

# 단어 출현 빈도 그래프...
# newlist = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
# newdict = OrderedDict(newlist)

# xs, ys = [], []
# curr_x = 1
# for item in newdict:
#     xs.append(curr_x)
#     ys.append(newdict[item])
#     curr_x += 1

# print(ys)
# plt.axis([300, 10000, 0, 100])
# plt.plot(xs, ys)
# plt.show()

# 훈련, 테스트 데이터를 벡터, 패딩, 텐서로
training_sequences = texts_to_sequences(training_sentences, word_index)
training_padded = pad_sequences(training_sequences, max_length)
training_padded = torch.tensor(training_padded, dtype=torch.long)

testing_sequences = texts_to_sequences(testing_sentences, word_index)
testing_padded = pad_sequences(testing_sequences, max_length)
testing_padded = torch.tensor(testing_padded, dtype=torch.long)

# 데이터 로더에 넣어서 준비...
batch_size = 32
train_dataset = TensorDataset(training_padded, training_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(testing_padded, testing_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델, 손실함수, 최적화 함수 설정
# 패딩 추가해서 단어사전 크기...과대적합 줄이기 위해서 위에서 단어사전 크기 제한...
# vocab_size = len(word_index) + 1
# 임베딩 차원도 단어사전 수의 네 제곱근이 좋다라...
# embedding_dim = 100
embedding_dim = int(vocab_size**0.25) + 1
# 임베딩을 7 정도로 줄였으니, 그걸 받는 히든도 24는 좀 크다는...
# hidden_dim = 24
hidden_dim = 8

model = TextClassificationModel(vocab_size, embedding_dim, hidden_dim)
criterion = nn.BCELoss()
# 최적화 Adam의 lr, β 값 등은 기본값으로...하면 과대적합 심하고, lr을 1/10으로 줄여서 과대적합 감소...약간...
# optimizer = optim.Adam(model.parameters())
# optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), amsgrad=False)
# L2 규제를 적용해서 과대적합 감소 시도...
optimizer = optim.Adam(
    model.parameters(), lr=0.0001, betas=(0.9, 0.999), amsgrad=False, weight_decay=0.01
)
# optimizer = optim.Adam(
#     [
#         {"params": model.fc1.parameters(), "weight_decay": 0.01},
#         {"params": [p for name, p in model.named_parameters() if "fc1" not in name]},
#     ],
#     lr=0.0001,
# )
print(model)
summary(
    model,
    input_size=(batch_size, max_length),
    # 뭐지? 이거 때문에 되고 안되고가...임베딩 층에 타입을 지정한다는데...
    dtypes=[torch.long],
)

# 학습을 시켜보는데...
# 먼저 모델을 GPU에 넣고...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

# 100번 돌아가면서...300으로 소스가...
# num_epochs = 300
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
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
        # 마지막 fc와 sigmoid 거쳐 나올 때 (32, 1)로 나오는 걸 (32,)로...모양은 비슷해보이지만 행렬을 벡터로...
        train_correct += ((outputs.squeeze() > 0.5) == targets).sum().item()

    # validation
    model.eval()
    val_loss = 0.0
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

# 모델을 저장해본다...
torch.save(model.state_dict(), "sarcasm_model.pth")


def plot_training_metrics(
    train_loss_history, train_acc_history, val_loss_history, val_acc_history
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(train_loss_history) + 1)
    # loss
    ax1.plot(epochs, train_loss_history, label="Train Loss")
    ax1.plot(epochs, val_loss_history, label="Val Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training and Validation Loss")
    ax1.grid(True)
    # accuracy
    ax2.plot(epochs, train_acc_history, label="Train Acc")
    ax2.plot(epochs, val_acc_history, label="Val Acc")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Training and Validation Accuracy")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


plot_training_metrics(
    train_loss_history, train_acc_history, val_loss_history, val_acc_history
)
# 리얼텍, VMware, total commander virtual disk 플러그인...
