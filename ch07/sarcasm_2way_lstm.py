import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(".")
from utils import *

# 데이터 셑 준비...
sentences, labels, urls = load_sarcasm_dataset()

vocab_size = 2000
embedding_dim = 7
max_length = 85
training_size = 23000
batch_size = 32

train_loader, test_loader, vocab = load_datasets_sarcasm(
    sentences, labels, vocab_size, max_length, training_size, batch_size
)


# 여기서 사용할 모델 정의하고 만들기...
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=24, lstm_layers=1):
        super(TextClassificationModel, self).__init__()
        # 임베딩 층
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM 층...근데 이게 뭘로 양방향인지 알지?
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # 이건 여전히 모르겠는 풀
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 선형 변환 층
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # 활성화 층
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 입력 x는 (배치, 문장 내 단어 수)...여기에 단어마다 임베딩 시키면...
        embedded = self.embedding(x)
        # embedded는 (배치, 문장 내 단어 수, 임베딩 차원)
        lstm_out, _ = self.lstm(embedded)
        # lstm_out은 임베딩을 은닉으로 바꿔서...(배치, 문장 내 단어 수, 은닉 차원)
        lstm_out = lstm_out.transpose(1, 2)
        # global_pool을 위해서 (배치, 은닉, 단어) 순서로 축 변경...여기서 단어 아이디가 풀링 되고...
        pooled = self.global_pool(lstm_out).squeeze(-1)
        # (배치, 은닉) 나오고

        # 마지막은 linear->relu->linear->sigmoid...
        x = self.relu(self.fc1(pooled))
        x = self.sigmoid(self.fc2(x))
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassificationModel(vocab_size, embedding_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), amsgrad=False)

# 모델 훈련
num_epochs = 300
trained_model = train_sarcasm_model(
    model, device, train_loader, test_loader, criterion, optimizer, num_epochs
)

# 모델 매개변수 저장
torch.save(trained_model.state_dict(), "data/lstm_2way_sarcasm_classification.pth")

# 간단 테스트
test_sentence_sarcasm(trained_model, sentences, vocab, max_length, device, 0.5)

# 텐서 프로젝터 용 파일도 만들어 보자...
dir = "data"
name = "lstm_2way_sarcasm_classification"
embedding_weights = trained_model.embedding.weight.data.cpu().numpy()
write_tensor_projector(dir, name, vocab, embedding_weights)
