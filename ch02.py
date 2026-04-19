import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# 데이터 전처리
# ToTensor - [0,1] 사이로 변경해서 텐서로
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

# 데이터 로더
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=6, pin_memory=True
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 모델 정의
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # 2D 이미지를 1D로 펼치는 내장 함수
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 모델 인스턴스 생성
model = FashionMNISTModel()

# 손실 함수 및 옵티마이저 설정
# Negative Log Likelyhood Loss
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())


# 모델 학습 함수
def train(dataloader, model, loss_fn, optimizer):
    # GPU 사용 관련...
    model.to(device)
    loss_fn.to(device)

    size = len(dataloader.dataset)
    total_loss, total_accuracy = 0, 0
    # 구체적으로 뭔진 모르겠지만, 이게 최적화를 한다고...
    model.train()

    # 위에서 Dataloader 정의할 때 배치 크기는 64로...
    for batch, (X, y) in enumerate(dataloader):
        # GPU 사용 관련...
        X, y = X.to(device), y.to(device)

        # 1. 예측
        pred = model(X)
        # 2. 손실 계산 - 정확도 추가
        loss = loss_fn(pred, y)
        accuracy = get_accuracy(pred, y)

        # 3. 미분 역전파
        optimizer.zero_grad()
        loss.backward()
        # 4. 매개변수 갱신
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy.item()

        if batch % 100 == 0:
            current = batch * len(X)
            avg_loss = total_loss / (batch + 1)
            avg_accuracy = total_accuracy / (batch + 1) * 100
            # >7 - 7칸 오른쪽 정렬
            print(
                f"Batch {batch:>5d}, Loss: {avg_loss:>7f}, Accuracy: {avg_accuracy:>0.2f}%, [{current:>5d}/{size:>5d}]"
            )

    # early stopping...
    if avg_accuracy > 95:
        print("Reached 95% accuracy. Stopping training.")
        # 여기선 True 만 반환하고 끝낸다...나머진 main에서...근데 return Flase는 없어도 되나?
        return True


# 테스트 함수
def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # train 모드에서 eval 모드로 변경...이것도 최적화?
    model.eval()
    test_loss, correct = 0, 0
    # 역전파 미분 계산 끄기...
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # argmax(dim=1) - 열 방향에서 최대값 인덱스
            # X가 (배치,소프트맥스)이므로 sum 해서 배치 방향을 합계내야 한다...
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# 정확도 계산
def get_accuracy(pred, labels):
    _, preds = torch.max(pred, dim=1)
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy


# 모델 출력 살펴보기
def predict_single_image(image, label, model):
    # eval 모드로 전환하고
    model.eval()
    # 배치 차원 추가
    image = image.unsqueeze(0)
    # 미분 역전파 없이...
    with torch.no_grad():
        prediction = model(image)
        # 예측된 날 벡터...면 예측치가 10개 원소로...
        print(prediction)
        # 그 중에 열 방향 최대 인덱스가 라벨이 된다...
        predicted_label = prediction.argmax(1).item()
    # 이미지와 예측 결과 출력
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Predicted: {predicted_label}, Actual: {label}")
    plt.show()

    return predicted_label


if __name__ == "__main__":
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        if train(train_loader, model, loss_fn, optimizer):
            print("Early stopping triggered.")
            break
    print("Done!")

    test(test_loader, model)

    image, label = test_dataset[torch.randint(0, len(test_dataset), (1,)).item()]
    predicted_label = predict_single_image(image, label, model)
    print(f"The model predicted {predicted_label}, the actual label is {label}")
