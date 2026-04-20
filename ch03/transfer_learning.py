import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import RMSprop
from torchsummary import summary
import os
from PIL import Image


# 이건 늘 해야하는 GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 폴더...는 여기서도 쓰나?
training_dir = "horse-or-human/training/"
validation_dir = "horse-or-human/validation/"

# 이미 훈련된 Inception V3라는 모델을 로딩...
inception_v3 = models.inception_v3(pretrained=True, aux_logits=True)
inception_v3.to(device)


def print_model_summary(model):
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")


# print_model_summary(inception_v3)
# print(summary(inception_v3, (3, 299, 299)))

# Mixed_7c 층을 사용해 봐?
for name, paramter in inception_v3.named_parameters():
    # Mixed_7c 층 전까지를 다 동결한다...미분 계산 안하도록 하는 것이 동결인가...
    paramter.requires_grad = False
    if "Mixed_7c" in name:
        break

# 1024 뉴런 층과 2개 뉴런 층을 만들어 마지막 fc 층을 대체한다...
num_ftrs = inception_v3.fc.in_features
inception_v3.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    # 원래 모델이 1000개 내놓는 fc 였으니까, 소프트맥스 보다는 그냥 Linear가 비슷한가...
    nn.Linear(1024, 2),
)

# 늘 하던대로, 데이터 준비, 학습/평가 함수, 모델..은 만들었고, 돌리기...
# 우선 1 데이터 준비
transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = ImageFolder(training_dir, transform=transform)
val_dataset = ImageFolder(validation_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 2 학습/평가 함수
def train_model(model, device, criterion, optimizer, train_loader, num_epochs=10):
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        running_loss, running_corrects = 0.0, 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # 이 모델은 보조 출력이 있는 경우와 없는 경우가 있나?
            if isinstance(outputs, tuple):
                outputs, aux_output = outputs
                # 아무튼 보조 출력이 있는 경우는 이런 식으로 손실을 계산해야 하는 모양...
                loss = criterion(outputs, labels) + 0.4 * criterion(aux_output, labels)
            else:
                loss = criterion(outputs, labels)

            # 이건 정확도 계산을 위해서만 필요하다고...근데
            _, preds = torch.max(outputs, 1)

            # 역전파하고 최적화...
            loss.backward()
            optimizer.step()

            # 평균 손실과 정확도 계산을 하는 방법이...다 더하고 더한 수 / 총 갯수 가중치로...
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
        )


# 모델의 매개변수(p)를 받아서 requires_grad 속성을 설정?
optimizer = RMSprop(
    filter(lambda p: p.requires_grad, inception_v3.parameters()), lr=0.001
)
criterion = nn.CrossEntropyLoss()

# 모델 학습시키고...
train_model(inception_v3, device, criterion, optimizer, train_loader, num_epochs=10)

# 픽사베이 이미지 평가...
my_img_dir = "horse-or-human/my"
my_imgs = os.listdir(my_img_dir)


def predict(model, img_path, transform, device):
    model.eval()
    # 이미지 열어서, 아닌 경우 RGB로 변환
    image = Image.open(img_path).convert("RGB")
    # unsqueeze(0) - 0 차원에 배치 차원 추가...
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        _, predicted = torch.max(outputs, 1)
        label = "사람" if predicted.item() == 1 else "말"
        print(f"Predicted: {label} for image {os.path.basename(img_path)}")


for img in my_imgs:
    predict(inception_v3, os.path.join(my_img_dir, img), transform, device)
