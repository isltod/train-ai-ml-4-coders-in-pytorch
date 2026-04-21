import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import RMSprop
from PIL import Image
import os
import sys

sys.path.append(".")
from utils import print_model_summary

# 이건 늘 해야하는 GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 다운받은 가위바위보 데이터 준비
training_dir = "data/rps"
transform = transforms.Compose(
    [
        # 인셉션 모델은 크기 299를 받는다...
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # 이건 계속 똑같은데...인셉션 모델 만들 때 설정인가?
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
# ImageFolder는 디렉터리의 알파벳 순서대로 레이블...p->r->s
train_dataset = ImageFolder(training_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 여기서도 Inception V3 사용...
inception_v3 = models.inception_v3(pretrained=True, aux_logits=True)
inception_v3.to(device)

print_model_summary(inception_v3)

# 모델 수정 - 먼저 Mixed_7c 이전까지 잠그고
for name, paramter in inception_v3.named_parameters():
    paramter.requires_grad = False
    if "Mixed_7c" in name:
        break
# 3개로 분류에 맞도록 마지막 층(fc)을 바꿔치기...
num_ftrs = inception_v3.fc.in_features
inception_v3.fc = nn.Sequential(
    # 드롭아웃을 추가하면 빨리 잘 훈련된다는데...잘은 모르겠고 빠르기는 하다..
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 3),
)

optimizer = RMSprop(
    filter(lambda p: p.requires_grad, inception_v3.parameters()), lr=0.001
)
criterion = nn.CrossEntropyLoss()

inception_v3.train()
inception_v3.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    running_loss, running_corrects = 0.0, 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = inception_v3(inputs)
        if isinstance(outputs, tuple):
            outputs, aux_output = outputs
            loss = criterion(outputs, labels) + 0.4 * criterion(aux_output, labels)
        else:
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        # labels.data가 책과 다른데...둘 다 되는 건지...
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / len(train_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# 테스트 이미지 평가...
inception_v3.eval()
for i in range(5):
    index = torch.randint(0, len(train_dataset), (1,)).item()
    img, label = train_dataset[index]
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = inception_v3(img)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted: {predicted.item()}, Actual: {label}")
