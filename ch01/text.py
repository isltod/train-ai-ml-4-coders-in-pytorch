import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print(torch.__version__)

# 모델 - Linear 안에는 (입력차원, 출력차원)
model = nn.Sequential(nn.Linear(1, 1))

# 손실함수
criterion = nn.MSELoss()

# 옵티마이저
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 데이터
xs = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
ys = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]], dtype=torch.float32)

# 학습 - 1. 추측 -> 2. 정확도 측정 -> 3. 추측 최적화
for epoch in range(500):
    # 옵티마이저 초기화...라지만, 결국 미분 값 누적을 막기 위한 초기화 아닐까...
    optimizer.zero_grad()
    # 1. 추측 - __call__에서 forward를 부르겠지?
    outputs = model(xs)
    # 2. 정확도 측정
    loss = criterion(outputs, ys)
    # 3. 추측 최적화 - 미분하고 그걸로 매개변수 갱신
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/500], Loss: {loss.item():.4f}")

# 예측
with torch.no_grad():
    print(model(torch.tensor([[10.0]], dtype=torch.float32)))

# model 내용을 보자...
layer = model[0]
print("weight", layer.weight.data.numpy())
print("bias", layer.bias.data.numpy())
