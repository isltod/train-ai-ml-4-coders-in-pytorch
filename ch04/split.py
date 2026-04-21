from torchvision import datasets, transforms
from torch.utils.data import random_split

# train 옵션을 지정하지 않으면 다 받는다? 아닌데?
dataset = datasets.FashionMNIST(
    root="./data", download=True, transform=transforms.ToTensor()
)
print(len(dataset))

total_count = len(dataset)
train_count = int(total_count * 0.7)
val_count = int(total_count * 0.15)
test_count = total_count - train_count - val_count

# 이름과는 다르게 무작위로 섞는게 아니고 자르는 지점만 무작위로 선택한다고...
for i in range(10):
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_count, val_count, test_count]
    )
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    # 이게 라벨 출력?
    print(train_dataset[0][1])
