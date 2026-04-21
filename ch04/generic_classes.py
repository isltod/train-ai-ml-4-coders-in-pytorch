from torchvision.datasets import ImageFolder, FakeData
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # ImageFolder 사용...원래는 알파벳 순서대로 라벨링이지만 아래처럼 바꿀 수 있다...
    custom_class_idx = {"rock": 0, "paper": 1, "scissors": 2}
    dataset = ImageFolder(
        "data/rps", target_transform=lambda x: custom_class_idx[dataset.classes[x]]
    )
    # 몇 개만 그림 라벨링 확인...
    # for i in range(3):
    #     index = torch.randint(0, len(dataset), (1,)).item()
    #     image, label = dataset[index]
    #     plt.imshow(image)
    #     plt.title(f"Label: {label}")
    #     plt.show()

    # 이건 안해도 이미 데이터 라벨은 제대로 나오는데...이건 뭐지?...아무튼 실제 데이터 라벨링과는 상관 없는듯...
    dataset.class_to_idx = custom_class_idx
    print(dataset.class_to_idx)

    # FakeData 사용...일단 이미지만 가능한 듯...
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    # 이게 가짜 이미지 만드는 부분인데...노이즈 화면 같은 랜덤 이미지가 나온다...
    fake_dataset = FakeData(
        size=100, image_size=(3, 224, 224), num_classes=10, transform=transform
    )
    data_loader = DataLoader(fake_dataset, batch_size=10, shuffle=True)
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        for i in range(len(inputs)):
            plt.imshow(inputs[i].permute(1, 2, 0))
            plt.title(f"Label: {labels[i]}")
            plt.show()
            if i == 2:
                break
        break
