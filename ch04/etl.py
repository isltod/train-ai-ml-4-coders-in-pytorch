# Extract -> Transform -> Load
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        # CPU 코어 수 또는 2배라는데, 이걸 늘리면 처음 진입, 루프 다시 시작에 오래 걸린다...그래서 밑에 persist
        num_workers=6,
        # GPU로 넘길 때 고정된 메모리 영역을 사용한다...
        pin_memory=True,
        # 에포크 루프 후에도 워커 프로세스를 종료하지 않는다...
        persistent_workers=True,
        # 미리 가져올 배치 수...
        prefetch_factor=4,
    )

    # 더미 모델
    model = torch.nn.Sequential(
        torch.nn.Linear(32 * 32 * 3, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 10),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Extract
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(dataloader):
            # 데이터 준비
            data = data.view(-1, 32 * 32 * 3).to(device)
            target = target.to(device)

            # 정방향, 손실
            output = model(data)
            loss = criterion(output, target)

            # 역전파, 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    "Epoch: {}, Batch: {}, Loss: {}".format(
                        epoch, batch_idx, loss.item()
                    )
                )
