from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # transforms는 그냥 함수인 모양...데이터에 적용해서 내보낸다...미리 적용해두는 것이 아니라...
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class LinearDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == "__main__":
    torch.manual_seed(0)
    # 이런 데이터를 예로 Customdataset을...
    x = torch.arange(0, 100, dtype=torch.float32)
    y = 2 * x - 1

    # 데이터셋 만들고, Dataloader에 걸기...
    dataset = LinearDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # enumerate 이용해서 반복해서 사용한다...
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        print(f"배치 인덱스: {batch_idx+1}")
        print(f"입력 데이터: {inputs}")
        print(f"레이블 데이터: {labels}")
        print("-" * 30)
