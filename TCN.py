import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import random

# ==================== 随机种子与数据集 ====================
seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


class VibrationDataset(Dataset):
    def __init__(self, x_file, y_file):
        self.x = pd.read_csv(x_file).values.astype(np.float32)
        self.y = pd.read_csv(y_file).values
        self.inputs = (self.x - np.mean(self.x)) / np.std(self.x)
        self.labels = torch.LongTensor(self.y).squeeze()

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

# ==================== TCN 模块定义 ====================
class Chomp1d(nn.Module):
    """用于保持时序长度一致"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """TCN 基本模块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    """多层堆叠的 TCN 网络"""
    def __init__(self, input_size, num_classes, num_channels=[64, 128, 256, 512],
                 kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                     dilation=dilation, padding=(kernel_size-1)*dilation,
                                     dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 输入 [B, L] → [B, 1, L]
        x = x.unsqueeze(1)
        y = self.tcn(x)
        y = self.global_pool(y)
        y = self.classifier(y)
        return y

# ==================== 训练函数 ====================
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# ==================== 主程序 ====================
def main():
    torch.cuda.set_device(0)
    dataset = VibrationDataset('.\\data\\zaosheng.csv', '.\\data\\label.csv')
    train_data, val_data = train_test_split(dataset, test_size=0.25, random_state=1, stratify=dataset.labels)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNModel(input_size=1, num_classes=4, num_channels=[64,128,256,512], kernel_size=3, dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, 601):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        model.eval()
        train_acc = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item() for x,y in train_loader) / len(train_loader.dataset)
        val_acc = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item() for x,y in val_loader) / len(val_loader.dataset)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print('Time:', time.perf_counter() - start)
