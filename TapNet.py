# ================================================================
# TapNet: Temporal Attentive Projection Network for Time Series Classification
# Reference: Zhang et al., AAAI 2020
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random, time
from sklearn.model_selection import train_test_split

# ---------------------- 随机种子 ----------------------
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# ---------------------- 数据集定义 ----------------------
class VibrationDataset(Dataset):
    """读取 CSV 格式数据"""
    def __init__(self, x_file, y_file):
        self.x = pd.read_csv(x_file).values.astype(np.float32)
        self.y = pd.read_csv(y_file).values
        self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        self.labels = torch.LongTensor(self.y).squeeze()

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


# ---------------------- TapNet 模型 ----------------------
class TapNet(nn.Module):
    def __init__(self, input_size=1, num_classes=4,
                 cnn_channels=(64, 128), cnn_kernel=5,
                 gru_hidden=128, bidirectional=True,
                 proj_dim=128, attn_dropout=0.1, proto_scale=10.0):
        super().__init__()
        assert input_size == 1, "假设单通道输入。"

        C1, C2 = cnn_channels
        pad = cnn_kernel // 2

        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv1d(1, C1, kernel_size=cnn_kernel, padding=pad),
            nn.BatchNorm1d(C1), nn.ReLU(inplace=True),
            nn.Conv1d(C1, C2, kernel_size=cnn_kernel, padding=pad),
            nn.BatchNorm1d(C2), nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )

        # GRU Encoder
        self.gru = nn.GRU(input_size=C2, hidden_size=gru_hidden,
                          batch_first=True, bidirectional=bidirectional)
        enc_dim = gru_hidden * (2 if bidirectional else 1)

        # Temporal Attention Projection
        self.attn_score = nn.Linear(enc_dim, 1)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(enc_dim, proj_dim)

        # Prototype-based Classifier
        self.prototypes = nn.Parameter(torch.randn(num_classes, proj_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        self.proto_scale = proto_scale

    def forward(self, x):
        # 输入: [B, L] → [B, 1, L]
        x = x.unsqueeze(1)
        feat = self.cnn(x)               # [B, C2, L/2]
        feat = feat.transpose(1, 2)      # [B, T, C2]
        enc, _ = self.gru(feat)          # [B, T, enc_dim]

        # Temporal Attention
        score = self.attn_score(enc).squeeze(-1)           # [B, T]
        alpha = torch.softmax(score, dim=-1).unsqueeze(-1) # [B, T, 1]
        enc_att = (enc * alpha).sum(dim=1)                 # [B, enc_dim]
        enc_att = self.attn_drop(enc_att)

        # Projection
        z = self.proj(enc_att)                             # [B, proj_dim]

        # Cosine prototype classification
        z_norm = F.normalize(z, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        logits = self.proto_scale * (z_norm @ proto_norm.t())
        return logits


# ---------------------- 训练与验证函数 ----------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ---------------------- 主程序入口 ----------------------
def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径
    x_file = "./data/zaosheng.csv"
    y_file = "./data/label.csv"

    dataset = VibrationDataset(x_file, y_file)
    train_data, val_data = train_test_split(dataset, test_size=0.25, random_state=1, stratify=dataset.labels)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # 初始化模型
    model = TapNet(input_size=1, num_classes=4,
                   cnn_channels=(64, 128), cnn_kernel=5,
                   gru_hidden=128, bidirectional=True,
                   proj_dim=128, attn_dropout=0.1, proto_scale=10.0).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training TapNet...")
    for epoch in range(1, 601):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d}: Loss={train_loss:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    start = time.time()
    main()
    print("Time elapsed:", round(time.time() - start, 2), "seconds")
