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

# 设置随机种子
seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class VibrationDataset(Dataset):
    def __init__(self, x_file, y_file):
        self.x = pd.read_csv(x_file).values
        self.y = pd.read_csv(y_file).values
        self.inputs = self.x.astype(np.float32)
        self.labels = torch.LongTensor(self.y).squeeze()
        self.inputs = (self.inputs - np.mean(self.inputs)) / np.std(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

def to_grid_offsets(offsets, seq_len):
    device = offsets.device
    base = torch.linspace(-1, 1, steps=seq_len, device=device).view(1, 1, seq_len, 1)
    grid1 = (base + offsets * 2 / max(1, seq_len - 1)).clamp(-1, 1)
    grid2 = torch.zeros_like(grid1)
    grid = torch.cat([grid1, grid2], dim=-1)
    return grid.view(-1, seq_len, offsets.size(-1), 2)


class ShiftAwareAttention1D(nn.Module):
    def __init__(self, d_model, nhead, npoints=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.npoints = npoints
        self.value_proj = nn.Linear(d_model, d_model)
        self.offset_proj = nn.Linear(d_model, nhead * npoints)
        self.attn_weight_proj = nn.Linear(d_model, nhead * npoints)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        B, L, D = query.size()
        H, P = self.nhead, self.npoints
        v = self.value_proj(value).view(B, L, H, self.d_k).permute(0,2,3,1)
        offsets = self.offset_proj(query).view(B, L, H, P).permute(0,2,1,3)
        attn_w = F.softmax(self.attn_weight_proj(query).view(B, L, H, P).permute(0,2,1,3), dim=-1)
        grid = to_grid_offsets(offsets, L)
        v2 = v.reshape(B*H, self.d_k, L).unsqueeze(-1)
        sampled = F.grid_sample(v2, grid, mode='bilinear', align_corners=True)
        sampled = sampled.view(B, H, self.d_k, L, P)
        agg = (sampled * attn_w.unsqueeze(2)).sum(-1)
        agg = agg.permute(0,3,1,2).contiguous().view(B, L, D)
        return self.output_proj(agg)


class DynamicPool(nn.Module):
    def __init__(self, input_dim, out_len=4):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)  # 得分机制
        self.out_len = out_len

    def forward(self, x):
        # x: [B, C, T]
        score = self.query(x.permute(0,2,1)).squeeze(-1)  # [B, T]
        weights = torch.softmax(score, dim=-1).unsqueeze(1)  # [B, 1, T]
        pooled = torch.matmul(weights, x.transpose(1,2))  # [B, 1, C]
        pooled = pooled.expand(-1, self.out_len, -1).transpose(1,2)  # [B, C, out_len]
        return pooled


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, d_model=128, nhead=8, num_layers=2, out_len=4):
        super().__init__()
        self.fc_in = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                ShiftAwareAttention1D(d_model, nhead),
                nn.LayerNorm(d_model),
                nn.Sequential(nn.Linear(d_model, d_model*4), nn.ReLU(), nn.Linear(d_model*4, d_model)),
                nn.LayerNorm(d_model)
            ]) for _ in range(num_layers)
        ])
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model*2, 3, padding=1), nn.BatchNorm1d(d_model*2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(d_model*2, d_model*4, 3, padding=1), nn.BatchNorm1d(d_model*4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(d_model*4, d_model*8, 3, padding=1), nn.BatchNorm1d(d_model*8), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.dynamic_pool = DynamicPool(d_model*8, out_len)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model*8*out_len, 30), nn.ReLU(),
            nn.Linear(30, 10), nn.ReLU(),
            nn.Linear(10, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.fc_in(x)
        for attn, norm1, ff, norm2 in self.layers:
            res = attn(x, x, x); x = norm1(x+res)
            res2 = ff(x); x = norm2(x+res2)
        x = x.permute(0,2,1)
        x = self.cnn(x)
        x = self.dynamic_pool(x)
        return self.classifier(x)

def train(model, dataloader, optimizer, criterion, device):
    model.train(); running_loss=0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(); outputs=model(inputs)
        loss=criterion(outputs, labels); loss.backward(); optimizer.step()
        running_loss+=loss.item()
    return running_loss/len(dataloader)

def main():
    torch.cuda.set_device(0)
    dataset = VibrationDataset('.\\data\\zaosheng.csv', '.\\data\\label.csv')
    train_data, val_data = train_test_split(dataset, test_size=0.25, random_state=1, stratify=dataset.labels)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_size=1, num_classes=4, out_len=4).to(device)
    criterion = nn.CrossEntropyLoss(); optimizer = optim.Adam(model.parameters(), lr=4e-6)

    for epoch in range(1, 601):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        model.eval()
        train_acc = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item() for x,y in train_loader)/len(train_loader.dataset)
        val_acc = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item() for x,y in val_loader)/len(val_loader.dataset)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print('Time:', time.perf_counter() - start)



