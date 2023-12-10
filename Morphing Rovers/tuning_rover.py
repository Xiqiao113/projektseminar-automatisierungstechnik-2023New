# Xiqiao Zhang
# 2023/12/10
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

mode_views = torch.load('mode_view.t')
# 读取形状信息，作为模型的初始化参数
chromosome = np.load('test.npy')
old_modes = chromosome[: 4*11**2]
old_modes = torch.Tensor(np.reshape(old_modes, (4,11,11)))

EPS_C = (0.03)**2
batch_size = 128

from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, mode_views):
        self.mode_views = mode_views

    def __len__(self):
        return len(self.mode_views)

    def __getitem__(self, idx):
        return mode_views[idx]

dataset = CustomDataset(mode_views)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class TuningMode(nn.Module):
    def __init__(self, modes):
        super().__init__()
        # self.modes.shape: (4,11,11)
        self.modes = nn.parameter.Parameter(modes)

    def forward(self, inputs):
        # inputs.shape: (batch, 11, 11)
        f_norm = torch.norm(inputs, dim=(1,2)) # batch
        mv_norm = torch.norm(self.modes, dim=(1,2)) # 4
        # luminance = how well do the vector norms agree?
        # 0 = no match, 1 = perfect match
        luminance = (2*f_norm.unsqueeze(1)*mv_norm+EPS_C)/(f_norm.unsqueeze(1)**2+mv_norm**2+EPS_C)
        # distance = Euclidean distance on unit sphere (i.e., of normalized vectors)
        # Rescaled to go from 0 to 1, with 0 = no match, 1 = perfect match
        # （batch * 1 * 11 * 11）  （4 * 11 * 11）  -> (batch * 4 * 11 * 11)
        distance = (2-((inputs/f_norm.reshape(-1,1,1)).unsqueeze(1) - self.modes/mv_norm.reshape(-1,1,1)).norm(dim=(2,3)))*0.5
        # final metric = product of distance and luminance
        metric = distance*luminance.sqrt()
        # turn metric into range 0 to 2
        # 0 = perfect match, 2 = worst match
        metric = (1-metric)*2
        return torch.min(metric, dim=1).values # shape: (batch, 1)

tuning_mode = TuningMode(old_modes)

lr = 1e-2
optimizer = torch.optim.Adam(tuning_mode.parameters(), lr=lr)

loss_fn = nn.MSELoss()

epochs = 10

for e_i in range(epochs):
    tuning_mode.train()
    for batch, x in enumerate(dataloader):
        y_true = torch.zeros(x.shape[0])
        outputs = tuning_mode(x)
        loss = loss_fn(outputs, y_true)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            print(f"epoch: {e_i}, batch: {batch}, loss: {loss:>7f}")

v2_modes = tuning_mode.modes.detach().numpy()

chromosome[: 4*11**2] = v2_modes.reshape(-1)

np.save('v2_test.npy', chromosome)