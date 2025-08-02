import torch
from torch import nn
from device import device

class ModelCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features_map = nn.Sequential( # 1, 28, 28
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, device=device, bias=False), # 32, 28, 28
            nn.BatchNorm2d(num_features=32, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32, 14, 14

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, device=device, bias=False), # 64, 14, 14
            nn.BatchNorm2d(num_features=64, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 64, 7, 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128, device=device),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10, device=device)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.features_map(x)
        return self.classifier(x)