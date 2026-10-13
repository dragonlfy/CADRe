"""1D ResNet over the raw 12-lead ECG (4.3M parameters)."""

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride, k=9, drop=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(drop)
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        y = self.drop(F.relu(self.b1(self.c1(x))))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class ResNet1D(nn.Module):
    def __init__(self, n_out, widths=(64, 128, 192, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, widths[0], 15, stride=2, padding=7, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for w in widths:
            layers += [Block(cin, w, stride=2), Block(w, w, stride=1)]
            cin = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(2 * cin, n_out)

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))
