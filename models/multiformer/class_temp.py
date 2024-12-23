import torch
import torch.nn as nn

class A(nn.Module):

    def __init__(self):
        super().__init__()
        self.linearA = nn.Linear(20, 2)

    def forward(self, x):
        return self.linearA(x)

class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = A()
        self.A_ = self.A
        self.A_.linearA = nn.Linear(10, 2)
        self.linearB = nn.Linear(2, 1)
        self.linearC = nn.Linear(1, 1)
        self.linearD = nn.Linear(1, 1)

    def forward(self, x):
        x = self.A_(x)
        x = self.linearB(x)
        x = self.linearC(x)
        x = self.linearD(x)
        return x

if __name__ == '__main__':
    x = torch.randn(1, 10)
    model = B()
    y = model(x)
    print(y)