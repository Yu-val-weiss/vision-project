from torch import nn

NUM_CLASSES = 7

class GestureModel(nn.Module):
    def __init__(self,dropout=0.2) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20*2, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, NUM_CLASSES),
        )
        
    def forward(self, x):
        return self.linear_relu_stack(x)