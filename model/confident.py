import torch.nn as nn
import torch.nn.functional as F

class ConfidentNet(nn.Module):
    def __init__(self):
        super(ConfidentNet, self).__init__()
        self.uncertainty1_1 = nn.Conv2d(64, 400, 3, 1, 1)
        self.uncertainty2_1 = nn.Conv2d(400, 120, 3, 1, 1)
        self.uncertainty3_1 = nn.Conv2d(120, 64, 3, 1, 1)
        self.uncertainty4_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.uncertainty5_1 = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, u):
        u = F.relu(self.uncertainty1_1(u))
        u = F.relu(self.uncertainty2_1(u))
        u = F.relu(self.uncertainty3_1(u))
        u = F.relu(self.uncertainty4_1(u))
        u = self.uncertainty5_1(u)
        return u