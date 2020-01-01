import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, voc_num, voc_dim):
        super(CBOW, self).__init__()

        self.codebook = nn.Embedding(voc_num, voc_dim)# 定义一个随机参数向量
        self.linear1 = nn.Linear(voc_dim, voc_dim, bias=False)
        self.linear2 = nn.Linear(voc_dim, voc_dim, bias=False)
        self.linear3 = nn.Linear(voc_dim, voc_dim, bias=False)
        self.linear4 = nn.Linear(voc_dim, voc_dim, bias=False)


    def forward(self, x1, x2, x4, x5):
        v1 = self.codebook(x1)
        v2 = self.codebook(x2)
        v4 = self.codebook(x4)
        v5 = self.codebook(x5)

        y1 = self.linear1(v1)
        y2 = self.linear2(v2)
        y4 = self.linear3(v4)
        y5 = self.linear4(v5)

        return y1 + y2 + y4 + y5

    def getloss(self, x3, y3):
        v3 = self.codebook(x3)

        return torch.mean((y3 - v3) ** 2)