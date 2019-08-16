import torch
import torch.nn as nn
import torch.nn.functional as F
import math

HIDDEN_SIZE = 512
DROPOUT_R = 0.1
MULTI_HEAD = 8
FF_SIZE = 2048

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        
        self.linear_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.dropout = nn.Dropout(DROPOUT_R)

    def forward(self, q, k, v, mask):
        num_batches = q.size(0)

        q = self.linear_q(q).view(
            num_batches,
            MULTI_HEAD,
            -1,
            int(HIDDEN_SIZE / MULTI_HEAD)
        )
        k = self.linear_k(k).view(
            num_batches,
            MULTI_HEAD,
            -1,
            int(HIDDEN_SIZE / MULTI_HEAD)
        )
        v = self.linear_v(v).view(
            num_batches,
            MULTI_HEAD,
            -1,
            int(HIDDEN_SIZE / MULTI_HEAD)
        )

        attn = self.att(q, k, v, mask)
        attn = attn.transpose(1,2).contiguous().view(
            num_batches,
            -1,
            HIDDEN_SIZE
        )
        attn = self.linear_merge(attn)
        return attn

    def att(self, q, k, v, mask):
        dk = q.size(-1)

        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)

        if mask is not None:
            s = s.mask_fill(mask, -1e9)

        s = F.softmax(s, dim=-1)
        s = self.dropout(s)

        return torch.matmul(s, v)


# Feed-forwardNet
class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()

        self.mlp = MLP(
            in_size = HIDDEN_SIZE,
            mid_size = FF_SIZE,
            out_size = HIDDEN_SIZE,
            dropout_r = DROPOUT_R,
            act = nn.ReLU
        )

    def forward(self, x):
        return self.mlp(x)

# Self Attention
class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

        self.mhatt = MultiHeadAttention()

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.ffn = FeedForwardNet()

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(
            y + self.dropout1(self.mhatt(y, y, y, y_mask))
        )

        y = self.norm2(
            y + self.dropout2(self.ffn(y))
        )
        return y

# Full Connect
class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., act=None):
        super(FC, self).__init__()

        fc = []
        fc.append(nn.Linear(in_size, out_size))

        if act is not None:
            fc.append(act())
        
        if dropout_r > 0:
            fc.append(nn.Dropout(dropout_r))
        
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(x)


# Multiple Layer Perceptron
class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., act=None):
        super(MLP, self).__init__()

        self.fc1 = FC(in_size, mid_size, dropout_r=DROPOUT_R, act=act)
        self.fc2 = FC(mid_size, out_size)
    
    def forward(self, x):
        return self.fc2(self.fc1(x))

# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeors(size))
    
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = x.std(-1, keepdim=True)

        return self.a * (x-u) / (s + self.eps) + self.b
