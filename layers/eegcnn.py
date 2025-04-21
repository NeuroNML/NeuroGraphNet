import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnPool1d(nn.Module):
    """Attention Pooling for 1D data."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.q     = nn.Parameter(torch.randn(heads, dim))   # (H, C)
        self.kv    = nn.Linear(dim, dim*2, bias=False)

    def forward(self, x):               # x: (B, C, T)
        B, C, T = x.shape
        k, v = self.kv(x.permute(0,2,1)).chunk(2, dim=-1)   # (B, T, C)
        k = k.view(B, T, self.heads, C//self.heads).permute(2,0,1,3)  # H,B,T,C'
        v = v.view(B, T, self.heads, C//self.heads).permute(2,0,1,3)

        q = self.q[:, None, None] * self.scale              # H,1,1,C'
        attn = (q * k).sum(-1).softmax(-1)                  # H,B,T
        pooled = (attn[..., None] * v).sum(-2)              # H,B,C'
        return pooled.permute(1,0,2).reshape(B, C)

class EEGCNN(nn.Module):
    def __init__(self, in_ch=19, n_classes=1, d=0.3):
        super().__init__()
        def block(c_in, c_out, dilation):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, 3, padding=dilation, dilation=dilation),
                nn.GroupNorm(1, c_out),
                nn.ReLU(inplace=True),
            )

        self.stem   = block(in_ch, 32, 1)
        self.layer0 = block(32, 32, 4)
        self.down0  = nn.MaxPool1d(2)

        self.layer1 = block(32, 64, 2)
        self.layer2 = block(64, 64, 4)
        self.down1  = nn.MaxPool1d(2)

        self.layer3 = block(64, 128, 2)
        self.layer4 = block(128,128, 4)

        self.attn_pool = AttnPool1d(128, heads=4)
        self.drop      = nn.Dropout(d)
        self.fc        = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.permute(0,2,1)          # (B, C, T)
        x = self.stem(x)
        x = self.layer0(x);  x = self.down0(x)

        res = self.layer1(x)
        x   = self.layer2(res) + res
        x   = self.down1(x)

        res = self.layer3(x)
        x   = self.layer4(res) + res

        x = self.attn_pool(x)         # (B, 128)
        x = self.drop(x)
        return self.fc(x)
