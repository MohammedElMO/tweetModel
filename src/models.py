import math
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """alpha_i = softmax(v^T tanh(W h_i)),  c = sum_i alpha_i h_i"""

    def __init__(self, d, da=64):
        super().__init__()
        self.W = nn.Linear(d, da)
        self.v = nn.Linear(da, 1, bias=False)

    def forward(self, h, mask):
        scores = self.v(torch.tanh(self.W(h))).squeeze(-1)
        # never attend to padding
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(scores, dim=-1)
        return (alpha.unsqueeze(-1) * h).sum(1), alpha


class BiLSTM(nn.Module):
    def __init__(self, emb, hidden=128, attention=False):
        super().__init__()
        self.emb = emb
        self.lstm = nn.LSTM(emb.embedding_dim, hidden,
                            batch_first=True, bidirectional=True)
        self.att = AttentionPool(2 * hidden) if attention else None
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(2 * hidden, 2)

    def forward(self, x, mask):
        h, _ = self.lstm(self.drop(self.emb(x)))
        if self.att is not None:
            z, alpha = self.att(h, mask)
        else:                                              # max-pooling over real tokens
            z, alpha = h.masked_fill(
                mask.unsqueeze(-1) == 0, -1e9).max(1).values, None
        return self.out(self.drop(z)), alpha


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class MiniTransformer(nn.Module):
    def __init__(self, emb, d_model=128, heads=4, layers=2, use_pe=True):
        super().__init__()
        self.emb = emb
        self.proj = nn.Linear(emb.embedding_dim, d_model)
        self.pe = PositionalEncoding(d_model) if use_pe else None
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, 2)

    def forward(self, x, mask):
        h = self.proj(self.emb(x))
        if self.pe is not None:
            h = self.pe(h)
        h = self.encoder(h, src_key_padding_mask=(mask == 0))
        m = mask.unsqueeze(-1).float()
        return self.out((h * m).sum(1) / m.sum(1)), None
