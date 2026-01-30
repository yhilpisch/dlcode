"""
Deep Learning with PyTorch
(c) Dr. Yves J. Hilpisch
AI-Powered by GPT-5.x.

Chapter 14 — Attention walkthrough (single and multi-head, with masks).

This script mirrors the step-by-step REPL in the chapter so you can run and
inspect shapes and values locally.
"""
from __future__ import annotations
import math
import torch


def scaled_dot_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = Q.size(-1)
    S = (Q @ K.transpose(-2, -1)) / math.sqrt(d)
    if mask is not None:
        S = S.masked_fill(~mask, float('-inf'))
    A = torch.softmax(S, dim=-1)
    return A @ V, A


def demo_single_head(T: int = 4, d: int = 3) -> None:
    torch.manual_seed(0)
    Q = torch.randn(T, d)
    K = torch.randn(T, d)
    V = torch.randn(T, d)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    O_c, A_c = scaled_dot_attention(Q, K, V, mask=causal)
    O_u, A_u = scaled_dot_attention(Q, K, V, mask=None)
    print('A_c (causal):\n', A_c)
    print('A_u (unmasked):\n', A_u)
    print('O_c shape:', O_c.shape)


def demo_multi_head(
    B: int = 2, T: int = 5, d_model: int = 8, h: int = 2
) -> None:
    torch.manual_seed(0)
    d_head = d_model // h
    x = torch.randn(B, T, d_model)
    Wq = torch.randn(d_model, d_model)
    Wk = torch.randn(d_model, d_model)
    Wv = torch.randn(d_model, d_model)
    Wo = torch.randn(d_model, d_model)
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    Q = Q.view(B, T, h, d_head).transpose(1, 2)  # (B, h, T, d_head)
    K = K.view(B, T, h, d_head).transpose(1, 2)
    V = V.view(B, T, h, d_head).transpose(1, 2)
    S = (Q @ K.transpose(-2, -1)) / math.sqrt(d_head)  # (B, h, T, T)
    A = torch.softmax(S, dim=-1)
    O = A @ V  # (B, h, T, d_head)
    O = O.transpose(1, 2).contiguous().view(B, T, d_model)
    y = O @ Wo  # final projection
    print('y shape:', y.shape)


if __name__ == '__main__':
    demo_single_head()
    demo_multi_head()
