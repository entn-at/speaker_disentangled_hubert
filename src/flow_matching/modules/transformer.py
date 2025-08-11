# Copied and modified from https://github.com/lucidrains/voicebox-pytorch/blob/main/voicebox_pytorch/voicebox_pytorch.py

# MIT License
#
# Copyright (c) 2023 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import PreTrainedModel

from .fastspeech import MLP
from .norm import AdaptiveRMSNorm


class RotaryEmbedding(nn.Module):
    """
    rotary positional embeddings
    https://arxiv.org/abs/2104.09864
    """

    def __init__(self, hidden_size: int, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @torch.autocast("cuda", enabled=False)
    def forward(self, t: Union[int, torch.Tensor]):
        if not torch.is_tensor(t):
            t = torch.arange(t, device=self.device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.autocast("cuda", enabled=False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.attn_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask: Optional[torch.BoolTensor] = None):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        q, k = map(lambda t: apply_rotary_pos_emb(position_embeddings, t), (q, k))

        bsz, heads, q_len, _ = q.shape

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L
        if attention_mask is not None and attention_mask.ndim != 4:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        if attention_mask is not None:
            attention_mask = attention_mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention
        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0
        )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = AdaptiveRMSNorm(config.hidden_size)
        self.post_attention_layernorm = AdaptiveRMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor],
        position_embeddings,
        rmsnorm_kwargs,
    ):
        attn_input = self.input_layernorm(hidden_states, **rmsnorm_kwargs)
        hidden_states = self.self_attn(attn_input, position_embeddings, attention_mask) + hidden_states

        ff_input = self.post_attention_layernorm(hidden_states, **rmsnorm_kwargs)
        hidden_states = self.mlp(ff_input, attention_mask) + hidden_states
        return hidden_states


class Transformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.rotary_emb = RotaryEmbedding(hidden_size=config.hidden_size // config.heads)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.depth)])
        self.norm = nn.RMSNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask: Optional[torch.BoolTensor] = None, adaptive_rmsnorm_cond=None):
        batch, seq_len, *_ = hidden_states.shape

        # rotary embeddings
        position_embeddings = self.rotary_emb(seq_len)

        # adaptive rmsnorm
        rmsnorm_kwargs = dict()
        if adaptive_rmsnorm_cond is not None:
            rmsnorm_kwargs = dict(condition=adaptive_rmsnorm_cond)

        # going through the attention layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_embeddings, rmsnorm_kwargs)

        return self.norm(hidden_states)
