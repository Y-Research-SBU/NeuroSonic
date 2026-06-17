import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from neurosonic.utils.model_util import RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final acoustic reconstruction layer of NeuroSonic.
    """
    def __init__(self, hidden_size, patch_dim):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class NeuroSonicBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class NeuroSonicTransformer(nn.Module):
    """
    Time-conditioned gated Transformer for EEG-to-speech reconstruction.
    """
    def __init__(
        self,
        audio_len=64000,
        audio_patch_len=256,
        eeg_channels=64,
        eeg_time=4000,
        eeg_frame_len=80,
        eeg_hop=80,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.audio_len = int(audio_len)
        self.audio_patch_len = int(audio_patch_len)
        self.eeg_channels = eeg_channels
        self.eeg_time = eeg_time
        self.eeg_frame_len = eeg_frame_len
        self.eeg_hop = eeg_hop
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # time embed
        self.t_embedder = TimestepEmbedder(hidden_size)

        # EEG encoder
        self.eeg_frames = self.eeg_time // self.eeg_hop
        self.eeg_mlp = nn.Sequential(
            nn.Linear(self.eeg_channels * self.eeg_frame_len, 1024, bias=True),
            nn.GELU(),
            nn.Linear(1024, hidden_size, bias=True)
        )
        self.eeg_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.eeg_pos = nn.Parameter(torch.zeros(1, self.eeg_frames, hidden_size))

        # Audio 1D patch embed
        if self.audio_patch_len <= 0:
            raise ValueError(f"audio_patch_len must be > 0, got {self.audio_patch_len}")
        self.audio_tokens = math.ceil(self.audio_len / self.audio_patch_len)
        self.audio_embed = nn.Linear(self.audio_patch_len, hidden_size, bias=True)
        self.audio_type = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.audio_pos = nn.Embedding(self.audio_tokens, hidden_size)
        self.register_buffer("audio_pos_idx", torch.arange(self.audio_tokens), persistent=False)

        # transformer
        self.blocks = nn.ModuleList([
            NeuroSonicBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                            proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, self.audio_patch_len)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.eeg_type, std=0.02)
        nn.init.normal_(self.eeg_pos, std=0.02)
        nn.init.normal_(self.audio_type, std=0.02)
        nn.init.normal_(self.audio_pos.weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _encode_eeg(self, eeg):
        """
        eeg: (B, 69, 4000)
        """
        if eeg.size(1) < self.eeg_channels:
            raise ValueError(f"EEG channels must be >= {self.eeg_channels}, got {eeg.size(1)}")

        if eeg.size(2) < self.eeg_time:
            pad_len = self.eeg_time - eeg.size(2)
            eeg = F.pad(eeg, (0, pad_len))
        elif eeg.size(2) > self.eeg_time:
            eeg = eeg[:, :, :self.eeg_time]

        eeg = eeg[:, :self.eeg_channels, :]
        total_len = self.eeg_frames * self.eeg_frame_len
        eeg = eeg[:, :, :total_len]
        eeg = eeg.view(eeg.size(0), self.eeg_channels, self.eeg_frames, self.eeg_frame_len)
        eeg = eeg.permute(0, 2, 1, 3).contiguous()
        eeg = eeg.view(eeg.size(0), self.eeg_frames, -1)
        eeg_tokens = self.eeg_mlp(eeg)
        eeg_tokens = eeg_tokens + self.eeg_type + self.eeg_pos
        return eeg_tokens

    def _encode_audio(self, audio):
        """
        audio: (B, L)
        """
        if audio.ndim != 2:
            raise ValueError(f"Audio must be 2D (B, L), got shape {tuple(audio.shape)}")
        if audio.size(1) < self.audio_len:
            pad_len = self.audio_len - audio.size(1)
            audio = F.pad(audio, (0, pad_len))
        elif audio.size(1) > self.audio_len:
            audio = audio[:, :self.audio_len]

        total_len = self.audio_tokens * self.audio_patch_len
        if audio.size(1) < total_len:
            audio = F.pad(audio, (0, total_len - audio.size(1)))
        elif audio.size(1) > total_len:
            audio = audio[:, :total_len]

        patches = audio.view(audio.size(0), self.audio_tokens, self.audio_patch_len)
        audio_tokens = self.audio_embed(patches)
        pos = self.audio_pos(self.audio_pos_idx)
        audio_tokens = audio_tokens + pos.unsqueeze(0) + self.audio_type
        return audio_tokens

    def _unpatchify_audio(self, x):
        """
        x: (B, N_audio, patch_dim)
        output: (B, L)
        """
        audio = x.reshape(x.size(0), -1)
        return audio[:, : self.audio_len]

    def forward(self, audio, eeg, t):
        """
        audio: (B, L)
        eeg: (B, 69, 4000)
        t: (N,)
        """
        t_emb = self.t_embedder(t)
        c = t_emb

        eeg_tokens = self._encode_eeg(eeg)
        audio_tokens = self._encode_audio(audio)
        x = torch.cat([eeg_tokens, audio_tokens], dim=1)

        for i, block in enumerate(self.blocks):
            x = block(x, c, feat_rope=None)

        x = x[:, self.eeg_frames:]
        x = self.final_layer(x, c)
        output = self._unpatchify_audio(x)
        return output


def neurosonic_base(**kwargs):
    return NeuroSonicTransformer(depth=12, hidden_size=768, num_heads=12, mlp_ratio=4.0, **kwargs)


def neurosonic_large(**kwargs):
    return NeuroSonicTransformer(depth=16, hidden_size=1024, num_heads=16, mlp_ratio=4.0, **kwargs)


def neurosonic_huge(**kwargs):
    return NeuroSonicTransformer(depth=32, hidden_size=1280, num_heads=16, mlp_ratio=4.0, **kwargs)


NEUROSONIC_MODELS = {
    'NeuroSonic-B': neurosonic_base,
    'NeuroSonic-L': neurosonic_large,
    'NeuroSonic-H': neurosonic_huge,
}
