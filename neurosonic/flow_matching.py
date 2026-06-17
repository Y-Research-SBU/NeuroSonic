#!/usr/bin/env python3
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurosonic.model import NEUROSONIC_MODELS


class NeuroSonicFlow(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.model not in NEUROSONIC_MODELS:
            raise ValueError(f"Unknown model: {args.model}. Available: {list(NEUROSONIC_MODELS.keys())}")

        self.model = NEUROSONIC_MODELS[args.model](
            audio_len=args.audio_len,
            audio_patch_len=args.audio_patch_len,
            eeg_channels=args.eeg_channels,
            eeg_time=args.eeg_time,
            eeg_frame_len=args.eeg_frame_len,
            eeg_hop=args.eeg_hop,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.noise_scale = args.noise_scale
        self.t_eps = args.t_eps
        self.cond_drop_prob = args.cond_drop_prob
        # how to sample time t in [0, 1):
        # - "uniform":   t ~ U([0, 1 - t_eps])
        # - "sigmoid":   t = sigmoid(N(P_mean, P_std)) clamped to [0, 1 - t_eps]
        # - "lognormal": sample sigma = exp(N(P_mean, P_std)) then map t = sigma / (1 + sigma),
        #               clamped to [0, 1 - t_eps]
        self.t_dist = getattr(args, "t_dist", "lognormal")

        # sampling config
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        # EMA decay (filled by the training entry point)
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2

    def forward(self, audio, eeg):
        """
        audio: (B, L)
        eeg: (B, 69, 4000)
        """
        bsz = audio.size(0)
        t = self.sample_t(bsz, device=audio.device)
        # Conditional flow matching linear interpolation:
        #   z_t = t * x + (1 - t) * eps,  t in [0, 1)
        # We keep an optional noise_scale: eps_scaled = noise_scale * eps
        eps = torch.randn_like(audio)
        eps_scaled = self.noise_scale * eps
        z_t = t.view(bsz, 1) * audio + (1.0 - t).view(bsz, 1) * eps_scaled

        eeg_cond = eeg
        if self.cond_drop_prob > 0.0:
            drop_mask = torch.rand(bsz, device=audio.device) < self.cond_drop_prob
            if drop_mask.any():
                eeg_cond = eeg_cond.clone()
                eeg_cond[drop_mask] = 0

        # model predicts clean x (x-prediction)
        x_pred = self.model(z_t, eeg_cond, t)
        # convert to velocity for v-loss:
        #   v_pred = (x_pred - z_t) / (1 - t)
        #   v_target = x - eps_scaled
        denom = (1.0 - t).clamp_min(self.t_eps).view(bsz, 1)
        v_pred = (x_pred - z_t) / denom
        v_target = audio - eps_scaled
        loss = F.l1_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def generate(
        self,
        eeg: torch.Tensor,
        x_init: torch.Tensor | None = None,
        t_start: float = 0.0,
    ) -> torch.Tensor:
        """
        eeg: (B, 69, 4000)
        x_init: optional starting point z(t_start) (B, L). If None, start from pure noise z(0) ~ N(0, I).
                (Kept name x_init for backward compatibility.)
        t_start: starting time in [0, 1). For unconditional generation, use t_start=0.
        Returns: generated 1D audio features (B, L)
        """
        device = eeg.device
        bsz = eeg.size(0)
        t_start = float(t_start)
        t_start = max(0.0, min(1.0 - self.t_eps, t_start))

        if x_init is None:
            # z_0 is pure noise (optionally scaled)
            x = torch.randn(bsz, self.model.audio_len, device=device) * self.noise_scale
            # Without a provided z(t_start), unconditional generation must start from t=0.
            t_start = 0.0
        else:
            if x_init.ndim != 2:
                raise ValueError(f"x_init must be (B, L), got shape {tuple(x_init.shape)}")
            x = x_init.to(device=device, dtype=torch.float32)
            if x.size(1) != self.model.audio_len:
                raise ValueError(f"x_init length must be {self.model.audio_len}, got {x.size(1)}")

        t_steps = torch.linspace(t_start, 1.0, self.steps, device=device)
        for i in range(len(t_steps) - 1):
            t = t_steps[i].expand(bsz)
            t_next = t_steps[i + 1].expand(bsz)
            m = self.method.lower()
            if m in ("euler", "edm-euler"):
                v = self._predict_v_cfg(x, eeg, t)
                x = x + (t_next - t).view(bsz, 1) * v
            elif m in ("heun", "edm-heun"):
                dt = (t_next - t).view(bsz, 1)
                v_t = self._predict_v_cfg(x, eeg, t)
                x_euler = x + dt * v_t
                v_t_next = self._predict_v_cfg(x_euler, eeg, t_next)
                v = 0.5 * (v_t + v_t_next)
                x = x + dt * v
            else:
                raise ValueError(f"Unknown sampling_method: {self.method}. Use 'euler' or 'heun'.")

        t_final = t_steps[-1].expand(bsz)
        # return predicted clean audio x_theta(z_t, t) at final time
        return self._predict_x_cfg(x, eeg, t_final)

    def sample_t(self, batch_size: int, device) -> torch.Tensor:
        # Sample t in [0, 1 - t_eps] to avoid division by ~0 in (1 - t).
        t_max = 1.0 - float(self.t_eps)
        if self.t_dist == "uniform":
            t = torch.rand(batch_size, device=device) * t_max
        else:
            s = torch.randn(batch_size, device=device) * self.P_std + self.P_mean
            if self.t_dist == "lognormal":
                sigma = torch.exp(s)
                t = sigma / (1.0 + sigma)
            elif self.t_dist == "sigmoid":
                t = torch.sigmoid(s)
            else:
                raise ValueError(f"Unknown t_dist: {self.t_dist}")
        return t.clamp(0.0, t_max)

    def _predict_x_cfg(self, z, eeg, t):
        # returns x_theta(z, t) with optional CFG
        if self.cfg_scale == 1.0:
            return self.model(z, eeg, t)

        t_val = float(t.mean().item())
        if not (self.cfg_interval[0] <= t_val <= self.cfg_interval[1]):
            return self.model(z, eeg, t)

        eeg_uncond = torch.zeros_like(eeg)
        x_uncond = self.model(z, eeg_uncond, t)
        x_cond = self.model(z, eeg, t)
        return x_uncond + self.cfg_scale * (x_cond - x_uncond)

    def _predict_v_cfg(self, z, eeg, t):
        # v_theta(z, t) = (x_theta(z, t) - z) / (1 - t)
        bsz = z.size(0)
        x_pred = self._predict_x_cfg(z, eeg, t)
        denom = (1.0 - t).clamp_min(self.t_eps).view(bsz, 1)
        v_pred = (x_pred - z) / denom
        return v_pred

    def update_ema(self):
        ema1 = self.ema_params1
        ema2 = self.ema_params2
        decay1 = self.ema_decay1
        decay2 = self.ema_decay2
        with torch.no_grad():
            for i, p in enumerate(self.parameters()):
                ema1[i].mul_(decay1).add_(p.data, alpha=1 - decay1)
                ema2[i].mul_(decay2).add_(p.data, alpha=1 - decay2)
