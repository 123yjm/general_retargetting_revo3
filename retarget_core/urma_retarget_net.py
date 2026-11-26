"""URMA-Hand Real-Time Retargeting Network (Section 2.2, 3.3, 3.4).

Static-dynamic separated architecture targeting < 1.5 ms pure PyTorch inference:
1. Startup/reset-only `encode_morphology(...)` -> cached `MorphologyCache`
2. Runtime `forward_step(...)`:
   - Joint Routing: attention-weighted pooling over dynamic joint obs `o_j = [q_{t-1}, qdot_{t-1}, e_{t-1}]` using `K_j_enc`
   - Tip Routing: attention-weighted pooling over `FK(q_{t-1})` using `K_f_enc`
   - Global Fusion + Core Policy MLP: `[Human 6 Tokens, g_palm, z_joint, z_tip, z_morph] -> z_act`
   - Universal Joint-Level Decoder: lightweight joint-to-human-token cross-attention + shared MLP `(z_act, E_j_dec, o_j, h_ctx_j) -> u_j -> [lo, hi]`
   - Optional critic value head `V(s)` and learnable action log-std for Stage 2 PPO fine-tuning
3. Online smoothing filters (`OneEuroFilter` & `EMAFilter`) for deployment/rendering.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .human_preprocess import HUMAN_NUM_TOKENS, HUMAN_TOKEN_DIM
from .diff_fk import JOINT_FB_DIM, TIP_FB_DIM
from .urdf_standardizer import JOINT_DESC_DIM, TIP_DESC_DIM

from .morph_gnn_encoder import MorphGNNEncoder, MorphologyCache


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SpatialJointSelfAttnLayer(nn.Module):
    """Full-joint spatial Multi-Head Self-Attention (N_max x N_max) with Pre-LN, joint_mask & structural bias.

    Enables explicit cross-finger joint coordination (e.g., thumb <-> index/middle/ring/pinky during precision
    pinch) with zero temporal memory (no Transformer-XL) and ~0.08 ms/layer GPU overhead.
    """

    def __init__(self, dim: int, num_heads: int = 4, ffn_mult: float = 1.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

        ffn_dim = int(round(dim * ffn_mult))
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,            # (B, N, D)
        joint_mask: torch.Tensor,   # (B, N) bool
        attn_bias: torch.Tensor,    # (B, H, N, N)
    ) -> torch.Tensor:
        B, N, D = x.shape
        H, Hd = self.num_heads, self.head_dim
        jm = joint_mask.unsqueeze(-1).to(x.dtype)

        h = self.norm1(x)
        qkv = self.qkv(h).view(B, N, 3, H, Hd)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # (B, H, N, Hd)
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + attn_bias  # (B, H, N, N)
        key_mask = (~joint_mask).view(B, 1, 1, N)
        scores = scores.masked_fill(key_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
        x = (x + self.proj(out)) * jm
        x = (x + self.ffn(self.norm2(x))) * jm
        return x


class URMARoutingPool(nn.Module):
    """URMA attention-weighted multi-slot routing from variable-size morphology tokens to a fixed-size latent."""

    def __init__(self, obs_dim: int, key_dim: int, num_slots: int = 8, slot_dim: int = 32, out_dim: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.logit_net = nn.Linear(key_dim, num_slots)
        self.val_net = nn.Sequential(
            nn.Linear(obs_dim + key_dim, slot_dim * 2),
            nn.SiLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(num_slots * slot_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, obs: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # obs: (B, N, obs_dim), keys: (B, N, key_dim), mask: (B, N) bool
        logits = self.logit_net(keys)                                  # (B, N, S)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(logits, dim=1)                         # (B, N, S)
        vals = self.val_net(torch.cat([obs, keys], dim=-1))            # (B, N, D)
        vals = vals * mask.unsqueeze(-1).to(vals.dtype)
        # (B, S, N) @ (B, N, D) -> (B, S, D)
        pooled = torch.bmm(weights.transpose(1, 2), vals)
        return self.out_proj(pooled.flatten(1))


class URMAHandNet(nn.Module):
    """Universal cross-embodiment dexterous hand retargeting network (URMA-Hand)."""

    def __init__(
        self,
        joint_desc_dim: int = JOINT_DESC_DIM,
        tip_desc_dim: int = TIP_DESC_DIM,
        human_token_dim: int = HUMAN_TOKEN_DIM,
        gnn_dim: int = 128,
        gnn_layers: int = 2,
        d_key: int = 64,
        d_dec: int = 128,
        d_morph: int = 64,
        d_tok: int = 64,
        d_route: int = 96,
        d_core: int = 256,
        core_blocks: int = 3,
        dec_hidden: int = 192,
        dec_attn_layers: int = 2,
        dec_attn_heads: int = 4,
        joint_fb_dim: int = JOINT_FB_DIM,
        tip_fb_dim: int = TIP_FB_DIM,
        residual_servo: bool = True,
        use_gnn: bool = True,
        use_dual_encoders: bool = True,
    ):
        super().__init__()
        self.d_tok = d_tok
        self.d_dec = d_dec
        self.dec_attn_layers = dec_attn_layers
        self.dec_attn_heads = dec_attn_heads
        self.joint_fb_dim = joint_fb_dim
        self.tip_fb_dim = tip_fb_dim
        self.residual_servo = residual_servo
        self.use_gnn = use_gnn
        self.use_dual_encoders = use_dual_encoders

        # 1. Startup/reset-only static morphology encoder
        self.morph_encoder = MorphGNNEncoder(
            joint_desc_dim=joint_desc_dim,
            tip_desc_dim=tip_desc_dim,
            gnn_dim=gnn_dim,
            gnn_layers=gnn_layers if use_gnn else 0,
            d_key=d_key,
            d_dec=d_dec,
            d_morph=d_morph,
            dec_attn_layers=dec_attn_layers,
            dec_attn_heads=dec_attn_heads,
        )
        if not use_dual_encoders:
            # Optional ablation flag (Section 5.4): share f_phi and g_omega projection
            self.shared_dec_proj = nn.Linear(d_key, d_dec)

        # 2. Runtime per-step encoders
        # Human 6-token encoder (Palm + 5 Fingers)
        self.human_tok_mlp = nn.Sequential(
            nn.Linear(human_token_dim, d_tok),
            nn.LayerNorm(d_tok),
            nn.SiLU(),
            nn.Linear(d_tok, d_tok),
            nn.LayerNorm(d_tok),
        )
        self.human_global_proj = nn.Sequential(
            nn.Linear(HUMAN_NUM_TOKENS * d_tok + 3, d_core),
            nn.LayerNorm(d_core),
            nn.SiLU(),
        )

        # Joint & Tip URMA routing modules (with analytical Jacobian + closed-loop error feedback)
        joint_obs_total = 3 + joint_fb_dim
        self.joint_router = URMARoutingPool(obs_dim=joint_obs_total, key_dim=d_key, num_slots=8, slot_dim=24, out_dim=d_route)
        self.tip_router = URMARoutingPool(obs_dim=tip_fb_dim, key_dim=d_key, num_slots=5, slot_dim=24, out_dim=d_route)

        # Core Policy MLP
        core_in = d_core + d_route + d_route + d_morph
        self.core_in = nn.Sequential(
            nn.Linear(core_in, d_core),
            nn.LayerNorm(d_core),
            nn.SiLU(),
        )
        self.core_blocks = nn.ModuleList([ResidualMLPBlock(d_core) for _ in range(core_blocks)])
        self.core_out_norm = nn.LayerNorm(d_core)

        # Lightweight joint-to-human-token cross-attention (N_max x 6 dot product; < 0.05 ms!)
        self.dec_q_proj = nn.Linear(d_dec, d_tok)
        self.dec_k_proj = nn.Linear(d_tok, d_tok)
        self.dec_v_proj = nn.Linear(d_tok, d_tok)
        self.attn_scale = 1.0 / math.sqrt(d_tok)

        # Universal Joint-Level Action Decoder:
        # (z_act, E_j_dec, o_j, joint_fb, h_ctx_j) -> 2-layer Full-Joint Spatial Self-Attention -> residual servo u_j
        self.decoder_in = nn.Sequential(
            nn.Linear(d_core + d_dec + joint_obs_total + d_tok, dec_hidden),
            nn.LayerNorm(dec_hidden),
            nn.SiLU(),
        )
        self.dec_coord = nn.ModuleList([
            SpatialJointSelfAttnLayer(dec_hidden, num_heads=dec_attn_heads)
            for _ in range(dec_attn_layers)
        ])
        self.dec_coord_out_norm = nn.LayerNorm(dec_hidden)
        self.decoder_head = nn.Sequential(
            nn.Linear(dec_hidden, dec_hidden // 2),
            nn.SiLU(),
            nn.Linear(dec_hidden // 2, 1),
        )

        # Stage 2 PPO Critic Value Head & Action Exploration Log-Std (zero overhead at inference)
        self.value_head = nn.Sequential(
            nn.Linear(d_core, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.action_log_std = nn.Parameter(torch.full((1,), -2.2))

    def encode_morphology(
        self,
        desc: torch.Tensor,
        tip_desc: torch.Tensor,
        adj: torch.Tensor,
        joint_mask: torch.Tensor,
        tip_mask: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
        scale: torch.Tensor,
        tree_dist: torch.Tensor | None = None,
        same_finger: torch.Tensor | None = None,
    ) -> MorphologyCache:
        """Startup / episode-reset static morphology encoding (call ONCE per hand/episode)."""
        cache = self.morph_encoder(
            desc, tip_desc, adj, joint_mask, tip_mask, lo, hi, scale,
            tree_dist=tree_dist, same_finger=same_finger,
        )
        if not self.use_dual_encoders:
            cache.E_joint_dec = self.shared_dec_proj(cache.K_joint_enc) * joint_mask.unsqueeze(-1).to(desc.dtype)
        return cache

    @staticmethod
    def normalize_joint_obs(
        q_prev: torch.Tensor,       # (B, N_max) raw rad/m
        qd_prev: torch.Tensor,      # (B, N_max) rad/s
        err_prev: torch.Tensor,     # (B, N_max) q_cmd_{t-1} - q_{t-1}
        lo: torch.Tensor,
        hi: torch.Tensor,
        joint_mask: torch.Tensor,
        vel_scale: float = 5.0,
    ) -> torch.Tensor:
        """Build normalized per-joint dynamic observation `o_j = [q_norm, qd_norm, err_norm]` `(B, N_max, 3)`."""
        span = (hi - lo).clamp_min(1e-4)
        q_norm = (2.0 * (q_prev - lo) / span - 1.0).clamp(-1.5, 1.5)
        qd_norm = (qd_prev / vel_scale).clamp(-3.0, 3.0)
        err_norm = (2.0 * err_prev / span).clamp(-2.0, 2.0)
        o_j = torch.stack([q_norm, qd_norm, err_norm], dim=-1)
        return o_j * joint_mask.unsqueeze(-1).to(o_j.dtype)

    def forward_core(
        self,
        cache: MorphologyCache,
        human_tokens: torch.Tensor,   # (B, 6, 26)
        g_palm: torch.Tensor,         # (B, 3) unit gravity in palm frame
        o_j: torch.Tensor,            # (B, N_max, 3 + joint_fb_dim) normalized joint obs + Jacobian fb
        tip_obs: torch.Tensor,        # (B, 5, tip_fb_dim) tip pos (+ optional error & pinch gap)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the fast dynamic step forward pass returning `(u_mean (B, N_max), q_cmd (B, N_max), z_act (B, d_core))`."""
        B = human_tokens.shape[0]
        if cache.K_joint_enc.shape[0] == 1 and B > 1:
            cache = cache.repeat_for_batch(B)

        # 1. Human 6 tokens + gravity vector
        h_tok = self.human_tok_mlp(human_tokens)                       # (B, 6, d_tok)
        z_hg = self.human_global_proj(torch.cat([h_tok.flatten(1), g_palm], dim=-1))

        # 2. URMA Joint Routing & Tip Routing
        z_joint = self.joint_router(o_j, cache.K_joint_enc, cache.joint_mask)
        z_tip = self.tip_router(tip_obs, cache.K_tip_enc, cache.tip_mask)

        # 3. Core Policy MLP
        z = self.core_in(torch.cat([z_hg, z_joint, z_tip, cache.z_morph], dim=-1))
        for blk in self.core_blocks:
            z = blk(z)
        z_act = self.core_out_norm(z)                                  # (B, d_core)

        # 4. Joint-to-Human cross-attention + Universal Action Decoder
        Q = self.dec_q_proj(cache.E_joint_dec)                         # (B, N_max, d_tok)
        K = self.dec_k_proj(h_tok)                                     # (B, 6, d_tok)
        V = self.dec_v_proj(h_tok)                                     # (B, 6, d_tok)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.attn_scale, dim=-1)
        h_ctx = torch.bmm(attn, V)                                     # (B, N_max, d_tok)

        N_max = cache.E_joint_dec.shape[1]
        z_exp = z_act.unsqueeze(1).expand(-1, N_max, -1)
        jm = cache.joint_mask.unsqueeze(-1).to(z_act.dtype)
        dec_h = self.decoder_in(torch.cat([z_exp, cache.E_joint_dec, o_j, h_ctx], dim=-1)) * jm
        # Full-joint spatial Multi-Head Self-Attention (N_max x N_max, explicit cross-finger coordination)
        for li, layer in enumerate(self.dec_coord):
            dec_h = layer(dec_h, cache.joint_mask, cache.dec_attn_bias[:, li])
        dec_h = self.dec_coord_out_norm(dec_h) * jm
        head_out = self.decoder_head(dec_h).squeeze(-1)                # (B, N_max)
        delta = torch.tanh(head_out)                                   # (-1, 1)
        if self.residual_servo:
            # Exact-identity bounded residual servo: u_raw = u_prev when delta=0; reaches [-1, 1] in 1 step
            u_prev = o_j[..., 0].clamp(-1.0, 1.0)
            u_raw = u_prev + torch.where(delta >= 0.0, (1.0 - u_prev) * delta, (1.0 + u_prev) * delta)
        else:
            u_raw = delta

        # Map strictly inside joint limits [lo, hi]
        q_cmd = cache.lo + 0.5 * (u_raw + 1.0) * (cache.hi - cache.lo)
        q_cmd = q_cmd * cache.joint_mask.to(q_cmd.dtype)
        return u_raw, q_cmd, z_act

    def forward_step(
        self,
        cache: MorphologyCache,
        human_tokens: torch.Tensor,
        g_palm: torch.Tensor,
        q_prev: torch.Tensor,
        qd_prev: torch.Tensor,
        err_prev: torch.Tensor,
        tip_prev: torch.Tensor,
        joint_fb: torch.Tensor | None = None,
        tip_fb: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """High-level real-time control step accepting physical-unit state + optional analytical Jacobian feedback."""
        B = human_tokens.shape[0]
        if cache.K_joint_enc.shape[0] == 1 and B > 1:
            cache = cache.repeat_for_batch(B)

        o_j = self.normalize_joint_obs(q_prev, qd_prev, err_prev, cache.lo, cache.hi, cache.joint_mask)
        if self.joint_fb_dim > 0:
            if joint_fb is None:
                joint_fb = torch.zeros(B, o_j.shape[1], self.joint_fb_dim, device=o_j.device, dtype=o_j.dtype)
            else:
                joint_fb = joint_fb * cache.joint_mask.unsqueeze(-1).to(o_j.dtype)
            o_j = torch.cat([o_j, joint_fb], dim=-1)

        if tip_fb is not None and tip_fb.shape[-1] == self.tip_fb_dim:
            tip_obs = tip_fb * cache.tip_mask.unsqueeze(-1).to(tip_fb.dtype)
        else:
            tip_prev_norm = (tip_prev / cache.scale[:, None, None].clamp_min(1e-4)).clamp(-4.0, 4.0)
            if self.tip_fb_dim > 3:
                pad = torch.zeros(B, tip_prev_norm.shape[1], self.tip_fb_dim - 3, device=tip_prev.device, dtype=tip_prev.dtype)
                tip_obs = torch.cat([tip_prev_norm, pad], dim=-1)
            else:
                tip_obs = tip_prev_norm
            tip_obs = tip_obs * cache.tip_mask.unsqueeze(-1).to(tip_obs.dtype)

        u_raw, q_cmd, z_act = self.forward_core(cache, human_tokens, g_palm, o_j, tip_obs)
        if not return_aux:
            return q_cmd
        value = self.value_head(z_act).squeeze(-1)
        return q_cmd, u_raw, value

    def map_normalized_action(self, u: torch.Tensor, cache: MorphologyCache) -> torch.Tensor:
        """Map normalized action `u` (clipped to `[-1, 1]`) strictly to `[lo, hi]`."""
        u_c = u.clamp(-1.0, 1.0)
        q = cache.lo + 0.5 * (u_c + 1.0) * (cache.hi - cache.lo)
        return q * cache.joint_mask.to(q.dtype)


# ----------------------------------------------------------------------------- Online smoothing filters (Section 2.2)
class EMAFilter:
    """Simple exponential moving average filter for joint targets."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = float(alpha)
        self.y: np.ndarray | None = None

    def reset(self):
        self.y = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.y is None:
            self.y = x.copy()
            return self.y.copy()
        self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y.copy()


class OneEuroFilter:
    """Casiez et al. One-Euro adaptive low-pass filter for low-latency, jitter-free retargeting output."""

    def __init__(self, rate_hz: float = 60.0, min_cutoff: float = 2.0, beta: float = 0.08, d_cutoff: float = 1.0):
        self.dt = 1.0 / max(float(rate_hz), 1e-3)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        dx = (x - self.x_prev) / self.dt
        a_d = self._alpha(self.d_cutoff, self.dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, self.dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat.astype(np.float32)
        self.dx_prev = dx_hat.astype(np.float32)
        return self.x_prev.copy()
