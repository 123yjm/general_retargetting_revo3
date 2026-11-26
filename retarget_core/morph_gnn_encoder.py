"""URMA-Hand Static Morphology GNN & Dual Independent Description Encoders (Section 2.2 & 3.2).

Designed to run ONCE at startup/episode reset per hand embodiment and produce a reusable `MorphologyCache`:
- 2-layer lightweight kinematic graph GNN over actuated joints (`adj` + normalized 58-D descriptors `d_j`)
- Strictly independent Encoder description network `f_phi` -> `K_j_enc`, `K_f_enc`
- Strictly independent Decoder description network `g_omega` -> `E_j_dec`
- Compact pooled morphology summary `z_morph`
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .urdf_standardizer import JOINT_DESC_DIM, TIP_DESC_DIM


@dataclass
class MorphologyCache:
    """Static morphology embeddings computed once per hand embodiment / episode reset."""

    K_joint_enc: torch.Tensor   # (B, N_max, d_key) encoder-side joint routing keys
    K_tip_enc: torch.Tensor     # (B, 5, d_key)     encoder-side fingertip routing keys
    E_joint_dec: torch.Tensor   # (B, N_max, d_dec) decoder-side joint action embeddings
    z_morph: torch.Tensor       # (B, d_morph)      global hand morphology summary
    joint_mask: torch.Tensor    # (B, N_max) bool   True for valid actuated joints
    tip_mask: torch.Tensor      # (B, 5) bool       True for present fingers
    lo: torch.Tensor            # (B, N_max)        lower joint limits
    hi: torch.Tensor            # (B, N_max)        upper joint limits
    scale: torch.Tensor         # (B,)              L_palm characteristic scale (m)
    adj: torch.Tensor           # (B, N_max, N_max) normalized kinematic graph adjacency
    dec_attn_bias: torch.Tensor # (B, L_dec, H_dec, N_max, N_max) cached pairwise structural bias

    def repeat_for_batch(self, batch_size: int) -> "MorphologyCache":
        """Broadcast a single-hand cache (B=1) to `batch_size` without recomputing GNN/encoders."""
        if self.K_joint_enc.shape[0] == batch_size:
            return self
        assert self.K_joint_enc.shape[0] == 1, "Can only broadcast B=1 MorphologyCache"
        return MorphologyCache(
            K_joint_enc=self.K_joint_enc.expand(batch_size, -1, -1),
            K_tip_enc=self.K_tip_enc.expand(batch_size, -1, -1),
            E_joint_dec=self.E_joint_dec.expand(batch_size, -1, -1),
            z_morph=self.z_morph.expand(batch_size, -1),
            joint_mask=self.joint_mask.expand(batch_size, -1),
            tip_mask=self.tip_mask.expand(batch_size, -1),
            lo=self.lo.expand(batch_size, -1),
            hi=self.hi.expand(batch_size, -1),
            scale=self.scale.expand(batch_size),
            adj=self.adj.expand(batch_size, -1, -1),
            dec_attn_bias=self.dec_attn_bias.expand(batch_size, -1, -1, -1, -1),
        )


class KinematicGNNLayer(nn.Module):
    """Lightweight residual message-passing layer over the hand's actuated joint kinematic graph."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # adj: (B, N, N) symmetric normalized adjacency; mask: (B, N) bool
        m = mask.unsqueeze(-1).to(h.dtype)
        h_masked = h * m
        neigh = torch.bmm(adj, h_masked)
        out = self.norm(h + F.silu(self.lin_self(h_masked) + self.lin_neigh(neigh)))
        return out * m


class MorphGNNEncoder(nn.Module):
    """Startup/reset-only static morphology encoder with strictly independent f_phi and g_omega branches."""

    MAX_TREE_DIST = 8

    def __init__(
        self,
        joint_desc_dim: int = JOINT_DESC_DIM,
        tip_desc_dim: int = TIP_DESC_DIM,
        gnn_dim: int = 128,
        gnn_layers: int = 2,
        d_key: int = 64,
        d_dec: int = 128,
        d_morph: int = 64,
        dec_attn_layers: int = 2,
        dec_attn_heads: int = 4,
    ):
        super().__init__()
        self.gnn_dim = gnn_dim
        self.d_key = d_key
        self.d_dec = d_dec
        self.d_morph = d_morph
        self.dec_attn_layers = dec_attn_layers
        self.dec_attn_heads = dec_attn_heads

        # Joint input projection + lightweight GNN (1-2 layers)
        self.joint_in = nn.Sequential(
            nn.Linear(joint_desc_dim, gnn_dim),
            nn.LayerNorm(gnn_dim),
            nn.SiLU(),
        )
        self.gnn_layers = nn.ModuleList([KinematicGNNLayer(gnn_dim) for _ in range(gnn_layers)])

        # Strictly independent encoder-side description network f_phi (joint & fingertip routing keys)
        self.f_phi_joint = nn.Sequential(
            nn.Linear(gnn_dim + joint_desc_dim, gnn_dim),
            nn.LayerNorm(gnn_dim),
            nn.SiLU(),
            nn.Linear(gnn_dim, d_key),
            nn.LayerNorm(d_key),
        )
        self.f_phi_tip = nn.Sequential(
            nn.Linear(tip_desc_dim, gnn_dim),
            nn.LayerNorm(gnn_dim),
            nn.SiLU(),
            nn.Linear(gnn_dim, d_key),
            nn.LayerNorm(d_key),
        )

        # Strictly independent decoder-side description network g_omega (joint action embeddings)
        self.g_omega_joint = nn.Sequential(
            nn.Linear(gnn_dim + joint_desc_dim, d_dec),
            nn.LayerNorm(d_dec),
            nn.SiLU(),
            nn.Linear(d_dec, d_dec),
            nn.LayerNorm(d_dec),
        )

        # Static global morphology summary z_morph (attention-pooled over joints + fingertips)
        self.morph_j_score = nn.Linear(gnn_dim, 1)
        self.morph_proj = nn.Sequential(
            nn.Linear(gnn_dim + d_key + 2, d_morph),
            nn.LayerNorm(d_morph),
            nn.SiLU(),
        )

        # Static pairwise structural bias for decoder full-joint spatial Self-Attention (Section 2 v6):
        # Combines topological tree distance, same-finger flag, thumb-cross-finger opposition flag,
        # and canonical 3D relative position (dx, dy, dz, ||dp||) — computed ONCE per hand!
        n_bias_out = max(dec_attn_layers, 1) * dec_attn_heads
        self.tree_dist_bias = nn.Embedding(self.MAX_TREE_DIST + 1, n_bias_out)
        self.pair_geom_bias = nn.Sequential(
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, n_bias_out),
        )

    @staticmethod
    def _infer_tree_dist_from_adj(adj: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
        """Fallback shortest-hop topological distance (0..MAX_TREE_DIST) when tree_dist is not explicitly passed."""
        B, N, _ = adj.shape
        device = adj.device
        # Treat adj > 0 (off-diagonal) as 1-hop graph edges; compute bounded Floyd-Warshall / powers up to 6
        edge = (adj > 1e-6) & joint_mask.unsqueeze(1) & joint_mask.unsqueeze(2)
        dist = torch.full((B, N, N), MorphGNNEncoder.MAX_TREE_DIST, dtype=torch.long, device=device)
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        dist = dist.masked_fill(eye, 0)
        reach = eye.expand(B, -1, -1)
        for hop in range(1, MorphGNNEncoder.MAX_TREE_DIST):
            nxt = (torch.bmm(reach.float(), edge.float()) > 0.5) & (~reach)
            dist = torch.where(nxt, torch.full_like(dist, hop), dist)
            reach = reach | nxt
        return dist

    def _compute_dec_attn_bias(
        self,
        desc: torch.Tensor,
        adj: torch.Tensor,
        joint_mask: torch.Tensor,
        tree_dist: torch.Tensor | None = None,
        same_finger: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute static `(B, L_dec, H_dec, N, N)` structural attention bias once at startup/reset."""
        B, N, _ = desc.shape
        if tree_dist is None:
            td = self._infer_tree_dist_from_adj(adj, joint_mask)
        else:
            td = tree_dist.long().clamp(0, self.MAX_TREE_DIST)

        # Finger labels from desc[:, :, 15:21] (one-hot: 0=thumb, 1..4=index..little, 5=other)
        f_oh = desc[:, :, 15:21]
        f_idx = f_oh.argmax(dim=-1)  # (B, N)
        valid_f = f_idx < 5
        if same_finger is None:
            sf = (f_idx.unsqueeze(2) == f_idx.unsqueeze(1)) & valid_f.unsqueeze(2) & valid_f.unsqueeze(1)
        else:
            sf = same_finger.bool()
        is_thumb = (f_idx == 0) & joint_mask
        is_nonthumb = (f_idx >= 1) & (f_idx < 5) & joint_mask
        thumb_cross = (is_thumb.unsqueeze(2) & is_nonthumb.unsqueeze(1)) | (
            is_nonthumb.unsqueeze(2) & is_thumb.unsqueeze(1)
        )

        # Canonical 3D relative joint position (from desc[:, :, 0:3])
        pos = desc[:, :, 0:3]  # (B, N, 3)
        rel_p = pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 3): p_j - p_i
        rel_d = rel_p.norm(dim=-1, keepdim=True)     # (B, N, N, 1)
        pair_feat = torch.cat(
            [rel_p, rel_d, sf.unsqueeze(-1).to(desc.dtype), thumb_cross.unsqueeze(-1).to(desc.dtype)],
            dim=-1,
        )  # (B, N, N, 6)

        bias = self.tree_dist_bias(td) + self.pair_geom_bias(pair_feat)  # (B, N, N, L*H)
        pair_mask = (joint_mask.unsqueeze(2) & joint_mask.unsqueeze(1)).unsqueeze(-1).to(bias.dtype)
        bias = bias * pair_mask
        L = max(self.dec_attn_layers, 1)
        H = self.dec_attn_heads
        return bias.view(B, N, N, L, H).permute(0, 3, 4, 1, 2).contiguous()

    def forward(
        self,
        desc: torch.Tensor,         # (B, N_max, 58)
        tip_desc: torch.Tensor,     # (B, 5, 16)
        adj: torch.Tensor,          # (B, N_max, N_max)
        joint_mask: torch.Tensor,   # (B, N_max) bool
        tip_mask: torch.Tensor,     # (B, 5) bool
        lo: torch.Tensor,           # (B, N_max)
        hi: torch.Tensor,           # (B, N_max)
        scale: torch.Tensor,        # (B,)
        tree_dist: torch.Tensor | None = None,
        same_finger: torch.Tensor | None = None,
    ) -> MorphologyCache:
        jm = joint_mask.unsqueeze(-1).to(desc.dtype)
        tm = tip_mask.unsqueeze(-1).to(tip_desc.dtype)

        h = self.joint_in(desc) * jm
        for layer in self.gnn_layers:
            h = layer(h, adj, joint_mask)

        h_cat = torch.cat([h, desc], dim=-1) * jm

        # Independent f_phi & g_omega
        K_joint_enc = self.f_phi_joint(h_cat) * jm
        K_tip_enc = self.f_phi_tip(tip_desc) * tm
        E_joint_dec = self.g_omega_joint(h_cat) * jm

        # Static pooled morphology vector
        scores = self.morph_j_score(h).squeeze(-1).masked_fill(~joint_mask, -1e9)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        h_pool = (w * h).sum(dim=1)
        t_pool = (K_tip_enc * tm).sum(dim=1) / tm.sum(dim=1).clamp_min(1.0)
        n_act_norm = joint_mask.float().sum(dim=-1, keepdim=True) / 24.0
        n_tip_norm = tip_mask.float().sum(dim=-1, keepdim=True) / 5.0
        z_morph = self.morph_proj(torch.cat([h_pool, t_pool, n_act_norm, n_tip_norm], dim=-1))

        # Static decoder pairwise structural attention bias
        dec_attn_bias = self._compute_dec_attn_bias(desc, adj, joint_mask, tree_dist=tree_dist, same_finger=same_finger)

        return MorphologyCache(
            K_joint_enc=K_joint_enc,
            K_tip_enc=K_tip_enc,
            E_joint_dec=E_joint_dec,
            z_morph=z_morph,
            joint_mask=joint_mask,
            tip_mask=tip_mask,
            lo=lo,
            hi=hi,
            scale=scale,
            adj=adj,
            dec_attn_bias=dec_attn_bias,
        )
