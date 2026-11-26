"""Human 21-point MediaPipe preprocessing, proportion jitter, and 6-token feature extraction (Sections 3.3 & 4.3)."""

from __future__ import annotations

import numpy as np
import torch

MP_PARENT = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], dtype=np.int64)
FINGER_KP = np.array(
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]],
    dtype=np.int64,
)
PALM_KP = np.array([0, 5, 9, 13, 17], dtype=np.int64)
HUMAN_TIP_IDX = np.array([4, 8, 12, 16, 20], dtype=np.int64)
HUMAN_NUM_TOKENS = 6   # 1 palm token + 5 finger tokens
HUMAN_TOKEN_DIM = 26   # unified feature dimension per token


def mirror_left_to_right(raw: np.ndarray) -> np.ndarray:
    """Mirror left-hand 21-point Manus MediaPipe coordinates across the palm symmetry plane to right-hand geometry.

    In Manus MediaPipe frames, flipping the local palmar lateral axis (or world X when wrist-centered)
    flips handedness; we mirror across the plane orthogonal to `(p_5 - p_17)` (index-to-pinky MCP vector)
    passing through the wrist so mirroring is completely invariant to global camera/glove orientation!
    """
    p = raw.astype(np.float32) - raw[:, :1].astype(np.float32)
    # Normal to the sagittal plane of the hand: index MCP (5) -> pinky MCP (17)
    n = p[:, 5] - p[:, 17]
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    proj = (p * n[:, None, :]).sum(axis=-1, keepdims=True)
    mirrored = p - 2.0 * proj * n[:, None, :]
    return mirrored.astype(np.float32)


def canonicalize(raw: np.ndarray, side: str = "right") -> np.ndarray:
    """Align `(N, 21, 3)` human landmarks (m) to the unified right-hand canonical palm frame.

    When `side == "left"`, landmarks are first reflected across the hand's sagittal symmetry plane
    (`mirror_left_to_right`) so both left and right human hands map into the exact same canonical frame:
    - Origin: mean of non-thumb MCP knuckles `[5, 9, 13, 17]`
    - +X: toward middle MCP/finger axis
    - +Z: palmar normal (`cross(p5 - p0, p17 - p0)`)
    - +Y: `Z x X` (thumb always sits on `-Y`)
    """
    side_norm = side.lower().strip()
    if side_norm not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    pts = mirror_left_to_right(raw) if side_norm == "left" else raw
    p0 = pts.astype(np.float64) - pts[:, :1].astype(np.float64)
    z = np.cross(p0[:, 5], p0[:, 17])
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    x = p0[:, 9] - (p0[:, 9] * z).sum(axis=-1, keepdims=True) * z
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-1)  # (N, 3, 3)
    # Center origin at mean non-thumb MCP so canonical frame origin matches HandSpec palm origin
    origin = p0[:, [5, 9, 13, 17]].mean(axis=1, keepdims=True)
    pc = np.einsum("nkd,nde->nke", p0 - origin, R)
    return pc.astype(np.float32)


def human_targets(pc: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute human finger polylines, arc fractions, lengths, characteristic scale S, and tips from `(B,21,3)`."""
    fkp = torch.as_tensor(FINGER_KP, device=pc.device)
    H = pc[:, fkp]                                    # (B, 5, 4, 3)
    seg = (H[:, :, 1:] - H[:, :, :-1]).norm(dim=-1)   # (B, 5, 3)
    L = seg.sum(dim=-1).clamp_min(1e-6)               # (B, 5)
    s = torch.cat([torch.zeros_like(seg[..., :1]), seg.cumsum(dim=-1)], dim=-1) / L[..., None]
    S = L[:, 1:].mean(dim=-1).clamp_min(1e-4)         # (B,) mean non-thumb finger length
    return dict(H=H, s=s, L=L, S=S, tips=H[:, :, 3], pc=pc)


def extract_human_6tokens(pc: torch.Tensor) -> torch.Tensor:
    """Extract 6 structured human tokens `(B, 6, 26)` in canonical palm frame (Section 3.3).

    - Token 0 (Palm): 5 normalized palm keypoints (15) + thumb->4-finger opposition distances (4)
      + 5 normalized finger length ratios (5) + palm width/length (2) = 26
    - Tokens 1..5 (Thumb..Little): 4 normalized keypoints (12) + 3 normalized bone vectors (9)
      + thumb->finger tip opposition vector (3) + opposition distance (1) + finger length ratio (1) = 26
    """
    B = pc.shape[0]
    t = human_targets(pc)
    S = t["S"][:, None, None]                         # (B, 1, 1)
    p = pc / S                                        # (B, 21, 3) scale-normalized

    fkp = torch.as_tensor(FINGER_KP, device=pc.device)
    pkp = torch.as_tensor(PALM_KP, device=pc.device)

    pf = p[:, fkp]                                    # (B, 5, 4, 3)
    bones = pf[:, :, 1:] - pf[:, :, :-1]              # (B, 5, 3, 3)
    tips_n = pf[:, :, 3]                              # (B, 5, 3)
    opp_vec = tips_n - tips_n[:, :1]                  # (B, 5, 3) thumb_tip -> finger_tip
    opp_dist = opp_vec.norm(dim=-1, keepdim=True)     # (B, 5, 1)
    len_ratio = (t["L"] / t["S"][:, None])[..., None] # (B, 5, 1)

    finger_tokens = torch.cat([
        pf.flatten(2),        # 12
        bones.flatten(2),     # 9
        opp_vec,              # 3
        opp_dist,             # 1
        len_ratio,            # 1
    ], dim=-1)                # (B, 5, 26)

    palm_pts = p[:, pkp].flatten(1)                   # (B, 15)
    palm_width = (p[:, 5] - p[:, 17]).norm(dim=-1, keepdim=True)
    palm_len = (p[:, 9] - p[:, 0]).norm(dim=-1, keepdim=True)
    palm_token = torch.cat([
        palm_pts,                      # 15
        opp_dist[:, 1:, 0],            # 4 (thumb to index/middle/ring/little)
        len_ratio[:, :, 0],            # 5
        palm_width,                    # 1
        palm_len,                      # 1
    ], dim=-1).unsqueeze(1)            # (B, 1, 26)

    return torch.cat([palm_token, finger_tokens], dim=1)  # (B, 6, 26)


def jitter_human_geometry(
    pc: torch.Tensor,
    palm_scale_jitter: float = 0.15,
    finger_len_jitter: float = 0.10,
    rot_sigma: float = 0.0,
) -> torch.Tensor:
    """Section 4.3 human geometry proportion jitter: +/-15% overall palm scale and +/-10% per-finger phalanx lengths.

    Works on `(B, 21, 3)` or `(B, T, 21, 3)` (applying one consistent scale per trajectory in the 4D case).
    """
    is_seq = pc.ndim == 4
    if is_seq:
        B, T, _, _ = pc.shape
        x = pc.clone()
    else:
        B = pc.shape[0]
        x = pc.unsqueeze(1).clone()  # (B, 1, 21, 3)
        T = 1

    dev = pc.device
    # Overall hand/palm scale jitter (+/-15%)
    if palm_scale_jitter > 0.0:
        g_scale = 1.0 + palm_scale_jitter * (2.0 * torch.rand(B, 1, 1, 1, device=dev) - 1.0)
        x = x * g_scale

    # Per-finger phalanx length jitter (+/-10%) + optional small joint angle perturbation
    fkp = torch.as_tensor(FINGER_KP, device=dev)
    f = x[:, :, fkp]                                  # (B, T, 5, 4, 3)
    bones = f[:, :, :, 1:] - f[:, :, :, :-1]          # (B, T, 5, 3, 3)
    if finger_len_jitter > 0.0:
        flen = 1.0 + finger_len_jitter * (2.0 * torch.rand(B, 1, 5, 3, 1, device=dev) - 1.0)
        bones = bones * flen

    if rot_sigma > 0.0:
        ang = (torch.randn(B, T, 5, 3, 3, device=dev) * rot_sigma).cumsum(dim=3)
        K = torch.zeros(B, T, 5, 3, 3, 3, device=dev)
        K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -ang[..., 2], ang[..., 1], -ang[..., 0]
        K = K - K.transpose(-1, -2)
        R = torch.matrix_exp(K)
        bones = (R @ bones[..., None]).squeeze(-1)

    x[:, :, fkp[:, 1:]] = f[:, :, :, :1] + bones.cumsum(dim=3)
    return x if is_seq else x.squeeze(1)


def _unit_t(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _fabrik(chain: torch.Tensor, target: torch.Tensor, iters: int = 24) -> torch.Tensor:
    c = chain.clone()
    root = c[:, 0].clone()
    ln = (c[:, 1:] - c[:, :-1]).norm(dim=-1)
    for _ in range(iters):
        c[:, 3] = target
        for j in (2, 1, 0):
            c[:, j] = c[:, j + 1] + _unit_t(c[:, j] - c[:, j + 1]) * ln[:, j:j + 1]
        c[:, 0] = root
        for j in (1, 2, 3):
            c[:, j] = c[:, j - 1] + _unit_t(c[:, j] - c[:, j - 1]) * ln[:, j - 1:j]
    return c


def synth_pinch_frames(
    pc: np.ndarray,
    n_per_finger: int = 3000,
    seed: int = 0,
    seed_dist: float = 0.055,
) -> np.ndarray:
    """Generate diverse synthetic pinch frames across all 4 non-thumb fingers using FABRIK."""
    if n_per_finger <= 0 or len(pc) == 0:
        return np.zeros((0, 21, 3), dtype=np.float32)
    g = torch.Generator().manual_seed(seed)
    p = torch.as_tensor(pc, dtype=torch.float32)
    th = FINGER_KP[0]
    out = []
    for k in range(1, 5):
        fk = FINGER_KP[k]
        d = (p[:, fk[3]] - p[:, th[3]]).norm(dim=-1)
        thr = max(seed_dist, float(d.kthvalue(min(500, len(d))).values.item()))
        cand = (d <= thr).nonzero().flatten()
        if len(cand) == 0:
            continue
        idx = cand[torch.randint(len(cand), (n_per_finger,), generator=g)]
        x = p[idx].clone()
        tt, ft = x[:, th[3]], x[:, fk[3]]
        gap = 0.002 + 0.016 * torch.rand(n_per_finger, 1, generator=g)
        w = 0.35 + 0.35 * torch.rand(n_per_finger, 1, generator=g)
        u = _unit_t(ft - tt)
        mid = tt + w * (ft - tt)
        x[:, fk] = _fabrik(x[:, fk], mid + 0.5 * gap * u)
        x[:, th] = _fabrik(x[:, th], mid - 0.5 * gap * u)
        ok = ((x[:, fk[3]] - x[:, th[3]]).norm(dim=-1) - gap[:, 0]).abs() < 1e-3
        if ok.any():
            out.append(x[ok])
    return torch.cat(out, dim=0).numpy() if out else np.zeros((0, 21, 3), dtype=np.float32)
