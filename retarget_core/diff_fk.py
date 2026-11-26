"""Batched differentiable forward kinematics supporting mimic joints and padded multi-hand batches."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .human_preprocess import human_targets

from .urdf_standardizer import (
    INERTIA_I0,
    INERTIAL_SLICE,
    JOINT_DESC_DIM,
    MASS_EPS,
    NF,
    OTHER,
    P_MAX,
    TIP_DESC_DIM,
    HandSpec,
)

MAX_TREE_DIST = 8
N_MAX_DEFAULT = 24
JOINT_FB_DIM = 8
TIP_FB_DIM = 7


class HandBank:
    """Packs heterogeneous HandSpecs into padded GPU tensors up to N_max actuated joints."""

    def __init__(self, specs: list[HandSpec], device="cpu", dtype=torch.float32, n_max: int = N_MAX_DEFAULT):
        self.specs = specs
        self.names = [s.name for s in specs]
        H = len(specs)
        J = max(s.J for s in specs)
        Na = max(n_max, max(s.Na for s in specs))
        K = max(len(s.kp_frame) for s in specs)
        S = max(len(s.sph_kp) for s in specs)
        self.H, self.J, self.Na, self.K, self.S = H, J, Na, K, S

        T_origin = np.tile(np.eye(4), (H, J, 1, 1))
        axis = np.zeros((H, J, 3))
        jtype = np.zeros((H, J), np.int64)
        parent = -np.ones((H, J), np.int64)
        A = np.zeros((H, J, Na))
        b = np.zeros((H, J))
        lo = np.zeros((H, Na))
        hi = np.zeros((H, Na))
        act_mask = np.zeros((H, Na), bool)
        kp_frame = np.zeros((H, K), np.int64)
        kp_offset = np.zeros((H, K, 3))
        poly_kp = np.zeros((H, NF, P_MAX), np.int64)
        poly_mask = np.zeros((H, NF, P_MAX), bool)
        poly_s = np.zeros((H, NF, P_MAX))
        tip_kp = np.zeros((H, NF), np.int64)
        sph_kp = np.zeros((H, S), np.int64)
        sph_r = np.zeros((H, S))
        sph_f = np.zeros((H, S), np.int64)
        sph_mask = np.zeros((H, S), bool)
        desc = np.zeros((H, Na, JOINT_DESC_DIM))
        tip_desc = np.zeros((H, NF, TIP_DESC_DIM))
        adj = np.zeros((H, Na, Na))
        tree_dist = np.full((H, Na, Na), MAX_TREE_DIST, np.int64)
        same_finger = np.zeros((H, Na, Na), bool)
        label = np.full((H, Na), OTHER, np.int64)
        primary = np.zeros((H, Na), np.int64)
        on_finger = np.zeros((H, J, NF), bool)
        act_mass = np.zeros((H, Na))
        act_com = np.zeros((H, Na, 3))
        act_inertia6 = np.zeros((H, Na, 6))

        for i, s in enumerate(specs):
            T_origin[i, :s.J] = s.T_origin
            axis[i, :s.J], jtype[i, :s.J], parent[i, :s.J] = s.axis, s.jtype, s.parent
            A[i, :s.J, :s.Na], b[i, :s.J] = s.A, s.b
            lo[i, :s.Na], hi[i, :s.Na], act_mask[i, :s.Na] = s.lo, s.hi, True
            k = len(s.kp_frame)
            kp_frame[i, :k], kp_offset[i, :k] = s.kp_frame, s.kp_offset
            m = s.poly_kp >= 0
            poly_kp[i] = np.where(m, s.poly_kp, 0)
            poly_mask[i], poly_s[i] = m, s.poly_s
            tip_kp[i] = np.where(m.any(1), s.poly_kp[np.arange(NF), np.maximum(m.sum(1) - 1, 0)], 0)
            n = len(s.sph_kp)
            sph_kp[i, :n], sph_r[i, :n], sph_f[i, :n], sph_mask[i, :n] = s.sph_kp, s.sph_r, s.sph_finger, True
            desc[i, :s.Na] = s.desc
            tip_desc[i] = s.tip_desc
            adj[i, :s.Na, :s.Na] = s.adj
            tree_dist[i, :s.Na, :s.Na] = np.clip(s.tree_dist, 0, MAX_TREE_DIST)
            same_finger[i, :s.Na, :s.Na] = s.same_finger
            label[i, :s.Na] = s.act_label
            primary[i, :s.Na] = s.primary
            on_finger[i, :s.J] = s.on_finger
            act_mass[i, :s.Na] = s.act_mass
            act_com[i, :s.Na] = s.act_com
            act_inertia6[i, :s.Na] = s.act_inertia6

        def _t(x):
            return torch.as_tensor(x, dtype=dtype if x.dtype.kind == "f" else None, device=device)

        self.T = dict(
            T_origin=_t(T_origin),
            axis=_t(axis),
            jtype=_t(jtype),
            parent=_t(parent),
            A=_t(A),
            b=_t(b),
            lo=_t(lo),
            hi=_t(hi),
            act_mask=_t(act_mask),
            T_root=_t(np.array([s.T_root for s in specs])),
            kp_frame=_t(kp_frame),
            kp_offset=_t(kp_offset),
            poly_kp=_t(poly_kp),
            poly_mask=_t(poly_mask),
            poly_s=_t(poly_s),
            tip_kp=_t(tip_kp),
            finger_present=_t(np.array([s.finger_present for s in specs])),
            finger_len=_t(np.array([s.finger_len for s in specs])),
            scale=_t(np.array([s.scale for s in specs])),
            sph_kp=_t(sph_kp),
            sph_r=_t(sph_r),
            sph_finger=_t(sph_f),
            sph_mask=_t(sph_mask),
            desc=_t(desc),
            tip_desc=_t(tip_desc),
            adj=_t(adj),
            tree_dist=_t(tree_dist),
            same_finger=_t(same_finger),
            label=_t(label),
            primary=_t(primary),
            on_finger=_t(on_finger),
            act_mass=_t(act_mass),
            act_com=_t(act_com),
            act_inertia6=_t(act_inertia6),
        )

    def gather(self, hid: torch.Tensor) -> dict[str, torch.Tensor]:
        return {k: v[hid] for k, v in self.T.items()}

    def to(self, device) -> "HandBank":
        self.T = {k: v.to(device) for k, v in self.T.items()}
        return self


def _motion(axis: torch.Tensor, q: torch.Tensor, jtype: torch.Tensor) -> torch.Tensor:
    """(...,3),(...),(...) -> (...,4,4) Rodrigues rotation (revolute) or translation (prismatic)."""
    rev = (jtype == 0).to(q.dtype)[..., None]
    pri = (jtype == 1).to(q.dtype)[..., None]
    a = axis * rev
    zero = torch.zeros_like(a[..., 0])
    K = torch.stack([
        torch.stack([zero, -a[..., 2], a[..., 1]], -1),
        torch.stack([a[..., 2], zero, -a[..., 0]], -1),
        torch.stack([-a[..., 1], a[..., 0], zero], -1),
    ], -2)
    s = torch.sin(q)[..., None, None]
    c = torch.cos(q)[..., None, None]
    eye = torch.eye(3, dtype=q.dtype, device=q.device)
    R = eye + s * K + (1.0 - c) * (K @ K)
    t = axis * pri * q[..., None]
    top = torch.cat([R, t[..., None]], -1)
    bottom = torch.zeros_like(top[..., :1, :])
    bottom[..., 0, 3] = 1.0
    return torch.cat([top, bottom], -2)


def fk_frames(h: dict[str, torch.Tensor], q_act: torch.Tensor, canonical: bool = True) -> torch.Tensor:
    """Compute (B, J+1, 4, 4) joint transforms from actuated angles q_act (B, Na) with mimic coupling."""
    q = torch.einsum("bja,ba->bj", h["A"], q_act) + h["b"]
    B, J = q.shape
    L = h["T_origin"] @ _motion(h["axis"], q, h["jtype"])
    T = q.new_zeros(B, J + 1, 4, 4)
    T[:, 0] = h["T_root"] if canonical else torch.eye(4, dtype=q.dtype, device=q.device)
    ar = torch.arange(B, device=q.device)
    par = h["parent"] + 1
    for j in range(J):
        T[:, j + 1] = T[ar, par[:, j]] @ L[:, j]
    return T


def frames_to_keypoints(h: dict[str, torch.Tensor], T: torch.Tensor) -> torch.Tensor:
    ar = torch.arange(T.shape[0], device=T.device)
    Tk = T[ar[:, None], h["kp_frame"]]
    return (Tk[..., :3, :3] @ h["kp_offset"][..., None]).squeeze(-1) + Tk[..., :3, 3]


def forward_kinematics(h: dict[str, torch.Tensor], q_act: torch.Tensor, canonical: bool = True) -> torch.Tensor:
    """Differentiable keypoints (B, K, 3) in canonical palm frame."""
    return frames_to_keypoints(h, fk_frames(h, q_act, canonical))


def fingertip_positions(h: dict[str, torch.Tensor], q_act: torch.Tensor, canonical: bool = True) -> torch.Tensor:
    """Differentiable fingertip 3D positions (B, 5, 3) in canonical palm frame (zeroed if finger absent)."""
    kp = forward_kinematics(h, q_act, canonical)
    ar = torch.arange(kp.shape[0], device=kp.device)
    tips = kp[ar[:, None], h["tip_kp"]]
    return tips * h["finger_present"].to(tips.dtype)[..., None]


@torch.no_grad()
def compute_kinematic_feedback(
    h: dict[str, torch.Tensor],
    q_prev: torch.Tensor,
    pc_target: torch.Tensor,
    pinch_on: float = 0.030,
    pinch_tau: float = 0.004,
    d_contact: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute current-pose fingertips and exact analytical Jacobian closed-loop feedback for any URDF hand.

    100% invariant to joint-axis sign flips, arbitrary rest-pose rotations (e.g., rotated thumb CMC frames),
    and underactuated mimic chains `q_full = A @ q_act + b`.

    Returns:
      r_tips:   `(B, 5, 3)` current fingertip positions in canonical palm frame (m)
      joint_fb: `(B, N_max, 8)` `[j_own(3), g_tip(1), g_opp(1), g_pinch(1), j_norm(1), pinch_ind(1)]`
      tip_fb:   `(B, 5, 7)` `[r_tip/S_r(3), e_tip(3), active_pinch_gap(1)]`
    """
    dtype = q_prev.dtype
    B = q_prev.shape[0]
    ar = torch.arange(B, device=q_prev.device)

    # 1. Single FK pass -> frames, keypoints, fingertip & base positions
    T = fk_frames(h, q_prev, canonical=True)                        # (B, J+1, 4, 4)
    kp = frames_to_keypoints(h, T)                                  # (B, K, 3)
    present = h["finger_present"]                                   # (B, 5) bool
    pf = present.to(dtype)                                          # (B, 5)
    r_tips = kp[ar[:, None], h["tip_kp"]] * pf[..., None]           # (B, 5, 3)
    r_base = kp[ar[:, None], h["poly_kp"][:, :, 0]] * pf[..., None] # (B, 5, 3)

    S_r = h["scale"].clamp_min(1e-4)                                # (B,)
    L_r = h["finger_len"].clamp_min(1e-5)                           # (B, 5)
    jm = h["act_mask"].to(dtype)                                    # (B, Na)
    half_span = 0.5 * (h["hi"] - h["lo"]).clamp_min(1e-4)           # (B, Na)

    # 2. Exact analytical fingertip Jacobian w.r.t. normalized action u in [-1, 1]
    p_j = T[:, 1:, :3, 3]                                           # (B, J, 3)
    a_j = (T[:, 1:, :3, :3] @ h["axis"][..., None]).squeeze(-1)     # (B, J, 3)
    r_vec = r_tips[:, :, None, :] - p_j[:, None, :, :]              # (B, 5, J, 3)
    a_exp = a_j[:, None, :, :].expand(-1, NF, -1, -1)               # (B, 5, J, 3)
    J_rev = torch.cross(a_exp, r_vec, dim=-1)                       # (B, 5, J, 3)
    is_pri = (h["jtype"] == 1)[:, None, :, None]                    # (B, 1, J, 1)
    J_full = torch.where(is_pri, a_exp, J_rev)                      # (B, 5, J, 3)
    J_full = J_full * h["on_finger"].permute(0, 2, 1)[..., None].to(dtype)
    # Project through mimic matrix A (B, J, Na) and scale by half_span -> d(r_tip)/du
    J_u = torch.einsum("bfjd,bja->bfad", J_full, h["A"]) * (half_span * jm)[:, None, :, None]

    # 3. Target human errors at current pose q_prev
    ht = human_targets(pc_target)
    H = ht["H"]                                                     # (B, 5, 4, 3)
    L_h = ht["L"].clamp_min(1e-5)                                   # (B, 5)
    S_h = ht["S"].clamp_min(1e-4)                                   # (B,)
    h_tips = ht["tips"]                                             # (B, 5, 3)

    ur_tip = (r_tips - r_base) / L_r[..., None]                     # (B, 5, 3)
    uh_tip = (h_tips - H[:, :, 0]) / L_h[..., None]                 # (B, 5, 3)
    e_tip = (uh_tip - ur_tip) * pf[..., None]                       # (B, 5, 3)

    pair = (present[:, :1] & present[:, 1:]).to(dtype)              # (B, 4)
    dh = (h_tips[:, :1] - h_tips[:, 1:]).norm(dim=-1)               # (B, 4)
    gate = torch.sigmoid((pinch_on - dh) / pinch_tau) * pair        # (B, 4)

    vr = (r_tips[:, :1] - r_tips[:, 1:]) / S_r[:, None, None]       # (B, 4, 3)
    vh = (h_tips[:, :1] - h_tips[:, 1:]) / S_h[:, None, None]       # (B, 4, 3)
    e_opp = (vh - vr) * pair[..., None]                             # (B, 4, 3)
    dr = ((r_tips[:, :1] - r_tips[:, 1:]).pow(2).sum(dim=-1) + 1e-12).sqrt() / S_r[:, None]
    dr_excess = F.relu(dr - d_contact)                              # (B, 4)
    vr_hat = vr / dr.clamp_min(1e-4)[..., None]                     # (B, 4, 3)

    # 4. Signed per-joint descent gradients w.r.t. u_a (positive => increase u_a)
    J_tip_norm = J_u / L_r[:, :, None, None]                        # (B, 5, Na, 3)
    g_tip = (J_tip_norm * e_tip[:, :, None, :]).sum(dim=(1, 3))     # (B, Na)

    J_opp = (J_u[:, :1] - J_u[:, 1:]) / S_r[:, None, None, None]    # (B, 4, Na, 3)
    opp_dot = (J_opp * e_opp[:, :, None, :]).sum(dim=-1)            # (B, 4, Na)
    g_opp = ((1.0 - gate)[:, :, None] * opp_dot).sum(dim=1)         # (B, Na)

    # Gradient of -0.5 * (dr - d_contact)^2 w.r.t. u_a is -dr_excess * (vr_hat . d(vr)/du_a)
    pinch_dot = -(dr_excess[:, :, None] * (J_opp * vr_hat[:, :, None, :]).sum(dim=-1)) + 0.15 * opp_dot
    g_pinch = (gate[:, :, None] * pinch_dot).sum(dim=1)             # (B, Na)

    # Own-finger current-pose 3D Jacobian vector (or summed for shared/coupled palm joints)
    lbl = h["label"]                                                # (B, Na)
    lbl_c = lbl.clamp(0, NF - 1)
    J_perm = J_tip_norm.permute(0, 2, 1, 3)                         # (B, Na, 5, 3)
    idx_f = lbl_c[..., None, None].expand(-1, -1, 1, 3)
    j_own_single = torch.gather(J_perm, 2, idx_f).squeeze(2)        # (B, Na, 3)
    j_own_sum = J_tip_norm.sum(dim=1)                               # (B, Na, 3)
    is_other = (lbl >= NF).unsqueeze(-1)
    j_own = torch.where(is_other, j_own_sum, j_own_single)          # (B, Na, 3)
    j_norm = j_own.norm(dim=-1)                                     # (B, Na)

    # Active pinch indicator per joint's finger
    pinch_per_f = torch.cat([gate.amax(dim=-1, keepdim=True), gate], dim=-1)  # (B, 5)
    pinch_ind = torch.gather(pinch_per_f, 1, lbl_c) * (lbl < NF).to(dtype)

    joint_fb = torch.cat([
        torch.tanh(j_own),
        torch.tanh(2.5 * g_tip)[..., None],
        torch.tanh(2.5 * g_opp)[..., None],
        torch.tanh(5.0 * g_pinch)[..., None],
        torch.tanh(j_norm)[..., None],
        pinch_ind[..., None],
    ], dim=-1) * jm[..., None]                                      # (B, Na, 8)

    tip_pinch_gap = torch.cat([(gate * dr_excess).amax(dim=-1, keepdim=True), gate * dr_excess], dim=-1)
    tip_fb = torch.cat([
        (r_tips / S_r[:, None, None]).clamp(-4.0, 4.0),
        e_tip.clamp(-2.0, 2.0),
        tip_pinch_gap.clamp(0.0, 2.0)[..., None],
    ], dim=-1) * pf[..., None]                                      # (B, 5, 7)

    return r_tips, joint_fb, tip_fb


def perturb_inertial_descriptor(
    desc: torch.Tensor,
    act_mass: torch.Tensor,
    act_com: torch.Tensor,
    act_inertia6: torch.Tensor,
    scale: torch.Tensor,
    mass_mult: torch.Tensor,
    com_offset: torch.Tensor | None = None,
    inertia_mult: torch.Tensor | None = None,
    obs_noise_frac: float = 0.0,
) -> torch.Tensor:
    """Recompute the inertial slice of d_j after dynamics DR and optional +-obs_noise_frac noise."""
    out = desc.clone()
    m = act_mass * mass_mult
    com = act_com if com_offset is None else (act_com + com_offset)
    i6 = act_inertia6 * (mass_mult[..., None] if inertia_mult is None else inertia_mult)
    if obs_noise_frac > 0.0:
        m = m * (1.0 + obs_noise_frac * (2.0 * torch.rand_like(m) - 1.0))
        com = com * (1.0 + obs_noise_frac * (2.0 * torch.rand_like(com) - 1.0))
        i6 = i6 * (1.0 + obs_noise_frac * (2.0 * torch.rand_like(i6) - 1.0))
    m_feat = torch.log(m.clamp_min(0.0) + MASS_EPS)[..., None]
    com_feat = com / scale.clamp_min(1e-4)[..., None, None]
    i_feat = torch.sign(i6) * torch.log1p(torch.abs(i6) / INERTIA_I0)
    out[..., INERTIAL_SLICE] = torch.cat([m_feat, com_feat, i_feat], dim=-1)
    return out
