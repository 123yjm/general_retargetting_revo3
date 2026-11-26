"""URDF standardizer: canonical palm frame, open pose, mimic resolution, kinematic + inertial descriptors.

Implements Sections 1.2, 3.1, and 3.2 of the URMA-Hand technical specification:
- Unifies palm local frame (+X toward middle tip, +Z palmar normal, +Y = Z x X so right thumb is -Y)
- Resolves passive mimic chains into q_full = A @ q_act + b
- Extracts normalized static joint descriptors d_j (kinematics + mimic + log-normalized m, CoM, 6D I + actuator)
- Extracts fingertip static descriptors d_f and parent-child GNN adjacency over actuated joints.
"""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

FINGERS = ("thumb", "index", "middle", "ring", "little")
NF = len(FINGERS)
OTHER = NF
P_MAX = 7
DEDUP_TOL = 0.004
LINK_R, TIP_R = 0.06, 0.04

INERTIA_I0 = 1e-6
MASS_EPS = 1e-4

# Slice layout of d_j (JOINT_DESC_DIM = 58):
# [0:3]   pos / scale
# [3:6]   parent_delta / scale
# [6:9]   axis (unit)
# [9:13]  limits [lo, hi, rng, mid]
# [13:15] jtype one-hot [rev, pri]
# [15:21] finger label one-hot (6)
# [21:24] vector to primary fingertip / scale
# [24:25] dist to primary fingertip / scale
# [25:27] hop counts [hops_to_palm/4, hops_to_tip/4]
# [27:30] mimic info [is_mimic_driver, num_mimic_children/4, sum(|A[:,a]|)]
# [30:45] nominal tip Jacobian sensitivity (5 * 3)
# [45:55] normalized inertial [log(m+eps), com/scale (3), sign(I)*log(1+|I|/I0) (6)]
# [55:58] actuator [kp/20, kd/1, effort/5]
INERTIAL_SLICE = slice(45, 55)
JOINT_DESC_DIM = 58
TIP_DESC_DIM = 16


# ----------------------------------------------------------------------------- geometry helpers
def rpy_to_R(rpy) -> np.ndarray:
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def R_to_rpy(R: np.ndarray) -> np.ndarray:
    sp = float(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(sp) < 1.0 - 1e-9:
        return np.array([math.atan2(R[2, 1], R[2, 2]), math.asin(sp), math.atan2(R[1, 0], R[0, 0])])
    return np.array([0.0, math.copysign(math.pi / 2.0, sp), math.atan2(-R[0, 1], R[1, 1])])


def make_T(R=None, p=None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    a = a / n
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def motion_T(jtype: int, axis, q: float) -> np.ndarray:
    if jtype == 1:
        return make_T(p=np.asarray(axis, dtype=np.float64) * q)
    return make_T(R=axis_angle(axis, q))


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def normalize_inertial(mass: float, com: np.ndarray, inertia6: np.ndarray, scale: float) -> np.ndarray:
    """Section 3.2 dimensional normalization for (m, CoM, 6D I)."""
    m_feat = np.array([math.log(max(float(mass), 0.0) + MASS_EPS)], dtype=np.float64)
    com_feat = np.asarray(com, dtype=np.float64) / max(scale, 1e-4)
    i6 = np.asarray(inertia6, dtype=np.float64)
    i_feat = np.sign(i6) * np.log1p(np.abs(i6) / INERTIA_I0)
    return np.concatenate([m_feat, com_feat, i_feat])


# ----------------------------------------------------------------------------- URDF data structures
@dataclass
class ULink:
    name: str
    mass: float
    com: np.ndarray        # (3,) local CoM in link frame
    inertia6: np.ndarray   # (6,) [ixx, iyy, izz, ixy, ixz, iyz] in link frame
    has_inertial: bool


@dataclass
class UJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    effort: float
    velocity: float
    damping: float
    friction: float
    mimic: tuple[str, float, float] | None

    @property
    def movable(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")


def _floats(s: str | None, default) -> np.ndarray:
    return np.array([float(v) for v in s.split()], dtype=np.float64) if s else np.array(default, dtype=np.float64)


def _rot_inertia(R: np.ndarray, i6: np.ndarray) -> np.ndarray:
    I = np.array([
        [i6[0], i6[3], i6[4]],
        [i6[3], i6[1], i6[5]],
        [i6[4], i6[5], i6[2]],
    ], dtype=np.float64)
    Ir = R @ I @ R.T
    return np.array([Ir[0, 0], Ir[1, 1], Ir[2, 2], Ir[0, 1], Ir[0, 2], Ir[1, 2]], dtype=np.float64)


def parse_urdf(text: str) -> tuple[dict[str, ULink], list[UJoint]]:
    root = ET.fromstring(text)
    links: dict[str, ULink] = {}
    for l in root.findall("link"):
        name = l.get("name")
        iner = l.find("inertial")
        if iner is not None:
            o = iner.find("origin")
            com = _floats(o.get("xyz") if o is not None else None, [0.0, 0.0, 0.0])
            rpy = _floats(o.get("rpy") if o is not None else None, [0.0, 0.0, 0.0])
            m_el = iner.find("mass")
            mass = float(m_el.get("value", 0.01)) if m_el is not None else 0.01
            i_el = iner.find("inertia")
            if i_el is not None:
                i6 = np.array([
                    float(i_el.get("ixx", 1e-6)), float(i_el.get("iyy", 1e-6)), float(i_el.get("izz", 1e-6)),
                    float(i_el.get("ixy", 0.0)), float(i_el.get("ixz", 0.0)), float(i_el.get("iyz", 0.0)),
                ], dtype=np.float64)
                i6 = _rot_inertia(rpy_to_R(rpy), i6)
            else:
                i6 = np.array([1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0], dtype=np.float64)
            has_iner = mass > 1e-8
        else:
            mass = 0.015
            com = np.zeros(3, dtype=np.float64)
            i6 = np.array([2e-6, 2e-6, 2e-6, 0.0, 0.0, 0.0], dtype=np.float64)
            has_iner = False
        if mass <= 1e-8:
            mass = 0.005
            i6 = np.array([1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0], dtype=np.float64)
        links[name] = ULink(name, mass, com, i6, has_iner)

    joints: list[UJoint] = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = _floats(o.get("xyz") if o is not None else None, [0.0, 0.0, 0.0])
        rpy = _floats(o.get("rpy") if o is not None else None, [0.0, 0.0, 0.0])
        ax = j.find("axis")
        axis = unit(_floats(ax.get("xyz") if ax is not None else None, [1.0, 0.0, 0.0]))
        lim = j.find("limit")
        lower = float(lim.get("lower", 0.0)) if lim is not None else 0.0
        upper = float(lim.get("upper", 0.0)) if lim is not None else 0.0
        effort = float(lim.get("effort", 3.0)) if lim is not None else 3.0
        velocity = float(lim.get("velocity", 6.0)) if lim is not None else 6.0
        if j.get("type") == "continuous":
            lower, upper = -math.pi, math.pi
        dyn = j.find("dynamics")
        damping = float(dyn.get("damping", 0.05)) if dyn is not None else 0.05
        friction = float(dyn.get("friction", 0.0)) if dyn is not None else 0.0
        m = j.find("mimic")
        mimic = (m.get("joint"), float(m.get("multiplier", 1.0)), float(m.get("offset", 0.0))) if m is not None else None
        joints.append(UJoint(
            j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
            make_T(rpy_to_R(rpy), xyz), axis, lower, upper, effort, velocity, damping, friction, mimic,
        ))
    return links, joints


def resolve_mesh_path(urdf_path: Path, filename: str) -> Path | None:
    if filename.startswith("package://"):
        rel = Path(filename[len("package://"):])
        rel_no_pkg = Path(*rel.parts[1:]) if len(rel.parts) > 1 else rel
        for base in (urdf_path.parent, *urdf_path.parents[:4]):
            for cand in (base / rel, base / rel_no_pkg):
                if cand.exists():
                    return cand.resolve()
        return None
    p = Path(filename).expanduser()
    if p.is_absolute():
        return p if p.exists() else None
    cand = (urdf_path.parent / p).resolve()
    return cand if cand.exists() else None


# ----------------------------------------------------------------------------- HandSpec
@dataclass
class HandSpec:
    name: str
    display_name: str
    split: str
    joint_names: list[str]      # J movable joints (topological order)
    act_names: list[str]        # Na actuated joints (mimic roots)
    parent: np.ndarray          # (J,) parent movable joint, -1 = palm
    T_origin: np.ndarray        # (J,4,4) parent frame -> joint frame at q=0
    axis: np.ndarray            # (J,3) joint axis in joint frame
    jtype: np.ndarray           # (J,) 0=revolute, 1=prismatic
    A: np.ndarray               # (J,Na) q_full = A @ q_act + b
    b: np.ndarray               # (J,)
    lo: np.ndarray              # (Na,)
    hi: np.ndarray              # (Na,)
    T_root: np.ndarray          # (4,4) palm_link -> canonical palm frame
    kp_frame: np.ndarray        # (K,) 0=palm, j+1=movable joint j
    kp_offset: np.ndarray       # (K,3)
    finger_present: np.ndarray  # (5,) bool
    poly_kp: np.ndarray         # (5,P_MAX) keypoint indices (-1 padded)
    poly_s: np.ndarray          # (5,P_MAX) arc-length fraction [0..1]
    finger_len: np.ndarray      # (5,) nominal finger length (m)
    sph_kp: np.ndarray          # (S,) collision sphere keypoint indices
    sph_r: np.ndarray           # (S,) collision sphere radii (m)
    sph_finger: np.ndarray      # (S,) finger index of each sphere
    scale: float                # L_palm characteristic scale (m)
    act_label: np.ndarray       # (Na,) finger 0..4 or OTHER
    desc: np.ndarray            # (Na, JOINT_DESC_DIM)
    tip_desc: np.ndarray        # (5, TIP_DESC_DIM)
    adj: np.ndarray             # (Na, Na) symmetric normalized adjacency for GNN (including self-loops)
    tree_dist: np.ndarray       # (Na, Na)
    same_finger: np.ndarray     # (Na, Na)
    act_mass: np.ndarray        # (Na,) raw effective mass (kg)
    act_com: np.ndarray         # (Na, 3) raw effective CoM (m)
    act_inertia6: np.ndarray    # (Na, 6) raw effective 6D inertia (kg*m^2)
    primary: np.ndarray         # (Na,) movable joint index of each actuated joint
    on_finger: np.ndarray       # (J, 5) bool
    side: str = "right"         # "right" or "left"
    urdf_path: str = ""
    yaml_path: str = ""
    info: dict = field(default_factory=dict)

    def to_policy_spec(self) -> "HandSpec":
        """Return a right-hand-canonical HandSpec suitable for the unified URMA-Hand policy.

        - Right hands are returned directly.
        - Left hands are mapped via the exact sagittal-plane reflection isomorphism `M = diag(1, -1, 1)`,
          preserving joint angles `q`, limits `[lo, hi]`, and mimic couplings `(A, b)` identically.
        """
        if self.side == "right":
            return self
        return mirror_hand_spec_chirality(self)

    @property
    def J(self) -> int:
        return len(self.joint_names)

    @property
    def Na(self) -> int:
        return len(self.act_names)

    def q_full(self, q_act: np.ndarray) -> np.ndarray:
        return self.A @ q_act + self.b

    def q_open(self) -> np.ndarray:
        return np.clip(0.0, self.lo, self.hi)

    def frames(self, q_act: np.ndarray, canonical: bool = True) -> np.ndarray:
        q = self.q_full(q_act)
        F = np.zeros((self.J + 1, 4, 4), dtype=np.float64)
        F[0] = self.T_root if canonical else np.eye(4)
        for j in range(self.J):
            F[j + 1] = F[self.parent[j] + 1] @ self.T_origin[j] @ motion_T(int(self.jtype[j]), self.axis[j], float(q[j]))
        return F

    def keypoints(self, q_act: np.ndarray, canonical: bool = True) -> np.ndarray:
        F = self.frames(q_act, canonical)[self.kp_frame]
        return np.einsum("kij,kj->ki", F[:, :3, :3], self.kp_offset) + F[:, :3, 3]

    def tip_positions(self, q_act: np.ndarray, canonical: bool = True) -> np.ndarray:
        kp = self.keypoints(q_act, canonical)
        tips = np.zeros((NF, 3), dtype=np.float64)
        for fi in range(NF):
            if self.finger_present[fi]:
                valid = self.poly_kp[fi][self.poly_kp[fi] >= 0]
                tips[fi] = kp[valid[-1]]
        return tips


def _curl_sign(lo: float, hi: float) -> int:
    if hi > 1.2 * max(-lo, 0.0) and hi > 0.2:
        return 1
    if -lo > 1.2 * max(hi, 0.0) and -lo > 0.2:
        return -1
    return 0


def build_hand_spec(
    urdf_text: str,
    ann: dict,
    name: str | None = None,
    urdf_path: str = "",
    yaml_path: str = "",
) -> HandSpec:
    name = name or ann.get("name", "hand")
    display_name = ann.get("display_name", name)
    split = ann.get("split", "train")
    side = str(ann.get("side", "left" if name.endswith("_left") else "right")).lower().strip()
    if side not in ("right", "left"):
        raise ValueError(f"{name}: side must be 'right' or 'left', got {side!r}")
    links, joints = parse_urdf(urdf_text)
    by_name = {j.name: j for j in joints}
    by_child = {j.child: j for j in joints}
    children = defaultdict(list)
    for j in joints:
        children[j.parent].append(j)

    palm = ann["palm_link"]
    excl = {k: float(v) for k, v in (ann.get("exclude_joints") or {}).items()}
    tips = {f: ann["fingers"][f] for f in FINGERS if f in ann["fingers"]}
    tip_offsets = ann.get("tip_offsets") or {}
    base_links = ann.get("base_links") or {}
    act_cfg = ann.get("actuator") or {}
    default_kp = float(act_cfg.get("kp", 12.0))
    default_kd = float(act_cfg.get("kd", 0.4))

    def path_from_palm(link: str) -> list[UJoint]:
        path = []
        cur = link
        while cur != palm:
            if cur not in by_child:
                raise ValueError(f"{name}: link '{link}' is not below palm_link '{palm}'")
            path.append(by_child[cur])
            cur = by_child[cur].parent
        return path[::-1]

    paths = {f: path_from_palm(t) for f, t in tips.items()}
    on_chain = {j.name for p in paths.values() for j in p}
    is_dof = lambda j: j.movable and j.name not in excl

    # Traverse palm subtree along finger chains, merging fixed/excluded joints into T_origin
    full: list[UJoint] = []
    T_origin_list: list[np.ndarray] = []
    parent_list: list[int] = []
    link_frame: dict[str, tuple[int, np.ndarray]] = {palm: (-1, np.eye(4))}

    def visit(link: str):
        for j in children[link]:
            if j.name not in on_chain:
                continue
            fidx, Trel = link_frame[link]
            To = Trel @ j.origin
            if is_dof(j):
                link_frame[j.child] = (len(full), np.eye(4))
                full.append(j)
                T_origin_list.append(To)
                parent_list.append(fidx)
            else:
                M = motion_T(1 if j.type == "prismatic" else 0, j.axis, excl.get(j.name, 0.0)) if j.movable else np.eye(4)
                link_frame[j.child] = (fidx, To @ M)
            visit(j.child)

    visit(palm)
    J = len(full)
    idx_of = {j.name: i for i, j in enumerate(full)}

    # Resolve mimic chains: q_j = m * q_root + o
    def resolve(jn: str, depth: int = 0):
        if depth > 16:
            raise ValueError(f"{name}: mimic cycle at {jn}")
        j = by_name[jn]
        if jn in excl:
            return None, 0.0, excl[jn]
        if j.mimic is None:
            return jn, 1.0, 0.0
        r, m2, o2 = resolve(j.mimic[0], depth + 1)
        m, o = j.mimic[1], j.mimic[2]
        return r, m * m2, m * o2 + o

    rows = [resolve(j.name) for j in full]
    act: list[str] = []
    for r, _, _ in rows:
        if r is not None and r not in act:
            act.append(r)
    Na = len(act)
    A = np.zeros((J, Na), dtype=np.float64)
    b = np.zeros(J, dtype=np.float64)
    for i, (r, m, o) in enumerate(rows):
        if r is not None:
            A[i, act.index(r)] = m
        b[i] = o

    lo = np.array([by_name[r].lower for r in act], dtype=np.float64)
    hi = np.array([by_name[r].upper for r in act], dtype=np.float64)
    axis = np.array([j.axis for j in full], dtype=np.float64)
    jtype = np.array([1 if j.type == "prismatic" else 0 for j in full], dtype=np.int64)
    parent = np.array(parent_list, dtype=np.int64)
    T_origin = np.array(T_origin_list, dtype=np.float64)

    # Temporary spec to run FK at canonical open pose
    def _frames_raw(q_act: np.ndarray) -> np.ndarray:
        q = A @ q_act + b
        F = np.zeros((J + 1, 4, 4), dtype=np.float64)
        F[0] = np.eye(4)
        for j in range(J):
            F[j + 1] = F[parent[j] + 1] @ T_origin[j] @ motion_T(int(jtype[j]), axis[j], float(q[j]))
        return F

    q0 = np.clip(0.0, lo, hi)
    F0 = _frames_raw(q0)

    kps: list[tuple[int, np.ndarray]] = []

    def add_kp(frame_idx: int, off) -> int:
        kps.append((frame_idx + 1, np.asarray(off, dtype=np.float64)))
        return len(kps) - 1

    pos0 = lambda fidx, off: (F0[fidx + 1] @ np.append(off, 1.0))[:3]
    chain_dofs = {f: [idx_of[j.name] for j in p if j.name in idx_of] for f, p in paths.items()}

    # Finger polylines
    poly = {}
    for f in tips:
        path = paths[f]
        dofs = chain_dofs[f]
        if f in base_links:
            bl = base_links[f]
            k = [j.child for j in path].index(bl) if bl != palm else -1
            fidx, Trel = link_frame[bl]
            base = (fidx, Trel[:3, 3])
            after = [idx_of[j.name] for j in path[k + 1:] if j.name in idx_of]
        else:
            others = {i for g in tips if g != f for i in chain_dofs[g]}
            uniq = [i for i in dofs if i not in others]
            if not uniq:
                raise ValueError(f"{name}: finger {f} has no unique joint; specify base_links")
            base = (uniq[0], np.zeros(3))
            after = dofs[dofs.index(uniq[0]) + 1:]
        fidx, Trel = link_frame[tips[f]]
        tip = (fidx, (Trel @ np.append(np.asarray(tip_offsets.get(f, [0, 0, 0]), dtype=np.float64), 1.0))[:3])
        pts = [base] + [(i, np.zeros(3)) for i in after] + [tip]
        P = [pos0(*p) for p in pts]
        keep = [0]
        for i in range(1, len(pts) - 1):
            if np.linalg.norm(P[i] - P[keep[-1]]) > DEDUP_TOL:
                keep.append(i)
        if len(keep) > 1 and np.linalg.norm(P[-1] - P[keep[-1]]) < DEDUP_TOL:
            keep.pop()
        keep.append(len(pts) - 1)
        if len(keep) > P_MAX:
            raise ValueError(f"{name}: finger {f} polyline has {len(keep)} > {P_MAX} points")
        poly[f] = [
            (pts[i][0], pts[i][1], tip[0] if i == len(pts) - 1 else (parent[pts[i][0]] if i > 0 else None), P[i])
            for i in keep
        ]

    finger_present = np.array([f in tips for f in FINGERS], dtype=bool)
    finger_len = np.zeros(NF, dtype=np.float64)
    poly_kp = -np.ones((NF, P_MAX), dtype=np.int64)
    poly_s = np.zeros((NF, P_MAX), dtype=np.float64)
    for fi, f in enumerate(FINGERS):
        if f not in poly:
            continue
        P = np.array([p[3] for p in poly[f]])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        finger_len[fi] = float(seg.sum())
        poly_s[fi, :len(P)] = np.concatenate([[0.0], np.cumsum(seg)]) / max(float(seg.sum()), 1e-9)
        for i, (fidx, off, _, _) in enumerate(poly[f]):
            poly_kp[fi, i] = add_kp(fidx, off)

    nonthumb = [fi for fi in range(1, NF) if finger_present[fi]]
    scale = float(np.mean(finger_len[nonthumb])) if nonthumb else float(finger_len[0])

    # Collision spheres along distal 2 segments + tip
    sph_kp, sph_r, sph_finger = [], [], []
    for fi, f in enumerate(FINGERS):
        if f not in poly:
            continue
        pl = poly[f]
        for i in range(max(1, len(pl) - 2), len(pl)):
            sf = pl[i][2]
            Finv = np.linalg.inv(F0[sf + 1])
            a_pt = (Finv @ np.append(pl[i - 1][3], 1.0))[:3]
            c_pt = (Finv @ np.append(pl[i][3], 1.0))[:3]
            for t_val in (0.0, 0.5):
                sph_kp.append(add_kp(sf, (1.0 - t_val) * a_pt + t_val * c_pt))
                sph_r.append(LINK_R * scale)
                sph_finger.append(fi)
        sph_kp.append(poly_kp[fi, len(pl) - 1])
        sph_r.append(TIP_R * scale)
        sph_finger.append(fi)

    # Canonical palm frame (+X toward fingertips, +Z palmar normal, +Y = Z x X so right thumb is -Y)
    tip_p = {fi: poly[f][-1][3] for fi, f in enumerate(FINGERS) if f in poly}
    base_p = {fi: poly[f][0][3] for fi, f in enumerate(FINGERS) if f in poly}
    canon = ann.get("canonical")
    if canon:
        z = unit(np.asarray(canon["z"], dtype=np.float64))
        x = np.asarray(canon["x"], dtype=np.float64)
    else:
        vs = []
        for fi in nonthumb:
            dofs = chain_dofs[FINGERS[fi]]
            for j in dofs:
                a_idx = np.flatnonzero(A[j])
                if len(a_idx) == 0 or jtype[j] == 1:
                    continue
                l2 = sorted([A[j, a_idx[0]] * lo[a_idx[0]] + b[j], A[j, a_idx[0]] * hi[a_idx[0]] + b[j]])
                c_sign = _curl_sign(*l2)
                if c_sign:
                    w = F0[j + 1][:3, :3] @ axis[j]
                    v = c_sign * np.cross(w, tip_p[fi] - F0[j + 1][:3, 3])
                    if np.linalg.norm(v) > 1e-6:
                        vs.append(unit(v))
        if not vs:
            raise ValueError(f"{name}: cannot infer palm normal automatically; provide 'canonical'")
        vs_arr = np.array(vs)
        z = unit(vs_arr.sum(0))
        for _ in range(3):
            keep_mask = vs_arr @ z > math.cos(math.radians(45))
            if keep_mask.any():
                z = unit(vs_arr[keep_mask].sum(0))
        x = np.sum([unit(tip_p[fi] - base_p[fi]) for fi in nonthumb], 0)
    x = unit(x - float(x.dot(z)) * z)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    origin = np.mean([base_p[fi] for fi in nonthumb], 0) if nonthumb else base_p[0]
    T_root = make_T(R.T, -R.T @ origin)
    if finger_present[0]:
        thumb_y = float((R.T @ (tip_p[0] - origin))[1])
        if side == "right" and thumb_y > 0:
            warnings.warn(f"{name}: right-hand thumb lies on +y of inferred palm frame (check hand chirality / canonical axes)")
        elif side == "left" and thumb_y < 0:
            warnings.warn(f"{name}: left-hand thumb lies on -y of inferred palm frame (check hand chirality / canonical axes)")

    # Canonical frames at open pose
    Fc = np.einsum("ij,njk->nik", T_root, F0)
    pc = Fc[1:, :3, 3]
    wc = np.einsum("jik,jk->ji", Fc[1:, :3, :3], axis)
    tipc = {fi: (T_root @ np.append(p, 1.0))[:3] for fi, p in tip_p.items()}
    basec = {fi: (T_root @ np.append(p, 1.0))[:3] for fi, p in base_p.items()}
    on_finger_sets = {fi: set(chain_dofs[FINGERS[fi]]) for fi in tip_p}

    # Tree depth & ancestors over movable joints
    depth = np.zeros(J, dtype=np.int64)
    for j in range(J):
        depth[j] = 0 if parent[j] < 0 else depth[parent[j]] + 1
    anc = [[j] for j in range(J)]
    for j in range(J):
        if parent[j] >= 0:
            anc[j] = [j] + anc[parent[j]]

    # Build static joint descriptors d_j (Section 3.2)
    act_label = np.zeros(Na, dtype=np.int64)
    primary = np.zeros(Na, dtype=np.int64)
    desc = np.zeros((Na, JOINT_DESC_DIM), dtype=np.float64)
    act_mass = np.zeros(Na, dtype=np.float64)
    act_com = np.zeros((Na, 3), dtype=np.float64)
    act_inertia6 = np.zeros((Na, 6), dtype=np.float64)

    for a in range(Na):
        driven = np.flatnonzero(A[:, a])
        primary[a] = idx_of.get(act[a], int(driven[0]) if len(driven) else 0)
        fs = sorted({fi for fi in tip_p for j in driven if j in on_finger_sets[fi]})
        act_label[a] = fs[0] if len(fs) == 1 else OTHER
        j0 = primary[a]
        p_parent = np.zeros(3) if parent[j0] < 0 else pc[parent[j0]]
        delta_p = (pc[j0] - p_parent) / scale

        lab = np.zeros(NF + 1, dtype=np.float64)
        lab[act_label[a]] = 1.0

        # Nominal tip Jacobian sensitivity (includes all mimic-driven joints)
        eff = np.zeros((NF, 3), dtype=np.float64)
        for fi in tip_p:
            for j in driven:
                if j in on_finger_sets[fi]:
                    eff[fi] += A[j, a] * (wc[j] if jtype[j] == 1 else np.cross(wc[j], tipc[fi] - pc[j]) / scale)

        f0 = fs[0] if fs else 0
        chain = chain_dofs[FINGERS[f0]] if fs else []
        k_hop = chain.index(j0) if j0 in chain else 0
        hops_to_palm = k_hop / 4.0
        hops_to_tip = max(len(chain) - k_hop - 1, 0) / 4.0
        vec_to_tip = (tipc[f0] - pc[j0]) / scale if fs else np.zeros(3)
        dist_tip = float(np.linalg.norm(vec_to_tip))

        rng_ = hi[a] - lo[a]
        if jtype[j0] == 1:
            lims = np.array([lo[a], hi[a], rng_, 0.5 * (lo[a] + hi[a])]) / scale
        else:
            lims = np.array([lo[a], hi[a], rng_, 0.5 * (lo[a] + hi[a])])

        # Mimic metadata
        num_mimic_children = max(len(driven) - 1, 0)
        is_mimic_driver = 1.0 if num_mimic_children > 0 else 0.0
        cum_ratio = float(np.abs(A[:, a]).sum())
        mimic_feat = np.array([is_mimic_driver, num_mimic_children / 4.0, cum_ratio])

        # Aggregate inertial properties of primary link + mimic-driven child links in primary joint's canonical frame
        total_m = 0.0
        weighted_com = np.zeros(3, dtype=np.float64)
        total_I = np.zeros((3, 3), dtype=np.float64)
        for j in driven:
            clink = links.get(full[j].child)
            if clink is None:
                continue
            Rj = Fc[j + 1, :3, :3]
            pj = Fc[j + 1, :3, 3]
            c_world = pj + Rj @ clink.com
            c_rel = c_world - pc[j0]
            total_m += clink.mass
            weighted_com += clink.mass * c_rel
            I_rot = _rot_inertia(Rj, clink.inertia6)
            I_mat = np.array([
                [I_rot[0], I_rot[3], I_rot[4]],
                [I_rot[3], I_rot[1], I_rot[5]],
                [I_rot[4], I_rot[5], I_rot[2]],
            ])
            # Parallel axis shift to primary joint origin
            r_sq = float(np.dot(c_rel, c_rel))
            total_I += I_mat + clink.mass * (r_sq * np.eye(3) - np.outer(c_rel, c_rel))

        if total_m > 1e-8:
            eff_com = weighted_com / total_m
        else:
            total_m = 0.015
            eff_com = np.zeros(3)
            total_I = np.eye(3) * 2e-6
        eff_i6 = np.array([
            total_I[0, 0], total_I[1, 1], total_I[2, 2],
            total_I[0, 1], total_I[0, 2], total_I[1, 2],
        ])
        act_mass[a] = total_m
        act_com[a] = eff_com
        act_inertia6[a] = eff_i6
        iner_feat = normalize_inertial(total_m, eff_com, eff_i6, scale)

        j_obj = by_name[act[a]]
        act_feat = np.array([default_kp / 20.0, default_kd / 1.0, min(j_obj.effort, 20.0) / 5.0])

        desc[a] = np.concatenate([
            pc[j0] / scale,                      # 3
            delta_p,                             # 3
            wc[j0],                              # 3
            lims,                                # 4
            [float(jtype[j0] == 0), float(jtype[j0] == 1)],  # 2
            lab,                                 # 6
            vec_to_tip,                          # 3
            [dist_tip],                          # 1
            [hops_to_palm, hops_to_tip],         # 2
            mimic_feat,                          # 3
            eff.ravel(),                         # 15
            iner_feat,                           # 10
            act_feat,                            # 3
        ])

    # Build actuated-joint tree distances & normalized GNN adjacency
    tree_dist = np.zeros((Na, Na), dtype=np.int64)
    raw_adj = np.eye(Na, dtype=np.float64)
    for a in range(Na):
        for c in range(Na):
            i, j = primary[a], primary[c]
            common = set(anc[i]) & set(anc[j])
            lca_depth = max((depth[k] for k in common), default=-1)
            d_ac = int(depth[i] + depth[j] - 2 * lca_depth)
            tree_dist[a, c] = d_ac
            if a != c and d_ac <= 2:
                # Direct parent-child (or separated only by a mimic/fixed node) or sibling on palm
                raw_adj[a, c] = 1.0 if d_ac == 1 else 0.5
    deg = raw_adj.sum(axis=1, keepdims=True)
    adj = raw_adj / np.maximum(deg, 1e-6)

    # Fingertip static descriptors d_f (NF, TIP_DESC_DIM)
    tip_desc = np.zeros((NF, TIP_DESC_DIM), dtype=np.float64)
    for fi in range(NF):
        one_hot = np.zeros(NF, dtype=np.float64)
        one_hot[fi] = 1.0
        if finger_present[fi]:
            dofs_f = chain_dofs[FINGERS[fi]]
            act_affect = sum(1 for a in range(Na) if any(j in on_finger_sets[fi] for j in np.flatnonzero(A[:, a])))
            has_mimic = float(any(np.flatnonzero(A[:, a]).size > 1 for a in range(Na)
                                  if any(j in on_finger_sets[fi] for j in np.flatnonzero(A[:, a]))))
            tip_desc[fi] = np.concatenate([
                one_hot,                         # 5
                tipc[fi] / scale,                # 3
                basec[fi] / scale,               # 3
                [finger_len[fi] / scale],        # 1
                [act_affect / 5.0],              # 1
                [len(dofs_f) / 5.0],             # 1
                [1.0],                           # 1
                [has_mimic],                     # 1
            ])
        else:
            tip_desc[fi, :NF] = one_hot

    on_finger_arr = np.zeros((J, NF), dtype=bool)
    for fi, js in on_finger_sets.items():
        on_finger_arr[sorted(js), fi] = True
    same_finger = (act_label[:, None] == act_label[None, :]) & (act_label[:, None] != OTHER)

    return HandSpec(
        name=name,
        display_name=display_name,
        split=split,
        joint_names=[j.name for j in full],
        act_names=act,
        parent=parent,
        T_origin=T_origin,
        axis=axis,
        jtype=jtype,
        A=A,
        b=b,
        lo=lo,
        hi=hi,
        T_root=T_root,
        kp_frame=np.array([k[0] for k in kps], dtype=np.int64),
        kp_offset=np.array([k[1] for k in kps], dtype=np.float64),
        finger_present=finger_present,
        poly_kp=poly_kp,
        poly_s=poly_s,
        finger_len=finger_len,
        sph_kp=np.array(sph_kp, dtype=np.int64),
        sph_r=np.array(sph_r, dtype=np.float64),
        sph_finger=np.array(sph_finger, dtype=np.int64),
        scale=scale,
        act_label=act_label,
        desc=desc,
        tip_desc=tip_desc,
        adj=adj,
        tree_dist=tree_dist,
        same_finger=same_finger,
        act_mass=act_mass,
        act_com=act_com,
        act_inertia6=act_inertia6,
        primary=primary,
        on_finger=on_finger_arr,
        side=side,
        urdf_path=urdf_path,
        yaml_path=yaml_path,
        info={"palm_link": palm, "fingers": tips, "R_canon": R, "origin": origin, "actuator": act_cfg, "side": side},
    )


def mirror_hand_spec_chirality(spec: HandSpec) -> HandSpec:
    """Exact sagittal-plane reflection isomorphism (`M = diag(1, -1, 1)`) converting left <-> right HandSpec.

    Mathematical identity (zero approximation error):
      For `M = diag(1, -1, 1)` (`M = M^T = M^{-1}`, `det(M) = -1`):
      - Every rigid transform conjugates cleanly into SE(3): `T' = M4 @ T @ M4` (`det(M R M) = +1`)
      - Revolute joint: `M @ R(a, q) @ M = R(-M @ a, q)` -> `axis' = -M @ axis`
      - Prismatic joint: `M @ (q * a) = q * (M @ a)` -> `axis' = +M @ axis`
      - Keypoints satisfy `spec'.keypoints(q) == M @ spec.keypoints(q)` identically for all `q`!
      - Joint names, joint order, joint angles `q`, limits `[lo, hi]`, and mimic `(A, b)` are 100% invariant.
    """
    M = np.diag([1.0, -1.0, 1.0])
    M4 = np.diag([1.0, -1.0, 1.0, 1.0])

    T_root_m = M4 @ spec.T_root @ M4
    T_origin_m = np.einsum("ij,njk,kl->nil", M4, spec.T_origin, M4)
    axis_m = np.empty_like(spec.axis)
    for j in range(spec.J):
        axis_m[j] = (M @ spec.axis[j]) if spec.jtype[j] == 1 else (-M @ spec.axis[j])
    kp_offset_m = spec.kp_offset @ M

    # Transform static joint descriptors d_j (58-dim) consistently with canonical reflection M
    desc_m = spec.desc.copy()
    desc_m[:, 1] *= -1.0                 # pos.y / scale
    desc_m[:, 4] *= -1.0                 # parent_delta.y / scale
    for a in range(spec.Na):
        j0 = int(spec.primary[a])
        if spec.jtype[j0] == 1:
            desc_m[a, 7] *= -1.0         # prismatic canonical axis: +M @ wc
        else:
            desc_m[a, 6] *= -1.0         # revolute canonical axis: -M @ wc = [-wx, +wy, -wz]
            desc_m[a, 8] *= -1.0
    desc_m[:, 22] *= -1.0                # vec_to_tip.y / scale
    for fi in range(NF):
        desc_m[:, 30 + 3 * fi + 1] *= -1.0  # eff[fi].y transforms as M @ eff[fi]

    # Transform inertial slice [45:55]: [log(m), com/scale (3), ixx, iyy, izz, ixy, ixz, iyz]
    act_com_m = spec.act_com @ M
    act_inertia6_m = spec.act_inertia6.copy()
    act_inertia6_m[:, 3] *= -1.0         # ixy -> -ixy under M = diag(1, -1, 1)
    act_inertia6_m[:, 5] *= -1.0         # iyz -> -iyz under M = diag(1, -1, 1)
    for a in range(spec.Na):
        desc_m[a, INERTIAL_SLICE] = normalize_inertial(
            float(spec.act_mass[a]), act_com_m[a], act_inertia6_m[a], spec.scale,
        )

    # Transform fingertip descriptors d_f (16-dim)
    tip_desc_m = spec.tip_desc.copy()
    tip_desc_m[:, 6] *= -1.0             # tipc.y / scale
    tip_desc_m[:, 9] *= -1.0             # basec.y / scale

    new_side = "right" if spec.side == "left" else "left"
    return HandSpec(
        name=spec.name,
        display_name=spec.display_name,
        split=spec.split,
        joint_names=list(spec.joint_names),
        act_names=list(spec.act_names),
        parent=spec.parent.copy(),
        T_origin=T_origin_m,
        axis=axis_m,
        jtype=spec.jtype.copy(),
        A=spec.A.copy(),
        b=spec.b.copy(),
        lo=spec.lo.copy(),
        hi=spec.hi.copy(),
        T_root=T_root_m,
        kp_frame=spec.kp_frame.copy(),
        kp_offset=kp_offset_m,
        finger_present=spec.finger_present.copy(),
        poly_kp=spec.poly_kp.copy(),
        poly_s=spec.poly_s.copy(),
        finger_len=spec.finger_len.copy(),
        sph_kp=spec.sph_kp.copy(),
        sph_r=spec.sph_r.copy(),
        sph_finger=spec.sph_finger.copy(),
        scale=spec.scale,
        act_label=spec.act_label.copy(),
        desc=desc_m,
        tip_desc=tip_desc_m,
        adj=spec.adj.copy(),
        tree_dist=spec.tree_dist.copy(),
        same_finger=spec.same_finger.copy(),
        act_mass=spec.act_mass.copy(),
        act_com=act_com_m,
        act_inertia6=act_inertia6_m,
        primary=spec.primary.copy(),
        on_finger=spec.on_finger.copy(),
        side=new_side,
        urdf_path=spec.urdf_path,
        yaml_path=spec.yaml_path,
        info={**spec.info, "side": new_side, "mirrored_from": spec.side},
    )


def load_hand_spec(yaml_path: str | Path) -> HandSpec:
    yaml_path = Path(yaml_path).resolve()
    ann = yaml.safe_load(yaml_path.read_text())
    urdf = Path(ann["urdf"]).expanduser()
    if not urdf.is_absolute():
        urdf = (yaml_path.parent / urdf).resolve()
    return build_hand_spec(urdf.read_text(), ann, ann.get("name", yaml_path.stem), str(urdf), str(yaml_path))


def load_hands_by_split(
    config_dir: str | Path,
    split: str | None = None,
    side: str | None = "right",
) -> list[HandSpec]:
    config_dir = Path(config_dir)
    specs = []
    for p in sorted(config_dir.glob("*.yaml")):
        spec = load_hand_spec(p)
        if split is not None and spec.split != split:
            continue
        if side is not None and spec.side != side:
            continue
        specs.append(spec)
    return specs
