#!/usr/bin/env python3
"""Standalone Real-Time Manus Glove -> BrainCo Revo 3 Retargeting Visualizer (MuJoCo).

Self-contained portable package: all required model weights, hand assets, hand config,
inference modules, and Manus Python SDK bindings live inside this directory with zero
external workspace dependencies.

Usage:
    # Default: connect directly to Manus USB dongle (Integrated mode)
    ./run.sh

    # Connect to a local MANUS Core daemon instead
    ./run.sh --connection local

    # Use left Manus glove (mirrored to right Revo 3 hand)
    ./run.sh --hand left
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import threading
import time
from typing import Optional
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = BASE_DIR / "models" / "stage2_best.pt"
DEFAULT_HAND_YAML = BASE_DIR / "config" / "revo3_right.yaml"
VENDOR_DIR = BASE_DIR / "vendor"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import mujoco
import mujoco.viewer
import numpy as np
import torch
import trimesh
import yaml

from retarget_core.diff_fk import HandBank, compute_kinematic_feedback
from retarget_core.human_preprocess import (
    FINGER_KP,
    canonicalize,
    extract_human_6tokens,
    mirror_left_to_right,
)
from retarget_core.urdf_standardizer import (
    HandSpec,
    load_hand_spec,
    make_T,
    motion_T,
    parse_urdf,
    rpy_to_R,
)
from retarget_core.urma_retarget_net import OneEuroFilter, URMAHandNet

# Finger colors: thumb(red), index(orange), middle(green), ring(blue), little(purple)
FINGER_COLORS = np.array(
    [
        [0.84, 0.15, 0.16, 0.95],
        [1.00, 0.50, 0.05, 0.95],
        [0.17, 0.63, 0.17, 0.95],
        [0.12, 0.47, 0.71, 0.95],
        [0.58, 0.40, 0.74, 0.95],
    ],
    dtype=np.float32,
)
PALM_EDGES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17)]


# --------------------------------------------------------------------------- Model Loader
def load_urma_model(ckpt_path: Path, device: str) -> tuple[URMAHandNet, dict]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = payload.get("model_cfg", {})
    model = URMAHandNet(
        gnn_dim=int(mcfg.get("gnn_dim", 128)),
        gnn_layers=int(mcfg.get("gnn_layers", 2)),
        d_key=int(mcfg.get("d_key", 64)),
        d_dec=int(mcfg.get("d_dec", 128)),
        d_morph=int(mcfg.get("d_morph", 64)),
        d_tok=int(mcfg.get("d_tok", 64)),
        d_route=int(mcfg.get("d_route", 96)),
        d_core=int(mcfg.get("d_core", 256)),
        core_blocks=int(mcfg.get("core_blocks", 3)),
        dec_hidden=int(mcfg.get("dec_hidden", 192)),
        dec_attn_layers=int(mcfg.get("dec_attn_layers", 2)),
        dec_attn_heads=int(mcfg.get("dec_attn_heads", 4)),
        joint_fb_dim=int(mcfg.get("joint_fb_dim", 0)),
        tip_fb_dim=int(mcfg.get("tip_fb_dim", 3)),
        residual_servo=bool(mcfg.get("residual_servo", False)),
        use_gnn=bool(mcfg.get("use_gnn", True)),
        use_dual_encoders=bool(mcfg.get("use_dual_encoders", True)),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, mcfg


# --------------------------------------------------------------------------- URDF Visual Mesh Loader
def _vec(s: Optional[str], default) -> np.ndarray:
    return np.array([float(v) for v in s.split()], dtype=np.float64) if s else np.array(default, dtype=np.float64)


def _read_mesh(path: Path):
    scene = trimesh.load(path, process=False)
    out = []
    for geom in (scene.dump() if isinstance(scene, trimesh.Scene) else [scene]):
        if not hasattr(geom, "vertices") or not hasattr(geom, "faces") or len(geom.faces) == 0:
            continue
        mat = getattr(geom.visual, "material", None)
        color = None
        if mat is not None and path.suffix.lower() != ".stl":
            mc = getattr(mat, "main_color", None)
            if mc is not None:
                color = np.asarray(mc, dtype=np.float64) / 255.0
        out.append((np.asarray(geom.vertices, dtype=np.float64), np.asarray(geom.faces), color))
    return out


def _resolve_mesh(urdf: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        rel = Path(filename[len("package://") :])
        for base in urdf.parents:
            for cand in (base / rel, base / Path(*rel.parts[1:])):
                if cand.exists():
                    return cand
    p = Path(filename).expanduser()
    return p if p.is_absolute() else (urdf.parent / p).resolve()


def _load_urdf_visuals(urdf: Path):
    root = ET.parse(urdf).getroot()
    named = {m.get("name"): m.find("color") for m in root.findall("material")}
    vis = defaultdict(list)
    for link in root.findall("link"):
        for v in link.findall("visual"):
            o = v.find("origin")
            T = make_T(
                rpy_to_R(_vec(o.get("rpy") if o is not None else None, [0, 0, 0])),
                _vec(o.get("xyz") if o is not None else None, [0, 0, 0]),
            )
            geom_el = v.find("geometry")
            if geom_el is None or len(geom_el) == 0:
                continue
            g = geom_el[0]
            m, c = v.find("material"), None
            if m is not None:
                c = m.find("color") if m.find("color") is not None else named.get(m.get("name"))
            rgba = _vec(c.get("rgba"), [0.72, 0.72, 0.74, 1.0]) if c is not None else None
            if g.tag == "mesh":
                sc = _vec(g.get("scale"), [1, 1, 1])
                mpath = _resolve_mesh(urdf, g.get("filename"))
                if not mpath.exists():
                    continue
                try:
                    parts = [(V * sc, F, pc) for V, F, pc in _read_mesh(mpath)]
                except Exception:
                    continue
            elif g.tag == "box":
                b = trimesh.creation.box(extents=_vec(g.get("size"), [0.01, 0.01, 0.01]))
                parts = [(b.vertices, b.faces, None)]
            elif g.tag == "cylinder":
                b = trimesh.creation.cylinder(radius=float(g.get("radius")), height=float(g.get("length")))
                parts = [(b.vertices, b.faces, None)]
            elif g.tag == "sphere":
                b = trimesh.creation.icosphere(2, radius=float(g.get("radius")))
                parts = [(b.vertices, b.faces, None)]
            else:
                continue
            for V, F, pc in parts:
                vis[link.get("name")].append((V @ T[:3, :3].T + T[:3, 3], F, rgba if rgba is not None else pc))
    return vis


# --------------------------------------------------------------------------- Manus SDK Reader
class ManusSDKReader:
    """Direct Manus Python SDK reader producing MediaPipe 21x3 hand coordinates (m)."""

    def __init__(self, hand: str = "right", connection: str = "integrated"):
        if VENDOR_DIR.is_dir() and str(VENDOR_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_DIR))

        from manus_sdk import ManusSDK, ffi, lib
        from manus_sdk.generated._enums import (
            AxisPolarity,
            AxisView,
            ChainType,
            FingerJointType,
            HandMotion,
            SDKReturnCode,
            Side,
        )

        self._lib = lib
        self._ffi = ffi
        self._sdk = ManusSDK()
        self._enums = dict(
            AxisPolarity=AxisPolarity,
            AxisView=AxisView,
            ChainType=ChainType,
            FingerJointType=FingerJointType,
            HandMotion=HandMotion,
            SDKReturnCode=SDKReturnCode,
            Side=Side,
        )
        self.hand = hand.lower()
        self.connection = connection.lower()
        self.target_side = Side.Left if self.hand == "left" else Side.Right

        # Finger chain mapping matching MediaPipe 21-point topology
        self._finger_chains = [
            (
                ChainType.FingerThumb,
                [
                    FingerJointType.Metacarpal,
                    FingerJointType.Proximal,
                    FingerJointType.Distal,
                    FingerJointType.Tip,
                ],
            ),
            (
                ChainType.FingerIndex,
                [
                    FingerJointType.Proximal,
                    FingerJointType.Intermediate,
                    FingerJointType.Distal,
                    FingerJointType.Tip,
                ],
            ),
            (
                ChainType.FingerMiddle,
                [
                    FingerJointType.Proximal,
                    FingerJointType.Intermediate,
                    FingerJointType.Distal,
                    FingerJointType.Tip,
                ],
            ),
            (
                ChainType.FingerRing,
                [
                    FingerJointType.Proximal,
                    FingerJointType.Intermediate,
                    FingerJointType.Distal,
                    FingerJointType.Tip,
                ],
            ),
            (
                ChainType.FingerPinky,
                [
                    FingerJointType.Proximal,
                    FingerJointType.Intermediate,
                    FingerJointType.Distal,
                    FingerJointType.Tip,
                ],
            ),
        ]

        self._lock = threading.Lock()
        self._latest_pts: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._frame_count: int = 0
        self._node_info_cache: dict[int, list] = {}
        self._glove_side_cache: dict[int, int] = {}
        self._initialized = False

    def start(self, connect_timeout_s: float = 15.0) -> None:
        lib = self._lib
        ffi = self._ffi
        E = self._enums

        if self.connection == "integrated":
            res = lib.CoreSdk_InitializeIntegrated()
        else:
            res = lib.CoreSdk_InitializeCore()
        if res != E["SDKReturnCode"].Success:
            raise RuntimeError(f"CoreSdk_Initialize failed: {E['SDKReturnCode'](res).name}")
        self._initialized = True

        res = self._sdk.register_raw_skeleton_callback(self._on_raw_skeleton)
        if res != E["SDKReturnCode"].Success:
            raise RuntimeError(f"register_raw_skeleton_callback failed: {E['SDKReturnCode'](res).name}")

        coord = ffi.new("CoordinateSystemVUH*")
        coord.view = E["AxisView"].XFromViewer
        coord.up = E["AxisPolarity"].PositiveZ
        coord.handedness = E["Side"].Right
        coord.unitScale = 1.0
        res = lib.CoreSdk_InitializeCoordinateSystemWithVUH(coord[0], True)
        if res != E["SDKReturnCode"].Success:
            raise RuntimeError(f"CoreSdk_InitializeCoordinateSystemWithVUH failed: {E['SDKReturnCode'](res).name}")

        loopback_only = self.connection == "local"
        t_deadline = time.monotonic() + connect_timeout_s
        connected = False
        print(f"[ManusSDK] Connecting ({self.connection})...", flush=True)
        while time.monotonic() < t_deadline:
            if self._try_connect(loopback_only):
                connected = True
                break
            time.sleep(0.5)
        if not connected:
            raise RuntimeError(
                f"Timed out ({connect_timeout_s:.0f}s) connecting to Manus ({self.connection}). "
                "Ensure the Manus USB dongle is plugged in (integrated) or MANUS Core is running (local)."
            )
        print(f"[ManusSDK] Connected ({self.connection}).", flush=True)

        res = lib.CoreSdk_SetRawSkeletonHandMotion(E["HandMotion"].Auto)
        if res != E["SDKReturnCode"].Success:
            print(f"[ManusSDK] Warning: SetRawSkeletonHandMotion returned {E['SDKReturnCode'](res).name}", flush=True)

    def _try_connect(self, loopback_only: bool) -> bool:
        lib = self._lib
        ffi = self._ffi
        ok = self._enums["SDKReturnCode"].Success
        if lib.CoreSdk_LookForHosts(1, loopback_only) != ok:
            return False
        count = ffi.new("uint32_t*")
        if lib.CoreSdk_GetNumberOfAvailableHostsFound(count) != ok or count[0] == 0:
            return False
        hosts = ffi.new("ManusHost[]", count[0])
        if lib.CoreSdk_GetAvailableHostsFound(hosts, count[0]) != ok:
            return False
        return lib.CoreSdk_ConnectToHost(hosts[0]) == ok

    def _get_node_infos(self, glove_id: int, node_count: int) -> Optional[list]:
        cached = self._node_info_cache.get(glove_id)
        if cached is not None and len(cached) == node_count:
            return cached
        arr = self._ffi.new("NodeInfo[]", node_count)
        res = self._lib.CoreSdk_GetRawSkeletonNodeInfoArray(glove_id, arr, node_count)
        if res != self._enums["SDKReturnCode"].Success:
            return None
        infos = [
            (
                int(arr[i].nodeId),
                int(arr[i].side),
                int(arr[i].chainType),
                int(arr[i].fingerJointType),
            )
            for i in range(node_count)
        ]
        self._node_info_cache[glove_id] = infos
        side_val = int(self._enums["Side"].Invalid)
        for _, s, _, _ in infos:
            if s in (int(self._enums["Side"].Left), int(self._enums["Side"].Right)):
                side_val = s
                break
        self._glove_side_cache[glove_id] = side_val
        side_name = (
            "Left"
            if side_val == int(self._enums["Side"].Left)
            else ("Right" if side_val == int(self._enums["Side"].Right) else "Unknown")
        )
        print(f"[ManusSDK] Discovered glove_id={glove_id} side={side_name} nodes={node_count}", flush=True)
        return infos

    def _extract_mediapipe_21(self, infos: list, nodes_array, node_count: int) -> Optional[np.ndarray]:
        E = self._enums
        id_to_pos: dict[int, np.ndarray] = {}
        for i in range(node_count):
            nid = int(nodes_array[i].id)
            p = nodes_array[i].transform.position
            id_to_pos[nid] = np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float32)

        palm_pos = None
        joint_pos: dict[tuple[int, int], np.ndarray] = {}
        for nid, _, ctype, jtype in infos:
            pos = id_to_pos.get(nid)
            if pos is None:
                continue
            if ctype == int(E["ChainType"].Hand) and palm_pos is None:
                palm_pos = pos
            else:
                joint_pos[(ctype, jtype)] = pos

        if palm_pos is None:
            return None

        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = palm_pos
        idx = 1
        for ctype, jtypes in self._finger_chains:
            for jt in jtypes:
                p = joint_pos.get((int(ctype), int(jt)))
                if p is None:
                    return None
                pts[idx] = p
                idx += 1

        if not np.isfinite(pts).all():
            return None
        # Reject uninitialized / degenerate frames (e.g., all-zero startup frames right after glove connect)
        palm_span = float(np.linalg.norm(pts[9] - pts[0]))
        palm_cross = float(np.linalg.norm(np.cross(pts[5] - pts[0], pts[17] - pts[0])))
        if palm_span < 0.02 or palm_cross < 1e-5:
            return None
        return pts

    def _on_raw_skeleton(self, stream_info) -> None:
        lib = self._lib
        ffi = self._ffi
        ok = self._enums["SDKReturnCode"].Success
        try:
            n_skel = int(stream_info.skeletonsCount)
            fallback_pts = None
            selected_pts = None
            for i in range(n_skel):
                info = ffi.new("RawSkeletonInfo*")
                if lib.CoreSdk_GetRawSkeletonInfo(i, info) != ok:
                    continue
                ncount = int(info.nodesCount)
                if ncount <= 0:
                    continue
                nodes = ffi.new("SkeletonNode[]", ncount)
                if lib.CoreSdk_GetRawSkeletonData(i, nodes, ncount) != ok:
                    continue
                glove_id = int(info.gloveId)
                infos = self._get_node_infos(glove_id, ncount)
                if infos is None:
                    continue
                pts = self._extract_mediapipe_21(infos, nodes, ncount)
                if pts is None:
                    continue
                side_val = self._glove_side_cache.get(glove_id, 0)
                if side_val == int(self.target_side):
                    selected_pts = pts
                    break
                if fallback_pts is None:
                    fallback_pts = pts

            chosen = selected_pts if selected_pts is not None else (fallback_pts if n_skel == 1 else None)
            if chosen is not None:
                with self._lock:
                    self._latest_pts = chosen
                    self._latest_ts = time.monotonic()
                    self._frame_count += 1
        except Exception as exc:
            print(f"[ManusSDK] Callback error: {exc}", file=sys.stderr, flush=True)

    def get_latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._latest_pts is None:
                return None, 0.0
            return self._latest_pts.copy(), self._latest_ts

    def stop(self) -> None:
        if self._initialized:
            try:
                self._lib.CoreSdk_ShutDown()
            except Exception:
                pass
            self._initialized = False


# --------------------------------------------------------------------------- Dexformer Retargeter
class DexformerRetargeter:
    """Wraps URMAHandNet for real-time closed-loop single-hand retargeting."""

    def __init__(
        self,
        spec: HandSpec,
        ckpt_path: Path,
        device: str = "cuda:0",
        rate_hz: float = 60.0,
        use_filter: bool = True,
        min_cutoff: float = 2.2,
        beta: float = 0.10,
        mirror_left: bool = False,
    ):
        self.spec = spec
        self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.rate_hz = rate_hz
        self.mirror_left = mirror_left
        self.filt = OneEuroFilter(rate_hz=rate_hz, min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0) if use_filter else None

        self.model, mcfg = load_urma_model(ckpt_path, device=self.device)
        print(
            f"[Dexformer] Loaded {ckpt_path.name} on {self.device} | "
            f"hand={spec.display_name} (Na={spec.Na}, J={spec.J}) | "
            f"dec_attn_layers={mcfg.get('dec_attn_layers', 2)}",
            flush=True,
        )

        self.bank = HandBank([spec], device=self.device)
        self.h1 = self.bank.gather(torch.tensor([0], dtype=torch.long, device=self.device))
        with torch.no_grad():
            self.morph_cache = self.model.encode_morphology(
                self.h1["desc"],
                self.h1["tip_desc"],
                self.h1["adj"],
                self.h1["act_mask"],
                self.h1["finger_present"],
                self.h1["lo"],
                self.h1["hi"],
                self.h1["scale"],
                tree_dist=self.h1["tree_dist"],
                same_finger=self.h1["same_finger"],
            )

        self.g_down = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=self.device)
        self.reset()

    def reset(self) -> None:
        self.q_pad = torch.zeros(1, self.bank.Na, dtype=torch.float32, device=self.device)
        self.q_pad[0, : self.spec.Na] = torch.as_tensor(self.spec.q_open(), dtype=torch.float32, device=self.device)
        self.qd_pad = torch.zeros_like(self.q_pad)
        self.err_pad = torch.zeros_like(self.q_pad)
        if self.filt is not None:
            self.filt.reset()

    @torch.no_grad()
    def step(self, raw_pts_21x3: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Retarget one MediaPipe (21, 3) frame -> (q_act (Na,), canonical_human (21, 3))."""
        if not np.isfinite(raw_pts_21x3).all():
            return self.q_pad[0, : self.spec.Na].cpu().numpy().astype(np.float64), np.zeros((21, 3), dtype=np.float32)
        pts = raw_pts_21x3[None, :, :].astype(np.float32)
        if self.mirror_left:
            pts = mirror_left_to_right(pts)
        pc = canonicalize(pts)  # (1, 21, 3)
        if not np.isfinite(pc).all():
            return self.q_pad[0, : self.spec.Na].cpu().numpy().astype(np.float64), np.zeros((21, 3), dtype=np.float32)
        pc_t = torch.as_tensor(pc, dtype=torch.float32, device=self.device)
        tok = extract_human_6tokens(pc_t)

        tip_prev, jfb, tfb = compute_kinematic_feedback(self.h1, self.q_pad, pc_t)
        q_cmd = self.model.forward_step(
            self.morph_cache,
            tok,
            self.g_down,
            self.q_pad,
            self.qd_pad,
            self.err_pad,
            tip_prev,
            joint_fb=jfb,
            tip_fb=tfb,
        )
        q_raw = q_cmd[0, : self.spec.Na].cpu().numpy()
        if not np.isfinite(q_raw).all():
            return self.q_pad[0, : self.spec.Na].cpu().numpy().astype(np.float64), pc[0]
        if self.filt is not None:
            q_filt = self.filt(q_raw)
            if not np.isfinite(q_filt).all():
                self.filt.reset()
                q_filt = q_raw
            q_out = np.clip(q_filt, self.spec.lo, self.spec.hi)
        else:
            q_out = np.clip(q_raw, self.spec.lo, self.spec.hi)

        q_next = torch.zeros_like(self.q_pad)
        q_next[0, : self.spec.Na] = torch.as_tensor(q_out, dtype=torch.float32, device=self.device)
        safe_dt = max(dt, 1.0 / 240.0)
        self.qd_pad = (q_next - self.q_pad) / safe_dt
        self.err_pad = q_cmd - q_next
        self.q_pad = q_next
        return q_out.astype(np.float64), pc[0]


# --------------------------------------------------------------------------- MuJoCo Revo 3 Scene
class Revo3MuJoCoScene:
    """Builds a MuJoCo visualization scene for Revo 3 from its URDF visual meshes."""

    def __init__(self, spec: HandSpec):
        self.spec = spec
        ann = yaml.safe_load(Path(spec.yaml_path).read_text())
        urdf_path = Path(spec.urdf_path)
        _, joints = parse_urdf(urdf_path.read_text())
        children = defaultdict(list)
        for j in joints:
            children[j.parent].append(j)
        self.by_name = {j.name: j for j in joints}
        self.excl = {k: float(v) for k, v in (ann.get("exclude_joints") or {}).items()}
        self.order: list = []
        stack = [ann["palm_link"]]
        while stack:
            for j in children[stack.pop()]:
                self.order.append(j)
                stack.append(j.child)
        self.links = [ann["palm_link"]] + [j.child for j in self.order]
        vis = _load_urdf_visuals(urdf_path)

        mj_spec = mujoco.MjSpec()
        mj_spec.modelname = "revo3_dexformer_live"
        mj_spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        mj_spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
        mj_spec.visual.headlight.specular = [0.25, 0.25, 0.25]
        mj_spec.stat.extent = 0.5
        mj_spec.add_texture(
            name="sky",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.95, 0.96, 0.98],
            rgb2=[0.88, 0.91, 0.95],
            width=32,
            height=32,
        )
        mj_spec.worldbody.add_light(
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            dir=[-0.4, 0.4, -0.8],
            diffuse=[0.6, 0.6, 0.6],
            specular=[0.35, 0.35, 0.35],
            castshadow=False,
        )
        mj_spec.worldbody.add_light(
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            dir=[0.5, -0.5, -0.6],
            diffuse=[0.35, 0.35, 0.35],
            specular=[0.2, 0.2, 0.2],
            castshadow=False,
        )
        mj_spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0.0, 0.0, -0.20],
            size=[1.0, 1.0, 0.05],
            rgba=[0.90, 0.92, 0.95, 1.0],
            contype=0,
            conaffinity=0,
        )

        # Proper right-handed rotation (det = +1):
        #   world_X = -canonical_Y, world_Y = -canonical_Z (palm facing -Y camera), world_Z = +canonical_X (fingers up)
        self.R_world = np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.T_world = np.eye(4, dtype=np.float64)
        self.T_world[:3, :3] = self.R_world

        for i, link in enumerate(self.links):
            body = mj_spec.worldbody.add_body(name=f"link_{i}_{link}", mocap=True)
            for k, (V, F, rgba) in enumerate(vis.get(link, [])):
                if len(F) < 4:
                    continue
                used, F_idx = np.unique(F, return_inverse=True)
                V_u, F_u = V[used], F_idx.reshape(-1, 3)
                if np.ptp(V_u, axis=0).max() < 1e-6:
                    continue
                mesh = mj_spec.add_mesh(name=f"m_{i}_{k}")
                mesh.uservert = V_u.astype(np.float32).ravel().tolist()
                mesh.userface = F_u.astype(np.int32).ravel().tolist()
                mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                c = np.array([0.72, 0.74, 0.78, 1.0]) if rgba is None else np.asarray(rgba, float)
                c = np.r_[0.18 + 0.78 * c[:3], 1.0]
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh.name,
                    rgba=c,
                    contype=0,
                    conaffinity=0,
                )

        self.model = mj_spec.compile()
        self.data = mujoco.MjData(self.model)
        self.update_hand(spec.q_open())

    def link_poses_world(self, q_act: np.ndarray) -> np.ndarray:
        v = dict(zip(self.spec.joint_names, self.spec.q_full(q_act)))
        v.update(self.excl)

        def val(j):
            if j.name in v:
                return v[j.name]
            if j.mimic is not None and j.mimic[0] in self.by_name:
                return j.mimic[1] * val(self.by_name[j.mimic[0]]) + j.mimic[2]
            return float(np.clip(0.0, j.lower, j.upper))

        jvals = {j.name: val(j) for j in self.order if j.movable}
        T = {self.links[0]: self.T_world @ self.spec.T_root}
        for j in self.order:
            M = motion_T(int(j.type == "prismatic"), j.axis, jvals[j.name]) if j.movable else np.eye(4)
            T[j.child] = T[j.parent] @ j.origin @ M
        return np.stack([T[k] for k in self.links])

    def update_hand(self, q_act: np.ndarray) -> None:
        poses = self.link_poses_world(q_act)
        quat = np.zeros(4, dtype=np.float64)
        for i, T in enumerate(poses):
            self.data.mocap_pos[i] = T[:3, 3]
            mujoco.mju_mat2Quat(quat, T[:3, :3].ravel())
            self.data.mocap_quat[i] = quat
        mujoco.mj_forward(self.model, self.data)

    def draw_overlays(
        self,
        scn: mujoco.MjvScene,
        q_act: np.ndarray,
        pc_human: Optional[np.ndarray],
        show_human: bool = True,
        show_tips: bool = True,
        human_offset_x: float = -0.22,
    ) -> None:
        scn.ngeom = 0

        def add_sphere(pos, rad, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([rad, 0.0, 0.0], dtype=np.float64),
                np.asarray(pos, dtype=np.float64),
                np.eye(3, dtype=np.float64).ravel(),
                np.asarray(rgba, dtype=np.float32),
            )
            scn.ngeom += 1

        def add_capsule(a, b, rad, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).ravel(),
                np.asarray(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                float(rad),
                np.asarray(a, dtype=np.float64),
                np.asarray(b, dtype=np.float64),
            )
            scn.ngeom += 1

        if show_tips:
            tips_c = self.spec.tip_positions(q_act, canonical=True)
            tips_w = tips_c @ self.R_world.T
            for fi in range(5):
                if self.spec.finger_present[fi]:
                    add_sphere(tips_w[fi], 0.0042, FINGER_COLORS[fi])

        if show_human and pc_human is not None:
            pts_w = pc_human.astype(np.float64) @ self.R_world.T
            pts_w[:, 0] += human_offset_x
            for a, b in PALM_EDGES:
                add_capsule(pts_w[a], pts_w[b], 0.0040, [0.35, 0.38, 0.44, 0.90])
            for fi in range(5):
                kp = FINGER_KP[fi]
                col = FINGER_COLORS[fi]
                for j in range(3):
                    add_capsule(pts_w[kp[j]], pts_w[kp[j + 1]], 0.0045, col)
                for j in range(4):
                    add_sphere(pts_w[kp[j]], 0.0068 if j == 3 else 0.0052, col)
            add_sphere(pts_w[0], 0.0065, [0.25, 0.28, 0.32, 1.0])


# --------------------------------------------------------------------------- Main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Manus Glove -> BrainCo Revo 3 Retargeting in MuJoCo")
    p.add_argument(
        "--connection",
        choices=("integrated", "local", "remote"),
        default="integrated",
        help="Manus SDK connection mode (default: integrated, direct USB dongle)",
    )
    p.add_argument("--hand", choices=("right", "left"), default="right", help="Glove hand side (left is mirrored to right)")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT, help="URMA-Hand checkpoint path")
    p.add_argument("--hand-yaml", type=Path, default=DEFAULT_HAND_YAML, help="Hand config YAML path")
    p.add_argument("--device", type=str, default="cuda:0", help="PyTorch device (e.g. cuda:0 or cpu)")
    p.add_argument("--rate", type=float, default=60.0, help="Retargeting & visualization loop rate (Hz)")
    p.add_argument("--no-filter", action="store_true", help="Disable One-Euro output smoothing")
    p.add_argument("--no-human", action="store_true", help="Hide side-by-side human 21-point skeleton")
    p.add_argument("--no-tips", action="store_true", help="Hide fingertip markers on Revo 3")
    p.add_argument("--headless", action="store_true", help="Run without opening MuJoCo GUI (for smoke testing)")
    p.add_argument("--headless-timeout", type=float, default=8.0, help="Max wait seconds in --headless mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_hand_spec(args.hand_yaml)
    retargeter = DexformerRetargeter(
        spec=spec,
        ckpt_path=args.checkpoint,
        device=args.device,
        rate_hz=args.rate,
        use_filter=not args.no_filter,
        mirror_left=(args.hand == "left"),
    )
    scene = Revo3MuJoCoScene(spec)
    reader = ManusSDKReader(hand=args.hand, connection=args.connection)
    reader.start()

    dt_target = 1.0 / max(args.rate, 1.0)
    q_cur = spec.q_open()
    pc_cur: Optional[np.ndarray] = None
    last_step_t = time.monotonic()
    last_log_t = last_step_t
    step_count = 0
    infer_ms_acc = 0.0

    try:
        if args.headless:
            deadline = time.monotonic() + args.headless_timeout
            while time.monotonic() < deadline and step_count < 30:
                pts, ts = reader.get_latest()
                if pts is not None and ts > 0.0:
                    t0 = time.perf_counter()
                    q_cur, pc_cur = retargeter.step(pts, dt_target)
                    infer_ms_acc += (time.perf_counter() - t0) * 1000.0
                    scene.update_hand(q_cur)
                    step_count += 1
                time.sleep(dt_target)
            avg_ms = infer_ms_acc / max(step_count, 1)
            print(
                f"[Headless] Done: steps={step_count}, avg_infer={avg_ms:.2f}ms, "
                f"q_range=[{q_cur.min():.3f}, {q_cur.max():.3f}]",
                flush=True,
            )
            return

        print("[Viewer] Launching MuJoCo viewer (close window or press Ctrl+C to exit)...", flush=True)
        with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
            viewer.cam.lookat[:] = [-0.09 if not args.no_human else 0.0, 0.0, 0.03]
            viewer.cam.distance = 0.55 if not args.no_human else 0.36
            viewer.cam.azimuth = -90.0
            viewer.cam.elevation = -10.0

            while viewer.is_running():
                loop_start = time.monotonic()
                pts, ts = reader.get_latest()
                stale = (ts == 0.0) or ((loop_start - ts) > 0.35)

                if pts is not None and not stale:
                    dt = loop_start - last_step_t
                    last_step_t = loop_start
                    t0 = time.perf_counter()
                    q_cur, pc_cur = retargeter.step(pts, dt)
                    infer_ms_acc += (time.perf_counter() - t0) * 1000.0
                    step_count += 1

                with viewer.lock():
                    scene.update_hand(q_cur)
                    scene.draw_overlays(
                        viewer.user_scn,
                        q_cur,
                        pc_cur,
                        show_human=not args.no_human,
                        show_tips=not args.no_tips,
                    )
                viewer.sync()

                if loop_start - last_log_t >= 2.0:
                    if step_count > 0:
                        fps = step_count / (loop_start - last_log_t)
                        avg_ms = infer_ms_acc / step_count
                        print(
                            f"[Live] {fps:.1f} Hz | infer={avg_ms:.2f} ms | "
                            f"q=[{q_cur.min():.2f}, {q_cur.max():.2f}]",
                            flush=True,
                        )
                    else:
                        print(f"[Live] Waiting for Manus {args.hand} glove data...", flush=True)
                    step_count = 0
                    infer_ms_acc = 0.0
                    last_log_t = loop_start

                sleep_s = dt_target - (time.monotonic() - loop_start)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user.")
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
