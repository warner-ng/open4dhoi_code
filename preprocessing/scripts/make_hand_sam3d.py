# -*- coding: utf-8 -*-
"""
使用 SAM-3D-Body 估计手部姿态，替代 WiLoR 的 make_hand.py
- 保存 SAM3D 原始参数到 motion/sam3d_params.pt，便于调试
- result_hand.pt：手腕+手替换，可选手臂重定向、四元数高斯平滑
"""
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
import cv2
from tqdm import tqdm
import smplx
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

# Add SAM-3D-Body paths (configurable via env var SAM3D_BODY_ROOT)
SAM3D_ROOT = Path(os.environ.get("SAM3D_BODY_ROOT", "/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/reconstruction/sam-3d-body"))
if SAM3D_ROOT.exists():
    sys.path.insert(0, str(SAM3D_ROOT))
    sys.path.insert(0, str(SAM3D_ROOT / "MHR"))
    sys.path.insert(0, str(SAM3D_ROOT / "MHR" / "tools" / "mhr_smpl_conversion"))

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from mhr.mhr import MHR
from conversion import Conversion

_SCRIPT_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()
PROJECT_DIR = _SCRIPT_DIR
UPLOAD_RECORDS_PATH = PROJECT_DIR / "upload_records.json"

# Default paths (can be overridden via args)
DEFAULT_CHECKPOINT = SAM3D_ROOT / "checkpoints" / "sam-3d-body-dinov3" / "model.ckpt"
DEFAULT_MHR_PATH = SAM3D_ROOT / "checkpoints" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
DEFAULT_SMPLX_PATH = Path(os.environ.get("SMPLX_MODEL") or str(
    PROJECT_DIR / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
))


def _resolve_path(p: str) -> Path:
    if not isinstance(p, str) or not p:
        return Path("")
    if p.startswith("./"):
        return (PROJECT_DIR / p[2:]).resolve()
    if p.startswith("tiktok_data/"):
        return (PROJECT_DIR / p).resolve()
    return Path(p).expanduser().resolve()


def _load_records() -> list:
    if not UPLOAD_RECORDS_PATH.exists():
        raise FileNotFoundError(f"upload_records.json not found: {UPLOAD_RECORDS_PATH}")
    with UPLOAD_RECORDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("upload_records.json must be a list")
    return data


def _update_record_progress(records: list, session_folder: str, progress: float) -> None:
    for rec in records:
        if str(rec.get("session_folder", "")) == session_folder:
            rec["annotation_progress"] = progress
            return


def _write_records(records: list) -> None:
    tmp_path = UPLOAD_RECORDS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    tmp_path.replace(UPLOAD_RECORDS_PATH)

# 导入进度上报模块
from task_status import TaskProgress


def _list_frames(frames_dir: Path) -> List[Path]:
    """List all frame images in sorted order."""
    exts = (".png", ".jpg", ".jpeg")
    paths = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    def _key(p: Path):
        try:
            return int(p.stem)
        except Exception:
            return p.stem

    return sorted(paths, key=_key)


def _compute_bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """快速计算 mask 的 bbox。"""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return None
    return np.array([x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()])


# 仅替换手腕 + 手：body_pose 57:60 L_wrist, 60:63 R_wrist
BODY_POSE_WRIST_SLICE = slice(57, 63)
# 整臂 45:63：肩/肘/腕，重定向与平滑用
BODY_POSE_ARM_SLICE = slice(45, 63)
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
ARM_JOINT_INDICES = [16, 17, 18, 19, 20, 21]


def _compute_global_rotations(full_pose: np.ndarray) -> np.ndarray:
    assert full_pose.shape == (22, 3)
    local_rots = R.from_rotvec(full_pose.reshape(-1, 3)).as_matrix().reshape(22, 3, 3)
    global_rots = np.empty_like(local_rots)
    for j, p in enumerate(SMPLX_PARENTS):
        if p == -1:
            global_rots[j] = local_rots[j]
        else:
            global_rots[j] = global_rots[p] @ local_rots[j]
    return global_rots


def _retarget_arm_one_frame(
    gvhmr_full: np.ndarray,
    sam3d_full: np.ndarray,
    align_idx: int = 0,
) -> np.ndarray:
    """手臂重定向：先 Root/Spine 对齐再 IK，避免 GVHMR World vs SAM3D Camera 坐标系混用导致 Monster。"""
    assert gvhmr_full.shape == sam3d_full.shape == (22, 3)
    gvhmr_globals = _compute_global_rotations(gvhmr_full)
    sam3d_globals = _compute_global_rotations(sam3d_full)
    R_gvhmr_ref = gvhmr_globals[align_idx]
    R_sam3d_ref = sam3d_globals[align_idx]
    R_align = R_gvhmr_ref @ R_sam3d_ref.T
    new_full = gvhmr_full.copy()
    for c in ARM_JOINT_INDICES:
        p = SMPLX_PARENTS[c]
        R_child_target = R_align @ sam3d_globals[c]
        R_new_local = gvhmr_globals[p].T @ R_child_target
        new_full[c] = R.from_matrix(R_new_local).as_rotvec()
    return new_full


def _retarget_arm_sequence(
    target_dict: dict,
    gvhmr_dict: dict,
    sam3d: dict,
    valid_indices: List[int],
    align_idx: int = 0,
) -> None:
    """父链用 GVHMR，手臂用 SAM3D（对齐后 IK），写回 target_dict body_pose[45:63]。"""
    body_out = target_dict["body_pose"]
    T = body_out.shape[0]
    if T == 0:
        return
    body_out_np = body_out.cpu().numpy()
    gvhmr_body = gvhmr_dict["body_pose"].detach().cpu().numpy()
    gvhmr_go = gvhmr_dict["global_orient"].detach().cpu().numpy()
    b = sam3d["body_pose"]
    s = sam3d["global_orient"]
    sam_body = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    sam_go = s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
    for bi, fi in enumerate(valid_indices):
        if fi >= T:
            continue
        gvhmr_full = np.zeros((22, 3), dtype=np.float32)
        gvhmr_full[0] = gvhmr_go[fi]
        gvhmr_full[1:] = gvhmr_body[fi].reshape(21, 3)
        sam3d_full = np.zeros((22, 3), dtype=np.float32)
        sam3d_full[0] = sam_go[bi]
        sam3d_full[1:] = sam_body[bi].reshape(21, 3)
        new_full = _retarget_arm_one_frame(gvhmr_full, sam3d_full, align_idx=align_idx)
        body_out_np[fi, BODY_POSE_ARM_SLICE] = new_full[16:22].reshape(-1)
    valid_set = set(valid_indices)
    for fi in range(T):
        if fi in valid_set:
            continue
        nn = min(valid_indices, key=lambda x: abs(x - fi))
        body_out_np[fi, BODY_POSE_ARM_SLICE] = body_out_np[nn, BODY_POSE_ARM_SLICE]
    target_dict["body_pose"] = torch.from_numpy(np.ascontiguousarray(body_out_np)).to(body_out.dtype)


def _smooth_pose_quat_gaussian(pose_vec: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """四元数空间高斯平滑，避免轴角直接滤波的周期跳变。"""
    if pose_vec.shape[0] < 3:
        return pose_vec
    quats = R.from_rotvec(pose_vec).as_quat()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    quats_smooth = gaussian_filter1d(quats, sigma=sigma, axis=0, mode="nearest")
    norms = np.linalg.norm(quats_smooth, axis=1, keepdims=True)
    quats_smooth /= norms + 1e-8
    return R.from_quat(quats_smooth).as_rotvec()


def _smooth_arm(
    params_dict: dict,
    cutoff: float = 0.3,
    slice_: slice = BODY_POSE_ARM_SLICE,
) -> None:
    """四元数高斯平滑上肢；肩/肘强平滑，腕弱平滑。cutoff 越小平滑越强。"""
    bp = params_dict.get("body_pose")
    if bp is None or bp.shape[1] < 63 or bp.shape[0] < 3:
        return
    arr = bp.detach().cpu().numpy()
    arm_data = arr[:, slice_].reshape(arr.shape[0], -1, 3)
    strong_sigma = 2.0 / (cutoff + 0.01) * 0.5
    weak_sigma = 1.0 / (cutoff + 0.01) * 0.5
    T, J, _ = arm_data.shape
    if J == 2:
        for j in range(J):
            arm_data[:, j, :] = _smooth_pose_quat_gaussian(arm_data[:, j, :], sigma=weak_sigma)
    else:
        for j in range(J):
            abs_j = 15 + j
            sigma = weak_sigma if abs_j in (20, 21) else strong_sigma
            arm_data[:, j, :] = _smooth_pose_quat_gaussian(arm_data[:, j, :], sigma=sigma)
    arr[:, slice_] = arm_data.reshape(T, -1)
    params_dict["body_pose"] = torch.from_numpy(np.ascontiguousarray(arr)).float()


def _load_mask_fast(mask_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """快速加载 mask 并计算 bbox，返回 (mask, bbox) 或 None。"""
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask_binary = (mask > 127).astype(np.uint8)
    bbox = _compute_bbox_from_mask(mask_binary)
    if bbox is None:
        return None
    return mask_binary, bbox


def _process_one(
    session_dir: Path,
    estimator: SAM3DBodyEstimator,
    converter: Conversion,
    device: torch.device,
    result_data: Dict,
    checkpoint_path: str,
    mhr_path: str,
    smplx_model_path: str,
    stride: int = 1,
    retarget: bool = False,
    align_idx: int = 0,
    smooth_cutoff: float = 0.0,
    smooth_wrist_only: bool = False,
) -> bool:
    """处理一个视频：保存 sam3d_params.pt；手部处理写出 result_hand.pt（可选重定向+平滑）。"""
    frames_dir = session_dir / "frames"
    human_mask_dir = session_dir / "human_mask_dir"
    result_pt_path = session_dir / "motion" / "result.pt"
    optim_result_pt_path = session_dir / "motion" / "result_hand.pt"

    if not frames_dir.exists():
        tqdm.write(f"[跳过] frames 目录不存在: {session_dir}")
        return False

    if not result_pt_path.exists():
        tqdm.write(f"[跳过] result.pt 不存在: {result_pt_path}")
        return False

    # Load result.pt
    try:
        result_data_local = torch.load(result_pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        result_data_local = torch.load(result_pt_path, map_location="cpu")

    frame_paths = _list_frames(frames_dir)
    num_frames = len(frame_paths)
    if num_frames == 0:
        tqdm.write(f"[跳过] 没有找到帧: {session_dir}")
        return False

    # Stride downsampling
    frame_indices = list(range(0, num_frames, stride))
    tqdm.write(f"[信息] 处理 {len(frame_indices)}/{num_frames} 帧 (stride={stride})")

    # Collect parameters
    all_lbs_model_params = []
    all_identity_coeffs = []
    all_face_expr_coeffs = []
    valid_indices = []

    # 预加载 mask（如果存在）以加速 - 并行加载
    mask_cache = {}
    if human_mask_dir.exists():
        for frame_idx in frame_indices:
            frame_path = frame_paths[frame_idx]
            mask_name = frame_path.stem + ".png"
            mask_path = human_mask_dir / mask_name
            mask_data = _load_mask_fast(mask_path)
            if mask_data is not None:
                mask_cache[frame_idx] = mask_data

    # Process frames - 主要性能瓶颈在 estimator.process_one_image
    # 优化：1) 已用 torch.no_grad() 2) 预加载 mask 3) stride 降采样
    for frame_idx in tqdm(frame_indices, desc="SAM3D", unit="fr", leave=False):
        frame_path = frame_paths[frame_idx]

        # Get mask and bbox from cache
        bboxes = None
        masks_input = None
        if frame_idx in mask_cache:
            mask_binary, bbox = mask_cache[frame_idx]
            bboxes = bbox[None, :]
            masks_input = mask_binary[None, :, :]

        # Run SAM3D - 这是主要瓶颈，每次都要加载图像和推理
        # 如果 SAM3D 支持批量处理，可以进一步优化
        with torch.no_grad():
            outputs = estimator.process_one_image(
                str(frame_path),
                bboxes=bboxes,
                masks=masks_input,
                use_mask=True if masks_input is not None else False,
            )

        if not outputs:
            continue

        output = outputs[0]

        # Collect parameters
        all_lbs_model_params.append(output["mhr_model_params"])
        all_identity_coeffs.append(output["shape_params"])
        all_face_expr_coeffs.append(output["expr_params"])
        valid_indices.append(frame_idx)

    if len(all_lbs_model_params) == 0:
        tqdm.write(f"[警告] 没有成功处理的帧")
        return False

    # Batch conversion MHR → SMPL-X
    # 注意：转换器内部需要梯度进行优化，不能使用 torch.no_grad()
    tqdm.write(f"[信息] 批量转换 {len(all_lbs_model_params)} 帧...")
    mhr_parameters = {
        "lbs_model_params": torch.from_numpy(np.stack(all_lbs_model_params)).to(device).requires_grad_(True),
        "identity_coeffs": torch.from_numpy(np.stack(all_identity_coeffs)).to(device).requires_grad_(True),
        "face_expr_coeffs": torch.from_numpy(np.stack(all_face_expr_coeffs)).to(device).requires_grad_(True),
    }

    conversion_result = converter.convert_mhr2smpl(
        mhr_parameters=mhr_parameters,
        single_identity=False,
        is_tracking=False,
        return_smpl_parameters=True,
        return_smpl_meshes=False,  # 不需要 mesh，节省内存和时间
    )

    smplx_params = conversion_result.result_parameters

    # 保存 SAM3D 原始参数，便于后续调试
    motion_dir = optim_result_pt_path.parent
    motion_dir.mkdir(parents=True, exist_ok=True)
    sam3d_save = {
        "valid_indices": valid_indices,
        "body_pose": smplx_params["body_pose"].detach().cpu().clone(),
        "global_orient": smplx_params["global_orient"].detach().cpu().clone(),
        "left_hand_pose": smplx_params["left_hand_pose"].detach().cpu().clone(),
        "right_hand_pose": smplx_params["right_hand_pose"].detach().cpu().clone(),
    }
    if "betas" in smplx_params:
        sam3d_save["betas"] = smplx_params["betas"].detach().cpu().clone()
    sam3d_pt_path = motion_dir / "sam3d_params.pt"
    torch.save(sam3d_save, sam3d_pt_path)
    tqdm.write(f"[完成] 已保存 SAM3D 参数: {sam3d_pt_path}")

    if "smpl_params_global" not in result_data_local:
        tqdm.write(f"[错误] result.pt 缺少 smpl_params_global")
        return False

    global_params = result_data_local["smpl_params_global"]
    incam_params = result_data_local.get("smpl_params_incam")
    gvhmr_ref = {
        "body_pose": global_params["body_pose"].clone(),
        "global_orient": global_params["global_orient"].clone(),
    }

    def _apply_hand_updates(params_dict: dict, wrists_too: bool) -> None:
        t_len = params_dict["body_pose"].shape[0] if "body_pose" in params_dict else 0
        if "left_hand_pose" not in params_dict:
            params_dict["left_hand_pose"] = torch.zeros((t_len, 45))
        if "right_hand_pose" not in params_dict:
            params_dict["right_hand_pose"] = torch.zeros((t_len, 45))
        for batch_idx, frame_idx in enumerate(valid_indices):
            if frame_idx >= t_len:
                continue
            if wrists_too and "body_pose" in params_dict and params_dict["body_pose"].shape[1] >= 63:
                params_dict["body_pose"][frame_idx, BODY_POSE_WRIST_SLICE] = smplx_params["body_pose"][batch_idx, BODY_POSE_WRIST_SLICE].detach().cpu()
            params_dict["left_hand_pose"][frame_idx] = smplx_params["left_hand_pose"][batch_idx].detach().cpu()
            params_dict["right_hand_pose"][frame_idx] = smplx_params["right_hand_pose"][batch_idx].detach().cpu()
        frame_to_batch = {f: b for b, f in enumerate(valid_indices)}
        valid_set = set(valid_indices)
        for frame_idx in range(t_len):
            if frame_idx in valid_set:
                continue
            nearest_idx = min(valid_indices, key=lambda x: abs(x - frame_idx))
            batch_idx = frame_to_batch[nearest_idx]
            if wrists_too and "body_pose" in params_dict and params_dict["body_pose"].shape[1] >= 63:
                params_dict["body_pose"][frame_idx, BODY_POSE_WRIST_SLICE] = smplx_params["body_pose"][batch_idx, BODY_POSE_WRIST_SLICE].detach().cpu()
            params_dict["left_hand_pose"][frame_idx] = smplx_params["left_hand_pose"][batch_idx].detach().cpu()
            params_dict["right_hand_pose"][frame_idx] = smplx_params["right_hand_pose"][batch_idx].detach().cpu()

    if retarget:
        _retarget_arm_sequence(global_params, gvhmr_ref, sam3d_save, valid_indices, align_idx=align_idx)
        if incam_params is not None and "body_pose" in incam_params:
            incam_params["body_pose"][:, BODY_POSE_ARM_SLICE] = global_params["body_pose"][:, BODY_POSE_ARM_SLICE]
        _apply_hand_updates(global_params, wrists_too=False)
        if incam_params is not None:
            _apply_hand_updates(incam_params, wrists_too=False)
        tqdm.write(f"[信息] 已做手臂重定向 (align_idx={align_idx})")
    else:
        _apply_hand_updates(global_params, wrists_too=True)
        if incam_params is not None:
            _apply_hand_updates(incam_params, wrists_too=True)

    if smooth_cutoff > 0:
        sl = BODY_POSE_WRIST_SLICE if smooth_wrist_only else BODY_POSE_ARM_SLICE
        _smooth_arm(global_params, cutoff=smooth_cutoff, slice_=sl)
        if incam_params is not None:
            _smooth_arm(incam_params, cutoff=smooth_cutoff, slice_=sl)
        tqdm.write(f"[信息] 已平滑 (cutoff={smooth_cutoff})")

    torch.save(result_data_local, optim_result_pt_path)
    tqdm.write(f"[完成] 已保存 result_hand.pt: {optim_result_pt_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="使用 SAM-3D-Body 估计手部姿态")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Session directory; if omitted, batch from upload_records (progress=1.5)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"SAM-3D-Body checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--mhr_path",
        type=str,
        default=None,
        help=f"MHR model path (default: {DEFAULT_MHR_PATH})",
    )
    parser.add_argument(
        "--smplx_path",
        type=str,
        default=None,
        help=f"SMPL-X model path (default: {DEFAULT_SMPLX_PATH})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Downsample stride: process every Nth frame (default: 1)",
    )
    parser.add_argument("--retarget", action="store_true", help="手臂重定向（GVHMR+SAM3D 坐标系对齐）")
    parser.add_argument(
        "--align_idx",
        type=int,
        default=0,
        choices=[0, 3, 9],
        help="重定向对齐基准：0=Root, 3=Spine1, 9=Spine3",
    )
    parser.add_argument(
        "--smooth_cutoff",
        type=float,
        default=0.0,
        help="平滑强度 (0–1)，0 表示不平滑；建议 0.3",
    )
    parser.add_argument("--smooth_wrist_only", action="store_true", help="平滑时仅手腕 [57:63]")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="守护进程模式：持续轮询等待新任务",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="轮询间隔（秒），默认 120",
    )
    args = parser.parse_args()

    # 互斥检查
    if args.daemon and args.video_dir:
        print("错误: --daemon 和 --video_dir 不能同时使用")
        sys.exit(1)

    # Resolve paths
    checkpoint_path = args.checkpoint or str(DEFAULT_CHECKPOINT)
    mhr_path = args.mhr_path or str(DEFAULT_MHR_PATH)
    smplx_model_path = args.smplx_path or str(DEFAULT_SMPLX_PATH)

    if not os.path.exists(checkpoint_path):
        print(f"[错误] Checkpoint 不存在: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(mhr_path):
        print(f"[错误] MHR model 不存在: {mhr_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(smplx_model_path):
        print(f"[错误] SMPL-X model 不存在: {smplx_model_path}", file=sys.stderr)
        sys.exit(1)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[信息] 使用设备: {device}")

    # Load models (once for batch processing)
    print("[信息] 加载 SAM-3D-Body...")
    model, model_cfg = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)

    print("[信息] 加载 MHR model...")
    mhr_model = MHR.from_files(device=device)

    print("[信息] 加载 SMPL-X model...")
    smplx_model = smplx.create(
        smplx_model_path,
        model_type="smplx",
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)

    print("[信息] 设置 Conversion...")
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smplx_model,
        method="pytorch",
        batch_size=64,
    )

    print("[信息] 设置 SAM3DBodyEstimator...")
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
    )

    # Dummy result_data for single video mode
    dummy_result = {"smpl_params_global": {}}

    if args.video_dir:
        # Single video mode
        session_dir = _resolve_path(args.video_dir)
        if not session_dir.exists():
            print(f"[错误] 目录不存在: {session_dir}", file=sys.stderr)
            sys.exit(1)
        success = _process_one(
            session_dir,
            estimator,
            converter,
            device,
            dummy_result,
            checkpoint_path,
            mhr_path,
            smplx_model_path,
            stride=args.stride,
            retarget=args.retarget,
            align_idx=args.align_idx,
            smooth_cutoff=args.smooth_cutoff,
            smooth_wrist_only=args.smooth_wrist_only,
        )
        sys.exit(0 if success else 1)

    # 守护模式
    if args.daemon:
        from daemon_runner import daemon_loop, resolve_path, atomic_update_progress

        TARGET_PROGRESS = 1.5
        NEXT_PROGRESS = 1.6

        def process_batch(records, target_records, context, args, progress_reporter=None):
            for i, rec in enumerate(tqdm(target_records, desc="make_hand_sam3d", unit="vid")):
                sf = rec.get("session_folder", "")
                if not sf:
                    tqdm.write(f"[跳过] 记录缺少 session_folder")
                    continue
                session_path = resolve_path(sf)
                if not session_path.exists():
                    tqdm.write(f"[跳过] 目录不存在: {session_path}")
                    continue
                video_name = rec.get("file_name", Path(sf).name)
                object_category = rec.get("object_category", "未知")
                tqdm.write(f"视频: {video_name} | 物体: {object_category}")

                if progress_reporter:
                    progress_reporter.update(i + 1, item_name=video_name)

                try:
                    success = _process_one(
                        session_path,
                        estimator,
                        converter,
                        device,
                        dummy_result,
                        checkpoint_path,
                        mhr_path,
                        smplx_model_path,
                        stride=args.stride,
                        retarget=args.retarget,
                        align_idx=args.align_idx,
                        smooth_cutoff=args.smooth_cutoff,
                        smooth_wrist_only=args.smooth_wrist_only,
                    )
                    if success:
                        atomic_update_progress(sf, NEXT_PROGRESS)
                        tqdm.write(f"[完成] {video_name} -> progress={NEXT_PROGRESS}")
                except Exception as e:
                    tqdm.write(f"[错误] {video_name}: {e}")
                    raise

        daemon_loop(
            task_name="make_hand_sam3d",
            target_progress=TARGET_PROGRESS,
            next_progress=NEXT_PROGRESS,
            process_batch_fn=process_batch,
            poll_interval=args.poll_interval,
            args=args,
        )
        sys.exit(0)

    # Batch mode: progress 1.5 -> 1.6
    records = _load_records()
    target = [r for r in records if r.get("annotation_progress", 0) == 1.5]
    if not target:
        print("没有找到 progress=1.5 的视频")
        sys.exit(0)

    # 初始化进度上报
    progress = TaskProgress("make_hand_sam3d")
    progress.start(total=len(target), message=f"开始处理 {len(target)} 个视频")

    try:
        for i, rec in enumerate(tqdm(target, desc="make_hand_sam3d", unit="vid")):
            sf = rec.get("session_folder", "")
            if not sf:
                raise RuntimeError("记录缺少 session_folder")
            session_path = _resolve_path(sf)
            if not session_path.exists():
                raise RuntimeError(f"目录不存在: {session_path}")

            video_name = rec.get("file_name", Path(sf).name)
            object_category = rec.get("object_category", "未知")
            tqdm.write(f"视频: {video_name} | 物体: {object_category}")

            # 上报当前进度
            progress.update(i + 1, item_name=video_name, message=f"正在处理: {video_name}")

            try:
                success = _process_one(
                    session_path,
                    estimator,
                    converter,
                    device,
                    dummy_result,
                    checkpoint_path,
                    mhr_path,
                    smplx_model_path,
                    stride=args.stride,
                    retarget=args.retarget,
                    align_idx=args.align_idx,
                    smooth_cutoff=args.smooth_cutoff,
                    smooth_wrist_only=args.smooth_wrist_only,
                )
                if success:
                    _update_record_progress(records, sf, 1.6)
                    _write_records(records)
            except Exception as e:
                tqdm.write(f"[错误] 处理失败 {video_name}: {e}")
                raise

        # 完成
        progress.complete(f"完成！共处理 {len(target)} 个视频")
    except Exception as e:
        progress.fail(str(e))
        raise


if __name__ == "__main__":
    main()
