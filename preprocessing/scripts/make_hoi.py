# -*- coding: utf-8 -*-
"""
简化的 HOI 处理：只在 select_id 帧计算物体 scale/位置，其他帧复用，直接渲染不保存中间文件
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
import smplx
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from task_status import TaskProgress

from hoi_utils import align, get_scene_pcd, get_obj_pcd, get_front_points

_SCRIPT_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()
PROJECT_DIR = _SCRIPT_DIR
UPLOAD_RECORDS_PATH = PROJECT_DIR / "upload_records.json"


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


def _result_pt_path(video_dir: str) -> str:
    p = os.path.join(video_dir, "motion", "result.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"result.pt not found: {p}")
    return p


def _result_hand_pt_path(video_dir: str) -> str:
    return os.path.join(video_dir, "motion", "result_hand.pt")


def _load_depths(video_dir: str) -> np.ndarray:
    depth_path = os.path.join(video_dir, "depth.npy")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"depth.npy not found: {depth_path}")
    return np.load(depth_path)


def _load_mask(video_dir: str, t: int) -> np.ndarray:
    cand = os.path.join(video_dir, "mask_dir", f"{t:05d}.png")
    if not os.path.exists(cand):
        cand0 = os.path.join(video_dir, "mask_dir", "00000.png")
        if not os.path.exists(cand0):
            raise FileNotFoundError(f"Mask not found: {cand} and fallback {cand0} also missing.")
        cand = cand0
    m = Image.open(cand).convert("L")
    m = np.asarray(m)
    return (m == 255).astype(np.uint8)


def apply_transform_to_model(verts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """verts: (N, 3), T: (4, 4). Apply: [v,1] @ T^T"""
    ones = np.ones((verts.shape[0], 1), dtype=verts.dtype)
    hom = np.concatenate([verts, ones], axis=1)
    out = hom @ T.T
    return out[:, :3]


def _mean_pairwise_distance(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    import scipy.spatial.distance
    return float(np.mean(scipy.spatial.distance.cdist(x, x)))


def build_smplx_model() -> Tuple[smplx.body_models.SMPLXLayer, np.ndarray]:
    """Build SMPL-X model."""
    model_type = "smplx"
    model_path = os.environ.get("SMPLX_MODEL") or str(
        PROJECT_DIR / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    )
    layer_arg = {
        "create_global_orient": False,
        "create_body_pose": False,
        "create_left_hand_pose": False,
        "create_right_hand_pose": False,
        "create_jaw_pose": False,
        "create_leye_pose": False,
        "create_reye_pose": False,
        "create_betas": False,
        "create_expression": False,
        "create_transl": False,
    }
    model = smplx.create(
        model_path,
        model_type=model_type,
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
        **layer_arg,
    )
    return model, model.faces


def _select_index_from_select_id(video_dir: str, t_len: int) -> int:
    """Use select_id from select_id.json as frame index; fallback 0."""
    p = os.path.join(video_dir, "select_id.json")
    if not os.path.exists(p):
        return 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        sid = int(d.get("select_id", 0))
        return max(0, min(sid, t_len - 1)) if t_len else 0
    except Exception:
        return 0


def _default_j_regressor_path() -> str:
    """Get default J_regressor path."""
    candidates = []

    # Prefer explicit GVHMR_ROOT from preprocessing/config.sh
    gvhmr_root = os.environ.get("GVHMR_ROOT")
    if gvhmr_root:
        candidates.append(Path(gvhmr_root) / "hmr4d" / "utils" / "body_model" / "smpl_neutral_J_regressor.pt")

    # Common project locations
    project_root = _SCRIPT_DIR.parent.parent  # .../open4dhoi_code
    candidates.extend([
        _SCRIPT_DIR / "J_regressor.pt",
        project_root / "preprocessing" / "J_regressor.pt",
        project_root / "preprocessing" / "third_party" / "GVHMR" / "hmr4d" / "utils" / "body_model" / "smpl_neutral_J_regressor.pt",
        project_root / "GVHMR" / "hmr4d" / "utils" / "body_model" / "smpl_neutral_J_regressor.pt",
    ])

    for p in candidates:
        if p.exists():
            return str(p)
    return ""


def render_global_video(
    verts: torch.Tensor,          # (T, V, 3) on CPU
    faces: np.ndarray,            # (F, 3) int
    output_path: str,
    video_path: str,
    j_regressor_path: str,
):
    """Render global.mp4 video."""
    try:
        from einops import einsum
        import torch.nn.functional as F
        from global_utils.utils import Renderer, get_global_cameras_static, get_ground_params_from_points
        from global_utils.video_io_utils import get_video_lwh, get_writer
        from global_utils.hmr_cam import create_camera_sensor
    except Exception as e:
        raise ImportError(
            f"Visualization modules missing: {repr(e)}\n"
            "Need: einops + global_utils (Renderer / video_io_utils / hmr_cam)."
        )

    def apply_T_on_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...ki,...ji->...jk", T[..., :3, :3], points) + T[..., None, :3, 3]

    def transform_mat(Rm: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if len(Rm.shape) > len(t.shape):
            t = t[..., None]
        return torch.cat([F.pad(Rm, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)

    def compute_T_ayfz2ay(joints: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        t_ayfz2ay = joints[:, 0, :].detach().clone()
        t_ayfz2ay[:, 1] = 0

        RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]
        RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]
        RL_xz = RL_xz_h + RL_xz_s
        I_mask = RL_xz.pow(2).sum(-1) < 1e-4

        x_dir = torch.zeros_like(t_ayfz2ay)
        x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
        y_dir = torch.zeros_like(x_dir)
        y_dir[..., 1] = 1
        z_dir = torch.cross(x_dir, y_dir, dim=-1)
        R_ayfz2ay = torch.stack([x_dir, y_dir, z_dir], dim=-1)
        R_ayfz2ay[I_mask] = torch.eye(3).to(R_ayfz2ay)

        if inverse:
            R_ay2ayfz = R_ayfz2ay.transpose(1, 2)
            t_ay2ayfz = -einsum(R_ayfz2ay, t_ayfz2ay, "b i j , b i -> b j")
            return transform_mat(R_ay2ayfz, t_ay2ayfz)
        else:
            return transform_mat(R_ayfz2ay, t_ayfz2ay)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video.mp4 not found: {video_path}")
    if not os.path.exists(j_regressor_path):
        raise FileNotFoundError(f"J_regressor not found: {j_regressor_path}")

    J_regressor = torch.load(j_regressor_path, map_location="cpu", weights_only=False).double().cuda()

    faces_t = torch.from_numpy(faces.astype(np.int64)).cuda()
    verts = verts.double().cuda()

    human_vertex_count = 10475

    # 检查 J_regressor 维度
    if J_regressor.shape[1] != human_vertex_count:
        smplx_npz_path = Path(os.environ.get("SMPLX_MODEL") or str(
            PROJECT_DIR / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
        ))
        if smplx_npz_path.exists():
            import scipy.sparse
            smplx_data = np.load(str(smplx_npz_path), allow_pickle=True)
            J_regressor_smplx = smplx_data["J_regressor"]
            if scipy.sparse.issparse(J_regressor_smplx):
                J_regressor_smplx = J_regressor_smplx.toarray()
            J_regressor_smplx = torch.from_numpy(J_regressor_smplx[:22, :]).double().cuda()
            if J_regressor_smplx.shape[1] == human_vertex_count:
                J_regressor = J_regressor_smplx
                print(f"[INFO] 使用 SMPL-X J_regressor (22, {human_vertex_count})")
            else:
                raise RuntimeError(
                    f"J_regressor 维度不匹配: 期望 (22, {human_vertex_count}), "
                    f"得到 {J_regressor.shape} 且 SMPL-X 为 {J_regressor_smplx.shape}"
                )
        else:
            raise RuntimeError(
                f"J_regressor 维度不匹配: 期望 (22, {human_vertex_count}), 得到 {J_regressor.shape}, "
                f"且 SMPL-X npz 未找到: {smplx_npz_path}"
            )

    def move_to_start_point_face_z(verts_in: torch.Tensor) -> torch.Tensor:
        verts_in = verts_in.clone()
        human_part = verts_in[:, :human_vertex_count, :]
        offset = einsum(J_regressor, human_part[0], "j v, v i -> j i")[0]
        offset[1] = human_part[:, :, [1]].min()
        verts_in = verts_in - offset
        T_ay2ayfz = compute_T_ayfz2ay(
            einsum(J_regressor, human_part[[0]], "j v, l v i -> l j i"),
            inverse=True
        )
        verts_in = apply_T_on_points(verts_in, T_ay2ayfz)
        return verts_in

    verts_glob = move_to_start_point_face_z(verts)

    human_part = verts_glob[:, :human_vertex_count, :]
    joints_glob = einsum(J_regressor, human_part, "j v, l v i -> l j i")

    global_R, global_T, global_lights = get_global_cameras_static(
        human_part.float().cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)

    renderer = Renderer(width, height, device="cuda", faces=faces_t, K=K)
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], human_part)
    renderer.set_ground(scale * 1.5, cx, cz)

    color = torch.ones(3).float().cuda() * 0.8

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = get_writer(output_path, fps=30, crf=23)

    Tm = int(verts_glob.shape[0])
    render_len = min(length, Tm)

    for i in tqdm(range(render_len), desc="Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(
            verts_glob[[i]].float(),
            color[None],
            cameras,
            global_lights
        )
        writer.write_frame(img)

    writer.close()


def main(args):
    video_dir = args.video_dir
    select_t = int(args.select_index)
    j_regressor_path = args.j_regressor_path
    stride = int(args.stride)
    do_render = getattr(args, "render", False)

    # 1) Load result_hand.pt（若存在）或 result.pt；手部已由 make_hand_sam3d 更新，无需 update_hand_pose
    result_hand_pt = _result_hand_pt_path(video_dir)
    result_pt = _result_pt_path(video_dir)
    if os.path.exists(result_hand_pt):
        output = torch.load(result_hand_pt, map_location="cpu", weights_only=False)
        print(f"[INFO] 使用 result_hand.pt（含手部更新）")
    else:
        output = torch.load(result_pt, map_location="cpu", weights_only=False)

    global_param = output["smpl_params_incam"]
    G_global_param = output["smpl_params_global"]

    # 2) Load depths
    depths = _load_depths(video_dir)

    # 3) SMPL-X model
    model, faces = build_smplx_model()

    # 4) Load object mesh + rotate z180
    obj_path = os.path.join(video_dir, "obj_org.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"obj_org.obj not found: {obj_path}")

    obj_pcd = o3d.io.read_triangle_mesh(obj_path)
    R_z180 = obj_pcd.get_rotation_matrix_from_xyz((0.0, 0.0, np.pi))
    obj_pcd.rotate(R_z180, center=(0, 0, 0))
    obj_pcd.compute_vertex_normals()

    overts_base = np.asarray(obj_pcd.vertices).astype(np.float32)
    ofaces = np.asarray(obj_pcd.triangles).astype(np.int32)

    # 5) Sequence length
    t_len = int(global_param["global_orient"].shape[0])
    if not (0 <= select_t < t_len):
        raise ValueError(f"select_index out of range: {select_t}, valid [0, {t_len-1}]")

    # Stride downsampling
    t_indices = list(range(0, t_len, stride))
    t_len_sub = len(t_indices)
    print(f"[INFO] 降采样: 总帧数 {t_len} -> {t_len_sub} (stride={stride})")

    # 6) 使用 result_hand / result 中的 body_pose、left/right_hand_pose 直接驱动 SMPL-X（无 update_hand_pose）
    left_hand = global_param.get("left_hand_pose")
    right_hand = global_param.get("right_hand_pose")
    if left_hand is None:
        left_hand = torch.zeros((t_len, 45))
    if right_hand is None:
        right_hand = torch.zeros((t_len, 45))

    # 7) Build human vertices for all frames
    print(f"[INFO] 计算人体顶点...")
    h_list: List[np.ndarray] = []
    zero_pose = torch.zeros((1, 3)).float()

    for t in tqdm(range(t_len), desc="SMPL-X verts"):
        params = {
            "global_orient": global_param["global_orient"][t].reshape(1, -1),
            "body_pose": global_param["body_pose"][t].reshape(1, -1).float(),
            "betas": global_param["betas"][t].reshape(1, -1),
            "expression": torch.zeros((1, 10)).float(),
            "left_hand_pose": left_hand[t].reshape(1, -1).float(),
            "right_hand_pose": right_hand[t].reshape(1, -1).float(),
            "jaw_pose": zero_pose,
            "leye_pose": zero_pose,
            "reye_pose": zero_pose,
            "transl": global_param["transl"][t].reshape(1, -1),
        }
        hverts = model(**params).vertices.detach().cpu().numpy()[0].astype(np.float32)
        h_list.append(hverts)

    # 8) 只在 select_t 帧计算物体 scale 和 center
    print(f"[INFO] 在帧 {select_t} 计算物体 scale 和位置...")
    depth_select = depths[select_t]
    obj_mask_select = _load_mask(video_dir, select_t)
    K_select = output["K_fullimg"][select_t]

    scene, _ = get_scene_pcd(depth_select)
    obj_pts, _ = get_obj_pcd(obj_mask_select, depth_select)

    s, b, *_ = align(scene, h_list[select_t], obj_mask_select.shape[0], obj_mask_select.shape[1], K_select)
    obj_pts = (obj_pts - b) / s

    if obj_pts.shape[0] == 0:
        print("[警告] select_t 帧没有检测到物体点，使用默认 scale=1.0")
        scale_value = 1.0
        center_value = np.mean(overts_base, axis=0)
    else:
        # 计算 front points (for stable scale compare)
        try:
            obj_pcd_n = obj_pcd.simplify_quadric_decimation(target_number_of_triangles=512)
            overts_n = np.asarray(obj_pcd_n.vertices).astype(np.float32)
            overts_c = np.asarray(get_front_points(overts_n, obj_pcd)).astype(np.float32)
            if overts_c.ndim != 2 or overts_c.shape[1] != 3:
                raise ValueError("front points shape unexpected")
        except Exception:
            idx = np.random.choice(len(overts_base), min(500, len(overts_base)), replace=False)
            overts_c = overts_base[idx]

        m = min(500, obj_pts.shape[0])
        pick = np.random.choice(obj_pts.shape[0], m, replace=False)
        obj_samp = obj_pts[pick].astype(np.float32)

        dis_b = _mean_pairwise_distance(obj_samp)
        dis_s = _mean_pairwise_distance(overts_c)
        if dis_s < 1e-8:
            raise RuntimeError("degenerate front points, cannot estimate scale")
        scale_value = dis_b / dis_s

        displace = np.mean(obj_samp, axis=0) - np.mean(overts_c * scale_value, axis=0)
        center_value = np.mean(overts_base * scale_value, axis=0) + displace

    print(f"[INFO] 物体 scale: {scale_value:.6f}, center: {center_value}")

    # 10) 保存 obj_poses.json（仅一帧的 scale、t）
    video_path = os.path.join(video_dir, "video.mp4")
    output_dir = os.path.join(video_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    obj_poses_path = os.path.join(output_dir, "obj_poses.json")
    with open(obj_poses_path, "w", encoding="utf-8") as f:
        json.dump({
            "scale": float(scale_value),
            "t": np.asarray(center_value).ravel().tolist(),
        }, f, indent=2)
    print(f"[INFO] 已保存 {obj_poses_path}")

    # 9) 可选：转换到 global 空间并渲染视频
    if do_render:
        print(f"[INFO] 转换到 global 空间并渲染...")
        all_frames: List[torch.Tensor] = []
        combined_faces: Optional[np.ndarray] = None

        for t in tqdm(t_indices, desc="Incam->Global + combine"):
            hverts = h_list[t]
            overts = overts_base.copy()

            # 应用相同的 scale 和 center
            overts = overts * scale_value
            overts = overts - np.mean(overts, axis=0)
            overts = overts + center_value

            # incam -> global 变换
            axis_angle = global_param["global_orient"][t].cpu().numpy()
            R_2ori = R.from_rotvec(axis_angle).as_matrix()

            T_2ori = global_param["transl"][t].cpu().numpy().squeeze()

            transformation_matrix = np.eye(4, dtype=np.float32)
            transformation_matrix[:3, :3] = R_2ori.T.astype(np.float32)
            transformation_matrix[:3, 3] = (-R_2ori.T @ T_2ori).astype(np.float32)

            hverts = apply_transform_to_model(hverts, transformation_matrix)
            overts = apply_transform_to_model(overts, transformation_matrix)

            axis_angle = G_global_param["global_orient"][t].cpu().numpy()
            global_R = R.from_rotvec(axis_angle).as_matrix()
            global_T = G_global_param["transl"][t].cpu().numpy().squeeze()

            transformation_matrix = np.eye(4, dtype=np.float32)
            transformation_matrix[:3, :3] = global_R.astype(np.float32)
            transformation_matrix[:3, 3] = global_T.astype(np.float32)

            hverts = apply_transform_to_model(hverts, transformation_matrix)
            overts = apply_transform_to_model(overts, transformation_matrix)

            if combined_faces is None:
                num_human_verts = hverts.shape[0]
                obj_faces_offset = ofaces.astype(np.int32) + num_human_verts
                combined_faces = np.concatenate([faces.astype(np.int32), obj_faces_offset], axis=0).astype(np.int32)

            combined_verts = np.concatenate([hverts, overts], axis=0).astype(np.float32)
            all_frames.append(torch.from_numpy(combined_verts))

        assert combined_faces is not None

        all_frames_t = torch.stack(all_frames, dim=0)  # (T, V, 3)

        out_mp4 = os.path.join(output_dir, "global.mp4")
        print(f"[INFO] 渲染全局视频: {out_mp4}")

        render_global_video(
            verts=all_frames_t,
            faces=combined_faces,
            output_path=out_mp4,
            video_path=video_path,
            j_regressor_path=j_regressor_path,
        )

        print(f"[OK] 渲染完成: {out_mp4}")
    else:
        print(f"[INFO] 跳过渲染（使用 --render 启用）")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("简化的 HOI 处理：只在 select_id 帧计算物体 pose，直接渲染")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Session directory; if omitted, batch from upload_records (progress=1.6)",
    )
    parser.add_argument(
        "--select_index",
        type=int,
        default=None,
        help="Frame index for scale; if omitted, use select_id.json",
    )
    parser.add_argument(
        "--j_regressor_path",
        type=str,
        default=None,
        help="Path to J_regressor.pt; default: PROJECT_DIR/J_regressor.pt",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Downsample stride: process every Nth frame (default: 4)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="渲染全局视频 global.mp4（默认不渲染，只保存 obj_poses.json）",
    )
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

    j_regressor_path = args.j_regressor_path or _default_j_regressor_path()
    if args.render and (not j_regressor_path or not os.path.exists(j_regressor_path)):
        print(f"[错误] J_regressor 未找到: {j_regressor_path}", file=sys.stderr)
        print("[提示] 当前仅在 --render 模式需要 J_regressor。", file=sys.stderr)
        sys.exit(1)

    # 互斥检查
    if args.daemon and args.video_dir:
        print("错误: --daemon 和 --video_dir 不能同时使用")
        sys.exit(1)

    if args.video_dir:
        video_dir = str(_resolve_path(args.video_dir))
        if not os.path.isdir(video_dir):
            print(f"[错误] 目录不存在: {video_dir}", file=sys.stderr)
            sys.exit(1)
        try:
            load_path = _result_hand_pt_path(video_dir) if os.path.exists(_result_hand_pt_path(video_dir)) else _result_pt_path(video_dir)
            out = torch.load(load_path, map_location="cpu", weights_only=False)
            t_len = int(out["smpl_params_incam"]["global_orient"].shape[0])
        except FileNotFoundError:
            print(f"[错误] result.pt / result_hand.pt 未找到: {video_dir}/motion/", file=sys.stderr)
            sys.exit(1)
        select_t = args.select_index if args.select_index is not None else _select_index_from_select_id(video_dir, t_len)
        main_arg = argparse.Namespace(
            video_dir=video_dir,
            select_index=select_t,
            j_regressor_path=j_regressor_path,
            stride=args.stride,
            render=args.render,
        )
        main(main_arg)
        sys.exit(0)

    # 守护模式
    if args.daemon:
        from daemon_runner import daemon_loop, resolve_path, atomic_update_progress

        TARGET_PROGRESS = 1.6
        NEXT_PROGRESS = 2.0

        def process_batch(records, target_records, context, args, progress_reporter=None):
            for i, rec in enumerate(tqdm(target_records, desc="make_hoi", unit="vid")):
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

                video_dir = str(session_path)
                try:
                    if os.path.exists(_result_hand_pt_path(video_dir)):
                        load_path = _result_hand_pt_path(video_dir)
                    else:
                        load_path = _result_pt_path(video_dir)
                except FileNotFoundError as e:
                    tqdm.write(f"[跳过] {e}")
                    continue

                output = torch.load(load_path, map_location="cpu", weights_only=False)
                t_len = int(output["smpl_params_incam"]["global_orient"].shape[0])
                select_t = _select_index_from_select_id(video_dir, t_len)
                main_arg = argparse.Namespace(
                    video_dir=video_dir,
                    select_index=select_t,
                    j_regressor_path=j_regressor_path,
                    stride=args.stride,
                    render=args.render,
                )
                try:
                    main(main_arg)
                    atomic_update_progress(sf, NEXT_PROGRESS)
                    tqdm.write(f"[完成] {video_name} -> progress={NEXT_PROGRESS}")
                except Exception as e:
                    tqdm.write(f"[错误] {video_name}: {e}")
                    raise

        daemon_loop(
            task_name="make_hoi",
            target_progress=TARGET_PROGRESS,
            next_progress=NEXT_PROGRESS,
            process_batch_fn=process_batch,
            poll_interval=args.poll_interval,
            args=args,
        )
        sys.exit(0)

    # Batch mode: progress 1.6 -> 2.0
    records = _load_records()
    target = [r for r in records if r.get("annotation_progress", 0) == 1.6]
    if not target:
        print("没有找到 progress=1.6 的视频")
        sys.exit(0)

    # 初始化进度上报
    task_progress = TaskProgress("make_hoi")
    task_progress.start(total=len(target), message=f"开始处理 {len(target)} 个视频")

    try:
        for i, rec in enumerate(tqdm(target, desc="make_hoi", unit="vid")):
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
            task_progress.update(i + 1, item_name=video_name, message=f"正在处理: {video_name}")

            video_dir = str(session_path)
            try:
                if os.path.exists(_result_hand_pt_path(video_dir)):
                    load_path = _result_hand_pt_path(video_dir)
                else:
                    load_path = _result_pt_path(video_dir)
            except FileNotFoundError as e:
                tqdm.write(f"[跳过] {e}")
                continue
            output = torch.load(load_path, map_location="cpu", weights_only=False)
            t_len = int(output["smpl_params_incam"]["global_orient"].shape[0])
            select_t = _select_index_from_select_id(video_dir, t_len)
            main_arg = argparse.Namespace(
                video_dir=video_dir,
                select_index=select_t,
                j_regressor_path=j_regressor_path,
                stride=args.stride,
                render=args.render,
            )
            try:
                main(main_arg)
                _update_record_progress(records, sf, 2.0)
                _write_records(records)
            except Exception as e:
                tqdm.write(f"[错误] 处理失败 {video_name}: {e}")
                raise

        # 完成
        task_progress.complete(f"完成！共处理 {len(target)} 个视频")
    except Exception as e:
        task_progress.fail(str(e))
        raise
