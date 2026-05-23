# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm
from rembg import remove
from PIL import Image

from task_status import TaskProgress

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Resolve sam-3d-objects root from ./sam-3d-objects (script dir or cwd)
_SCRIPT_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()
PROJECT_DIR = _SCRIPT_DIR
UPLOAD_RECORDS_PATH = PROJECT_DIR / "upload_records.json"
SELECT_ID_FILENAME = "select_id.json"

SAM3D_ROOT = Path(os.environ.get("SAM3D_OBJ_ROOT", "/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/reconstruction/sam-3d-objects"))


assert SAM3D_ROOT.exists(), (
    f"sam-3d-objects not found.\n"
    f"Expected at: {_SCRIPT_DIR / 'sam-3d-objects'} or {_CWD / 'sam-3d-objects'}\n"
    f"Current script dir: {_SCRIPT_DIR}\n"
    f"Current working dir: {_CWD}"
)

if str(SAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3D_ROOT))

NOTEBOOK_DIR = (SAM3D_ROOT / "notebook").resolve()
assert NOTEBOOK_DIR.exists(), f"notebook dir not found: {NOTEBOOK_DIR}"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from inference import Inference, load_image, load_single_mask
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform


def _resolve_path(p: str) -> Path:
    if not isinstance(p, str) or not p:
        return Path("")
    if p.startswith("./"):
        return (PROJECT_DIR / p[2:]).resolve()
    if p.startswith("tiktok_data/"):
        return (PROJECT_DIR / p).resolve()
    return Path(p).expanduser().resolve()


def _load_records() -> list[dict]:
    if not UPLOAD_RECORDS_PATH.exists():
        raise FileNotFoundError(f"upload_records.json not found: {UPLOAD_RECORDS_PATH}")
    with UPLOAD_RECORDS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("upload_records.json must be a list")
    return data


def _update_record_progress(records: list[dict], session_folder: str, progress: float | int) -> None:
    for rec in records:
        if str(rec.get("session_folder", "")) == session_folder:
            rec["annotation_progress"] = progress
            return


def _write_records(records: list[dict]) -> None:
    tmp_path = UPLOAD_RECORDS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    tmp_path.replace(UPLOAD_RECORDS_PATH)


def _load_select_json(session_dir: Path) -> dict | None:
    p = session_dir / SELECT_ID_FILENAME
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _load_select_id(session_dir: Path) -> int | None:
    """从 select_id.json 读取 select_id（相对 start 的偏移），无效或缺失则返回 None。"""
    data = _load_select_json(session_dir)
    if not data:
        return None
    sid = data.get("select_id")
    if sid is None:
        return None
    try:
        return int(sid)
    except (TypeError, ValueError):
        return None


def _detect_multiview(session_dir: Path) -> bool:
    """检测会话目录是否包含 multiview 文件夹"""
    multiview_dir = session_dir / "multiview"
    return multiview_dir.exists() and multiview_dir.is_dir()


def _get_first_multiview_image(session_dir: Path) -> Path | None:
    """获取 multiview 目录中的第一张图片 (按数字排序)"""
    multiview_dir = session_dir / "multiview"
    if not multiview_dir.exists():
        return None
    images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        images.extend(multiview_dir.glob(f'*{ext}'))
    if not images:
        return None
    def _sort_key(p: Path) -> int:
        try:
            return int(p.stem)
        except ValueError:
            return 999999
    images.sort(key=_sort_key)
    return images[0]


def _generate_mask_with_rembg(image_path: Path) -> np.ndarray:
    """使用 rembg 自动生成 mask"""
    img = Image.open(image_path)
    result = remove(img)
    result_np = np.array(result)
    if result_np.shape[-1] == 4:
        mask = result_np[:, :, 3] > 127
    else:
        mask = np.mean(result_np, axis=-1) > 0
    return mask.astype(bool)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _extract_rgb_from_vertex_attrs(vertex_attrs, n_verts: int):
    if vertex_attrs is None:
        return None
    attrs = _to_numpy(vertex_attrs)
    if attrs.ndim != 2 or attrs.shape[0] != n_verts or attrs.shape[1] < 3:
        return None
    rgb = attrs[:, 0:3].astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)

def _write_obj_with_vcolor(obj_path: str, vertices: np.ndarray, faces: np.ndarray, rgb: np.ndarray | None):
    v = np.asarray(vertices, dtype=np.float64)[:, :3]
    f = np.asarray(faces, dtype=np.int64)[:, :3]

    # normalize 1-based to 0-based if needed
    if f.size > 0 and f.min() >= 1 and f.max() <= v.shape[0]:
        f = f - 1

    with open(obj_path, "w", encoding="utf-8") as fp:
        fp.write("# Generated by make_obj_org_with_color\n")
        if rgb is not None:
            for i in range(v.shape[0]):
                fp.write(
                    f"v {v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f} "
                    f"{rgb[i,0]:.6f} {rgb[i,1]:.6f} {rgb[i,2]:.6f}\n"
                )
        else:
            for i in range(v.shape[0]):
                fp.write(f"v {v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f}\n")

        for i in range(f.shape[0]):
            a, b, c = int(f[i, 0]) + 1, int(f[i, 1]) + 1, int(f[i, 2]) + 1
            fp.write(f"f {a} {b} {c}\n")


def make_scene_untextured_mesh(output):
    mesh = output["mesh"][0]

    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vertex_attrs = getattr(mesh, "vertex_attrs", None)

    rgb = _extract_rgb_from_vertex_attrs(vertex_attrs, n_verts=int(vertices.shape[0]))

    vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
    R_l2c = quaternion_to_matrix(output["rotation"])
    l2c_transform = compose_transform(
        scale=output["scale"],
        rotation=R_l2c,
        translation=output["translation"],
    )
    vertices_world = (
        l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        .squeeze(0)
        .cpu()
        .numpy()
    )

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()
    if rgb is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    return vertices_world, faces, rgb, o3d_mesh


def _process_one(video_dir: Path, config_path: str, select_id: int, use_multiview: bool = False, inference=None) -> None:
    """
    select_id 为相对 start 的帧偏移，即 frames 列表中的索引。
    use_multiview: 如果为 True，从 multiview/ 读取第一张图片并使用 rembg 生成 mask。
    inference: 可选的预加载 Inference 对象，如果为 None 则内部加载。
    """
    if inference is None:
        inference = Inference(config_path, compile=False)

    if use_multiview:
        # multiview 模式：使用 multiview 目录的第一张图片 + rembg 生成 mask
        multiview_image = _get_first_multiview_image(video_dir)
        if multiview_image is None:
            raise RuntimeError(f"multiview 目录为空或不存在: {video_dir / 'multiview'}")

        image = load_image(str(multiview_image))
        mask = _generate_mask_with_rembg(multiview_image)

        # 检查 mask 是否有效
        mask_pixels = np.sum(mask)
        if mask_pixels < 100:
            print(f"[WARN] rembg 生成的 mask 像素数过少 ({mask_pixels})，可能无效", file=sys.stderr)
    else:
        # 原有逻辑：从 frames/ 读取指定帧 + mask_dir 获取 mask
        frame_dir = video_dir / "frames"
        mask_dir = video_dir / "mask_dir"
        frames = sorted([x for x in os.listdir(frame_dir) if x.endswith(".jpg") or x.endswith(".png")])
        if not frames:
            raise RuntimeError(f"frames 为空: {frame_dir}")
        idx = min(max(0, select_id), len(frames) - 1)
        if select_id != idx:
            print(f"[WARN] select_id {select_id} 超出 [0,{len(frames)-1}]，已 clamp 为 {idx}", file=sys.stderr)

        frame_name = frames[idx]
        image_path = frame_dir / frame_name
        image = load_image(str(image_path))
        mask = load_single_mask(str(mask_dir), index=idx, extension=".png")

    output = inference(image, mask, seed=42)
    vertices_world, faces, rgb, o3d_mesh = make_scene_untextured_mesh(output)
    output_path = video_dir / "obj_org.obj"
    _write_obj_with_vcolor(str(output_path), vertices_world, faces, rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--select_id", type=int, default=None, help="单视频时可覆盖 select_id.json 中的 select_id")
    parser.add_argument("--config", type=str, required=False)
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
    tag = "hf-download"
    config_path = "/home/warner/_projects/open4dhoi_code/preprocessing/third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml"
    # 互斥检查
    if args.daemon and args.video_dir:
        print("错误: --daemon 和 --video_dir 不能同时使用")
        sys.exit(1)

    if args.video_dir:
        video_dir = Path(args.video_dir).expanduser().resolve()
        use_multiview = _detect_multiview(video_dir)

        if use_multiview:
            # multiview 模式不需要 select_id
            print(f"[INFO] 检测到 multiview 目录，使用 rembg 生成 mask", file=sys.stderr)
            _process_one(video_dir, config_path, select_id=0, use_multiview=True)
        else:
            # 原有逻辑
            sid = args.select_id if args.select_id is not None else _load_select_id(video_dir)
            if sid is None:
                print("select_id.json 不存在或无效，且未提供 --select_id", file=sys.stderr)
                sys.exit(1)
            _process_one(video_dir, config_path, sid, use_multiview=False)
        sys.exit(0)

    # 守护模式
    if args.daemon:
        from daemon_runner import daemon_loop, resolve_path, atomic_update_progress

        TARGET_PROGRESS = 1.2
        NEXT_PROGRESS = 1.3

        def init_fn(args):
            # 预加载 Inference 模型
            inference = Inference(config_path, compile=False)
            return {"inference": inference, "config_path": config_path}

        def process_batch(records, target_records, context, args, progress_reporter=None):
            inference = context["inference"]

            for i, rec in enumerate(tqdm(target_records, desc="make_obj_org", unit="vid")):
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

                # 检测是否使用 multiview 模式
                use_multiview = _detect_multiview(session_path)

                if use_multiview:
                    sid = 0  # multiview 模式不需要 select_id
                    tqdm.write(f"[INFO] 检测到 multiview 目录: {video_name}")
                else:
                    sid = _load_select_id(session_path)
                    if sid is None:
                        atomic_update_progress(sf, 0)
                        tqdm.write(f"[跳过] select_id.json 不存在或无效: {video_name} ({session_path}) -> progress=0")
                        continue

                tqdm.write(f"视频: {video_name} | 物体: {object_category}")
                if progress_reporter:
                    progress_reporter.update(i + 1, item_name=video_name)
                try:
                    if use_multiview:
                        # multiview 模式：使用 multiview 目录的第一张图片 + rembg 生成 mask
                        multiview_image = _get_first_multiview_image(session_path)
                        if multiview_image is None:
                            raise RuntimeError(f"multiview 目录为空或不存在: {session_path / 'multiview'}")

                        image = load_image(str(multiview_image))
                        mask = _generate_mask_with_rembg(multiview_image)

                        mask_pixels = np.sum(mask)
                        if mask_pixels < 100:
                            tqdm.write(f"[WARN] rembg 生成的 mask 像素数过少 ({mask_pixels})，可能无效")
                    else:
                        # 原有逻辑
                        frame_dir = session_path / "frames"
                        mask_dir = session_path / "mask_dir"
                        frames = sorted([x for x in os.listdir(frame_dir) if x.endswith(".jpg") or x.endswith(".png")])
                        if not frames:
                            raise RuntimeError(f"frames 为空: {frame_dir}")
                        idx = min(max(0, sid), len(frames) - 1)

                        frame_name = frames[idx]
                        image_path = frame_dir / frame_name
                        image = load_image(str(image_path))
                        mask = load_single_mask(str(mask_dir), index=idx, extension=".png")

                    output = inference(image, mask, seed=42)
                    vertices_world, faces, rgb, o3d_mesh = make_scene_untextured_mesh(output)
                    output_path = session_path / "obj_org.obj"
                    _write_obj_with_vcolor(str(output_path), vertices_world, faces, rgb)

                    atomic_update_progress(sf, NEXT_PROGRESS)
                    tqdm.write(f"[完成] {video_name} -> progress={NEXT_PROGRESS}")
                except Exception as e:
                    tqdm.write(f"[错误] {video_name}: {e}")
                    raise

        daemon_loop(
            task_name="make_obj_org",
            target_progress=TARGET_PROGRESS,
            next_progress=NEXT_PROGRESS,
            process_batch_fn=process_batch,
            init_fn=init_fn,
            poll_interval=args.poll_interval,
            args=args,
        )
        sys.exit(0)

    # 原有批量模式
    records = _load_records()
    target = [r for r in records if r.get("annotation_progress", 0) == 1.2]
    if not target:
        sys.exit(0)

    # 初始化进度上报
    progress = TaskProgress("make_obj_org")
    progress.start(total=len(target), message=f"开始处理 {len(target)} 个视频")

    # 预加载模型（只加载一次）
    print("正在加载模型...", file=sys.stderr)
    inference = Inference(config_path, compile=False)
    print("模型加载完成", file=sys.stderr)

    try:
        for i, rec in enumerate(tqdm(target, desc="make_obj_org", unit="vid")):
            sf = rec.get("session_folder", "")
            if not sf:
                raise RuntimeError("记录缺少 session_folder")
            session_path = _resolve_path(sf)
            if not session_path.exists():
                raise RuntimeError(f"目录不存在: {session_path}")
            video_name = rec.get("file_name", Path(sf).name)
            object_category = rec.get("object_category", "未知")

            # 上报当前进度
            progress.update(i + 1, item_name=video_name, message=f"正在处理: {video_name}")

            # 检测是否使用 multiview 模式
            use_multiview = _detect_multiview(session_path)

            if use_multiview:
                sid = 0  # multiview 模式不需要 select_id
                tqdm.write(f"[INFO] 检测到 multiview 目录: {video_name}")
            else:
                sid = _load_select_id(session_path)
                if sid is None:
                    _update_record_progress(records, sf, 0)
                    _write_records(records)
                    tqdm.write(f"[跳过] select_id.json 不存在或无效: {video_name} ({session_path}) -> progress=0")
                    continue

            tqdm.write(f"视频: {video_name} | 物体: {object_category}")
            _process_one(session_path, config_path, sid, use_multiview=use_multiview, inference=inference)
            _update_record_progress(records, sf, 1.3)
            _write_records(records)

        # 完成
        progress.complete(f"完成！共处理 {len(target)} 个视频")
    except Exception as e:
        progress.fail(str(e))
        raise
